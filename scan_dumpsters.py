import os
import sys
import time
import json
import base64
import argparse
import logging
import mimetypes
from typing import Generator, Iterable, Optional, Tuple, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Optional: auto-load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def determine_mime_type(file_path: str) -> str:
    guessed, _ = mimetypes.guess_type(file_path)
    if guessed:
        return guessed
    # default fallback for common extensions
    ext = os.path.splitext(file_path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    return "application/octet-stream"


def encode_image_as_data_url(file_path: str) -> str:
    mime = determine_mime_type(file_path)
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def iter_images(tiles_dir: str, exts: Iterable[str] = (".jpg", ".jpeg", ".png")) -> Generator[str, None, None]:
    exts_lower = {e.lower() for e in exts}
    for root, _dirs, files in os.walk(tiles_dir):
        for name in files:
            if os.path.splitext(name)[1].lower() in exts_lower:
                yield os.path.join(root, name)


def parse_zxy(tiles_dir: str, abs_path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    rel = os.path.relpath(abs_path, tiles_dir)
    parts = rel.split(os.sep)
    if len(parts) < 3:
        return None, None, None
    try:
        z = int(parts[-3])
        x = int(parts[-2])
        y = int(os.path.splitext(parts[-1])[0])
        return z, x, y
    except ValueError:
        return None, None, None


def load_processed_paths(*jsonl_paths: str) -> set:
    processed: set = set()
    for p in jsonl_paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        path = obj.get("path")
                        if path:
                            processed.add(path)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return processed


def ensure_rate(rpm: float, last_call_time: Optional[float]) -> float:
    if rpm <= 0:
        return time.time()
    min_interval = 60.0 / rpm
    now = time.time()
    if last_call_time is not None:
        elapsed = now - last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            now = time.time()
    return now


def build_langchain_llm(model: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        top_p=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
        default_headers={
            "HTTP-Referer": "https://local.script",
            "X-Title": "Dumpster scanner",
        },
    )


def call_openrouter_with_image(
    *,
    llm: ChatOpenAI,
    image_data_url: str,
    min_confidence: float,
    rpm: float,
    last_call_time: Optional[float],
    max_retries: int = 3,
) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
    """Send image using LangChain ChatOpenAI and return parsed JSON result and updated last_call_time.

    Returns (result_dict_or_none, last_call_time, info)
    """
    system_prompt = "Return ONLY JSON. No extra text."
    prompt = (
        "Analyze the image. Respond ONLY with JSON: "
        "{\"dumpster\": true|false, \"confidence\": number between 0 and 1}."
    )

    attempt = 0
    last = last_call_time
    while True:
        attempt += 1
        last = ensure_rate(rpm, last)
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]
                ),
            ]
            ai_msg = llm.invoke(messages)
        except Exception as e:
            if attempt <= max_retries:
                backoff = min(2 ** (attempt - 1), 30)
                time.sleep(backoff + (0.2 * backoff * (0.5 - os.urandom(1)[0] / 255.0)))
                continue
            info = {"variant": None, "error": f"llm_exception: {e.__class__.__name__}: {e}"}
            return None, last, info

        # Extract content
        content_text = ""
        if isinstance(ai_msg.content, str):
            content_text = ai_msg.content
        elif isinstance(ai_msg.content, list):
            parts = []
            for p in ai_msg.content:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            content_text = "\n".join(parts)

        result = extract_json(content_text)
        info = {
            "variant": None,
            "status": getattr(ai_msg, "response_metadata", {}).get("status_code") if hasattr(ai_msg, "response_metadata") else None,
            "response_excerpt": None,
            "content_text_excerpt": (content_text or "")[:400],
            "usage": getattr(ai_msg, "usage_metadata", None),
        }
        return result, last, info


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt to extract the first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan tiles for dumpsters using OpenRouter google/gemini-2.5-pro.")
    ap.add_argument("--tiles_dir", type=str, required=True, help="Root directory of tiles (e.g., tiles/)")
    ap.add_argument("--out", type=str, default="dumpsters.jsonl", help="Output JSONL of positives only.")
    ap.add_argument("--log_all", type=str, default=None, help="Optional JSONL logging all results for auditing/resume.")
    ap.add_argument("--min_confidence", type=float, default=0.5, help="Minimum confidence to treat as positive.")
    ap.add_argument("--rpm", type=float, default=60.0, help="Requests per minute rate limit.")
    ap.add_argument("--model", type=str, default="google/gemini-2.5-pro", help="OpenRouter model id.")
    ap.add_argument("--limit", type=int, default=None, help="Limit the number of images to send.")
    ap.add_argument("--resume", action="store_true", help="Skip files already present in out/log_all.")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug logging to console.")

    args = ap.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY in environment", file=sys.stderr)
        sys.exit(1)

    tiles_dir = os.path.abspath(args.tiles_dir)
    if not os.path.isdir(tiles_dir):
        print(f"tiles_dir not found: {tiles_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.log_all:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_all)) or ".", exist_ok=True)

    processed_paths: set = set()
    if args.resume:
        processed_paths = load_processed_paths(args.out, args.log_all)
        if processed_paths:
            print(f"Resume mode: will skip {len(processed_paths)} already processed images", file=sys.stderr)

    sent = 0
    last_call_time: Optional[float] = None

    # Initialize LangChain LLM once
    llm = build_langchain_llm(args.model, api_key)

    # Files opened lazily on first write
    out_fp = None
    log_all_fp = None

    try:
        for image_path in iter_images(tiles_dir):
            rel_path = os.path.relpath(image_path, tiles_dir)
            if args.resume and rel_path in processed_paths:
                continue

            # Enforce limit before sending
            if args.limit is not None and sent >= args.limit:
                break

            z, x, y = parse_zxy(tiles_dir, image_path)
            try:
                data_url = encode_image_as_data_url(image_path)
            except Exception as e:
                print(f"Skip unreadable image {rel_path}: {e}", file=sys.stderr)
                continue

            result, last_call_time, info = call_openrouter_with_image(
                llm=llm,
                image_data_url=data_url,
                min_confidence=args.min_confidence,
                rpm=args.rpm,
                last_call_time=last_call_time,
            )
            sent += 1

            # Ensure files are open only when needed
            if log_all_fp is None and args.log_all:
                log_all_fp = open(args.log_all, "a", encoding="utf-8")
            if out_fp is None:
                out_fp = open(args.out, "a", encoding="utf-8")

            # Default structure
            record_all: Dict[str, Any] = {
                "path": rel_path,
                "z": z,
                "x": x,
                "y": y,
                "model": args.model,
                "result_raw": result,
                "http_status": info.get("status"),
                "variant": info.get("variant"),
                "response_excerpt": info.get("response_excerpt"),
                "content_text_excerpt": info.get("content_text_excerpt"),
                "error": info.get("error"),
            }

            is_positive = False
            confidence_val: float = 0.0

            if isinstance(result, dict):
                dumpster = result.get("dumpster")
                confidence = result.get("confidence")
                try:
                    confidence_val = float(confidence)
                except (TypeError, ValueError):
                    confidence_val = 0.0
                is_positive = bool(dumpster) and (confidence_val >= args.min_confidence)

            # Write all-results line if requested
            if log_all_fp is not None:
                to_write = record_all.copy()
                to_write["positive"] = is_positive
                to_write["confidence"] = confidence_val
                log_all_fp.write(json.dumps(to_write) + "\n")
                log_all_fp.flush()

            # Write positives-only line
            if is_positive:
                out_line = {
                    "path": rel_path,
                    "z": z,
                    "x": x,
                    "y": y,
                    "confidence": confidence_val,
                    "model": args.model,
                }
                out_fp.write(json.dumps(out_line) + "\n")
                out_fp.flush()

    finally:
        if out_fp is not None:
            out_fp.close()
        if log_all_fp is not None:
            log_all_fp.close()


if __name__ == "__main__":
    main()


