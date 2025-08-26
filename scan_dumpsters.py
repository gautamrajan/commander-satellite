import os
import sys
import time
import json
import base64
import argparse
import logging
import mimetypes
import threading
import queue
from typing import Generator, Iterable, Optional, Tuple, Dict, Any
from io import BytesIO

from PIL import Image as PILImage

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import enhanced prompting module
from prompts import get_enhanced_system_prompt, get_prompt_for_scan_type, add_hard_negatives_to_prompt

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


def encode_pil_image_as_data_url(img: PILImage.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() in {"JPEG", "JPG"}:
        save_kwargs["quality"] = quality
    img.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() in {"JPEG", "JPG"} else f"image/{fmt.lower()}"
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


class RateLimiter:
    """Thread-safe global rate limiter enforcing an average RPM across workers."""

    def __init__(self, rpm: float) -> None:
        self.min_interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next_time:
                    self._next_time = now + self.min_interval
                    return
                sleep_for = self._next_time - now
            time.sleep(min(sleep_for, 0.1))


def _invoke_with_timeout(fn, args: tuple, timeout_sec: float):
    """Run a blocking function in a helper thread and enforce a timeout.

    Returns (ok: bool, result_or_exc: Any). If ok is False and result is TimeoutError, the call timed out.
    """
    result_box = {"done": False, "value": None}

    def runner():
        try:
            result_box["value"] = fn(*args)
        except Exception as e:
            result_box["value"] = e
        finally:
            result_box["done"] = True

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if not result_box["done"]:
        return False, TimeoutError(f"invoke timeout after {timeout_sec}s")
    val = result_box["value"]
    if isinstance(val, Exception):
        return False, val
    return True, val


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
    prompt_text: Optional[str] = None,
    scan_type: str = 'base',
    context_radius: int = 0,
) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
    """Send image using LangChain ChatOpenAI and return parsed JSON result and updated last_call_time.

    Returns (result_dict_or_none, last_call_time, info)
    """
    system_prompt = get_enhanced_system_prompt()
    
    if prompt_text is None:
        prompt = get_prompt_for_scan_type(scan_type, context_radius)
    else:
        prompt = prompt_text

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
            ok, val = _invoke_with_timeout(llm.invoke, (messages,), 60.0)
            if not ok:
                if attempt <= max_retries:
                    backoff = min(2 ** (attempt - 1), 30)
                    time.sleep(backoff)
                    continue
                info = {"variant": None, "error": f"llm_timeout_or_error: {val}"}
                return None, last, info
            ai_msg = val
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


def call_openrouter_with_image_parallel(
    *,
    llm: ChatOpenAI,
    image_data_url: str,
    rate_limiter: RateLimiter,
    max_retries: int = 3,
    prompt_text: Optional[str] = None,
    scan_type: str = 'base',
    context_radius: int = 0,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Parallel-safe variant that relies on a shared RateLimiter (no per-call state)."""
    system_prompt = get_enhanced_system_prompt()
    
    if prompt_text is None:
        prompt = get_prompt_for_scan_type(scan_type, context_radius)
    else:
        prompt = prompt_text

    attempt = 0
    while True:
        attempt += 1
        rate_limiter.wait()
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
            ok, val = _invoke_with_timeout(llm.invoke, (messages,), 60.0)
            if not ok:
                if attempt <= max_retries:
                    backoff = min(2 ** (attempt - 1), 30)
                    time.sleep(backoff)
                    continue
                info = {"variant": None, "error": f"llm_timeout_or_error: {val}"}
                return None, info
            ai_msg = val
        except Exception as e:
            if attempt <= max_retries:
                backoff = min(2 ** (attempt - 1), 30)
                time.sleep(backoff + (0.2 * backoff * (0.5 - os.urandom(1)[0] / 255.0)))
                continue
            info = {"variant": None, "error": f"llm_exception: {e.__class__.__name__}: {e}"}
            return None, info

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
        return result, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan tiles for dumpsters using OpenRouter google/gemini-2.5-pro.")
    ap.add_argument("--tiles_dir", type=str, required=True, help="Root directory of tiles (e.g., tiles/)")
    ap.add_argument("--out", type=str, default="dumpsters.jsonl", help="Output JSONL of positives only.")
    ap.add_argument("--log_all", type=str, default=None, help="Optional JSONL logging all results for auditing/resume.")
    ap.add_argument("--min_confidence", type=float, default=0.5, help="Minimum confidence to treat as positive.")
    ap.add_argument("--rpm", type=float, default=60.0, help="Requests per minute rate limit.")
    ap.add_argument("--concurrency", type=int, default=1, help="Parallel workers for LLM calls (>=1).")
    ap.add_argument("--model", type=str, default="google/gemini-2.5-pro", help="OpenRouter model id.")
    ap.add_argument("--limit", type=int, default=None, help="Limit the number of images to send.")
    ap.add_argument("--resume", action="store_true", help="Skip files already present in out/log_all.")
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug logging to console.")
    ap.add_argument("--context_radius", type=int, default=0, help="Neighborhood radius in tiles. 0 = disabled, 1 = 3x3, 2 = 5x5, ...")
    ap.add_argument("--coarse_factor", type=int, default=0, help="If >1, run a coarse prefilter by stitching factor x factor tiles (e.g., 2 for 2x2).")
    ap.add_argument("--coarse_downscale", type=int, default=256, help="Resize stitched coarse image to this size (square) before sending.")
    ap.add_argument("--coarse_threshold", type=float, default=0.3, help="Confidence threshold for coarse positives that trigger refinement.")
    ap.add_argument("--coarse_log", type=str, default=None, help="Optional JSONL to log coarse-stage results.")
    ap.add_argument("--use_enhanced_prompts", action="store_true", help="Use enhanced prompts with negative examples and confidence calibration.")
    ap.add_argument("--hard_negatives_file", type=str, default=None, help="JSONL file with false positive examples to include as hard negatives in prompts.")

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

    # Load hard negatives if specified
    hard_negatives = []
    if args.hard_negatives_file and os.path.exists(args.hard_negatives_file):
        try:
            with open(args.hard_negatives_file, 'r') as f:
                for line in f:
                    try:
                        neg_example = json.loads(line)
                        if not neg_example.get('approved', True):  # False positives
                            hard_negatives.append(neg_example.get('path', 'Unknown'))
                    except json.JSONDecodeError:
                        continue
            if hard_negatives:
                print(f"Loaded {len(hard_negatives)} hard negative examples for enhanced prompting", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Could not load hard negatives file: {e}", file=sys.stderr)
    
    # If parallel requested, run multi-threaded pipeline and return
    if args.concurrency and args.concurrency > 1:
        limiter = RateLimiter(args.rpm)
        llm = build_langchain_llm(args.model, api_key)
        # Pre-open files and synchronization primitives
        out_fp = open(args.out, "a", encoding="utf-8")
        log_all_fp = open(args.log_all, "a", encoding="utf-8") if args.log_all else None
        coarse_fp = open(args.coarse_log, "a", encoding="utf-8") if args.coarse_log else None
        write_lock = threading.Lock()
        sent_lock = threading.Lock()
        stop_event = threading.Event()
        sent = 0

        def write_jsonl(fp, obj: dict):
            if fp is None:
                return
            with write_lock:
                fp.write(json.dumps(obj) + "\n")
                fp.flush()

        q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=max(2, args.concurrency * 2))

        def process_and_write_record(record_all: Dict[str, Any], result_obj: Optional[Dict[str, Any]]):
            is_positive = False
            confidence_val: float = 0.0
            if isinstance(result_obj, dict):
                dumpster = result_obj.get("dumpster")
                try:
                    confidence_val = float(result_obj.get("confidence"))
                except (TypeError, ValueError):
                    confidence_val = 0.0
                is_positive = bool(dumpster) and (confidence_val >= args.min_confidence)
            if log_all_fp is not None:
                to_write = record_all.copy()
                to_write["positive"] = is_positive
                to_write["confidence"] = confidence_val
                write_jsonl(log_all_fp, to_write)
            if is_positive:
                out_line = {
                    "path": record_all["path"],
                    "z": record_all.get("z"),
                    "x": record_all.get("x"),
                    "y": record_all.get("y"),
                    "confidence": confidence_val,
                    "model": args.model,
                }
                write_jsonl(out_fp, out_line)

        def build_context_if_needed(z, x, y):
            if z is None or x is None or y is None or not args.context_radius or args.context_radius <= 0:
                return None
            r = int(args.context_radius)
            tile_size = 256
            size = (2*r + 1) * tile_size
            canvas = PILImage.new("RGB", (size, size), (255,255,255))
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    px = x + dx
                    py = y + dy
                    neighbor_path = os.path.join(tiles_dir, str(z), str(px), f"{py}.jpg")
                    alt_png_path = os.path.join(tiles_dir, str(z), str(px), f"{py}.png")
                    chosen = neighbor_path if os.path.exists(neighbor_path) else (alt_png_path if os.path.exists(alt_png_path) else None)
                    if not chosen:
                        continue
                    try:
                        img = PILImage.open(chosen).convert("RGB").resize((tile_size, tile_size), PILImage.BILINEAR)
                        canvas.paste(img, ((dx + r) * tile_size, (dy + r) * tile_size))
                    except Exception:
                        pass
            return canvas

        def worker():
            nonlocal sent
            while True:
                task = q.get()
                if task is None:
                    q.task_done(); return
                if stop_event.is_set():
                    q.task_done(); continue
                ttype = task.get("type")
                try:
                    if ttype == "tile":
                        rel_path = task["rel_path"]
                        if args.resume and rel_path in processed_paths:
                            continue
                        # send budget check
                        with sent_lock:
                            if args.limit is not None and sent >= args.limit:
                                stop_event.set()
                                continue
                            sent += 1
                            if args.limit is not None and sent >= args.limit:
                                stop_event.set()
                        # Determine scan type based on context
                        if args.context_radius and args.context_radius > 0 and task.get("z") is not None:
                            canvas = build_context_if_needed(task.get("z"), task.get("x"), task.get("y"))
                            data_url = encode_pil_image_as_data_url(canvas, fmt="JPEG", quality=88) if canvas else encode_image_as_data_url(task["abs_path"])
                            scan_type = 'context'
                        else:
                            data_url = encode_image_as_data_url(task["abs_path"])
                            scan_type = 'base'
                        
                        result_obj, info = call_openrouter_with_image_parallel(
                            llm=llm,
                            image_data_url=data_url,
                            rate_limiter=limiter,
                            scan_type=scan_type,
                            context_radius=args.context_radius or 0,
                        )
                        record_all = {
                            "path": rel_path,
                            "z": task.get("z"),
                            "x": task.get("x"),
                            "y": task.get("y"),
                            "model": args.model,
                            "result_raw": result_obj,
                            "http_status": info.get("status"),
                            "variant": info.get("variant"),
                            "response_excerpt": info.get("response_excerpt"),
                            "content_text_excerpt": info.get("content_text_excerpt"),
                            "error": info.get("error"),
                        }
                        process_and_write_record(record_all, result_obj)
                    elif ttype == "coarse":
                        z = task["z"]; x = task["x"]; y = task["y"]; f = int(task["factor"])
                        # Build coarse block canvas
                        tile_size = 256
                        canvas = PILImage.new("RGB", (f*tile_size, f*tile_size), (255,255,255))
                        for j in range(f):
                            for i in range(f):
                                cx = x + i
                                cy = y + j
                                child_jpg = os.path.join(tiles_dir, str(z), str(cx), f"{cy}.jpg")
                                child_png = os.path.join(tiles_dir, str(z), str(cx), f"{cy}.png")
                                src = child_jpg if os.path.exists(child_jpg) else (child_png if os.path.exists(child_png) else None)
                                if not src:
                                    continue
                                try:
                                    img = PILImage.open(src).convert("RGB").resize((tile_size, tile_size), PILImage.BILINEAR)
                                    canvas.paste(img, (i*tile_size, j*tile_size))
                                except Exception:
                                    pass
                        canvas_to_send = canvas
                        if args.coarse_downscale and args.coarse_downscale > 0:
                            try:
                                ds = int(args.coarse_downscale)
                                canvas_to_send = canvas.resize((ds, ds), PILImage.BILINEAR)
                            except Exception:
                                pass
                        data_url = encode_pil_image_as_data_url(canvas_to_send, fmt="JPEG", quality=85)
                        
                        with sent_lock:
                            if args.limit is not None and sent >= args.limit:
                                stop_event.set(); continue
                            sent += 1
                            if args.limit is not None and sent >= args.limit:
                                stop_event.set()
                        result_obj, info = call_openrouter_with_image_parallel(
                            llm=llm,
                            image_data_url=data_url,
                            rate_limiter=limiter,
                            scan_type='coarse',
                        )
                        coarse_conf = 0.0
                        coarse_pos = False
                        if isinstance(result_obj, dict):
                            try:
                                coarse_conf = float(result_obj.get("confidence"))
                            except (TypeError, ValueError):
                                coarse_conf = 0.0
                            coarse_pos = bool(result_obj.get("dumpster")) and (coarse_conf >= args.coarse_threshold)
                        if coarse_fp is not None:
                            write_jsonl(coarse_fp, {
                                "stage": "coarse",
                                "z": z, "x": x, "y": y,
                                "factor": f,
                                "positive": coarse_pos,
                                "confidence": coarse_conf,
                                "model": args.model,
                                "http_status": info.get("status"),
                                "error": info.get("error"),
                            })
                        if coarse_pos:
                            for j in range(f):
                                for i in range(f):
                                    cx = x + i
                                    cy = y + j
                                    child_rel = os.path.join(str(z), str(cx), f"{cy}.jpg")
                                    if args.resume and child_rel in processed_paths:
                                        continue
                                    child_abs = os.path.join(tiles_dir, child_rel)
                                    q.put({
                                        "type": "tile",
                                        "rel_path": child_rel,
                                        "abs_path": child_abs,
                                        "z": z, "x": cx, "y": cy,
                                    })
                finally:
                    q.task_done()

        # start workers
        threads = []
        for _ in range(int(max(1, args.concurrency))):
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)

        # enqueue tasks
        if args.coarse_factor and args.coarse_factor > 1:
            f = int(args.coarse_factor)
            for image_path in iter_images(tiles_dir):
                if stop_event.is_set():
                    break
                z, x, y = parse_zxy(tiles_dir, image_path)
                if z is None or x is None or y is None:
                    continue
                if (x % f != 0) or (y % f != 0):
                    continue
                q.put({"type": "coarse", "z": z, "x": x, "y": y, "factor": f})
        else:
            for image_path in iter_images(tiles_dir):
                if stop_event.is_set():
                    break
                rel_path = os.path.relpath(image_path, tiles_dir)
                if args.resume and rel_path in processed_paths:
                    continue
                z, x, y = parse_zxy(tiles_dir, image_path)
                q.put({
                    "type": "tile",
                    "rel_path": rel_path,
                    "abs_path": image_path,
                    "z": z, "x": x, "y": y,
                })

        # finalize
        q.join()
        for _ in threads:
            q.put(None)
        for t in threads:
            t.join()
        out_fp.close()
        if log_all_fp is not None:
            log_all_fp.close()
        if coarse_fp is not None:
            coarse_fp.close()
        return

    sent = 0
    last_call_time: Optional[float] = None

    # Initialize LangChain LLM once
    llm = build_langchain_llm(args.model, api_key)

    # Files opened lazily on first write
    out_fp = None
    log_all_fp = None
    coarse_fp = None

    try:
        for image_path in iter_images(tiles_dir):
            rel_path = os.path.relpath(image_path, tiles_dir)
            if args.resume and rel_path in processed_paths:
                continue

            # Enforce limit before sending
            if args.limit is not None and sent >= args.limit:
                break

            z, x, y = parse_zxy(tiles_dir, image_path)

            # Coarse prefilter path
            if args.coarse_factor and args.coarse_factor > 1 and z is not None and x is not None and y is not None:
                f = int(args.coarse_factor)
                # Process only the top-left tile of each fxf block
                if (x % f != 0) or (y % f != 0):
                    continue

                # Stitch fxf tiles
                tile_size = 256
                canvas = PILImage.new("RGB", (f*tile_size, f*tile_size), (255,255,255))
                for j in range(f):
                    for i in range(f):
                        cx = x + i
                        cy = y + j
                        child_jpg = os.path.join(tiles_dir, str(z), str(cx), f"{cy}.jpg")
                        child_png = os.path.join(tiles_dir, str(z), str(cx), f"{cy}.png")
                        src_path = child_jpg if os.path.exists(child_jpg) else (child_png if os.path.exists(child_png) else None)
                        if not src_path:
                            continue
                        try:
                            img = PILImage.open(src_path).convert("RGB").resize((tile_size, tile_size), PILImage.BILINEAR)
                            canvas.paste(img, (i*tile_size, j*tile_size))
                        except Exception:
                            pass

                # Downscale for payload if requested
                if args.coarse_downscale and args.coarse_downscale > 0:
                    ds = int(args.coarse_downscale)
                    canvas_to_send = canvas.resize((ds, ds), PILImage.BILINEAR)
                else:
                    canvas_to_send = canvas

                coarse_data_url = encode_pil_image_as_data_url(canvas_to_send, fmt="JPEG", quality=85)
                
                result, last_call_time, info = call_openrouter_with_image(
                    llm=llm,
                    image_data_url=coarse_data_url,
                    min_confidence=args.min_confidence,
                    rpm=args.rpm,
                    last_call_time=last_call_time,
                    scan_type='coarse',
                )
                sent += 1

                # Lazy-open coarse log
                if coarse_fp is None and args.coarse_log:
                    coarse_fp = open(args.coarse_log, "a", encoding="utf-8")

                coarse_conf: float = 0.0
                coarse_pos = False
                if isinstance(result, dict):
                    try:
                        coarse_conf = float(result.get("confidence"))
                    except (TypeError, ValueError):
                        coarse_conf = 0.0
                    coarse_pos = bool(result.get("dumpster")) and (coarse_conf >= args.coarse_threshold)

                if coarse_fp is not None:
                    coarse_fp.write(json.dumps({
                        "stage": "coarse",
                        "z": z,
                        "x": x,
                        "y": y,
                        "factor": f,
                        "positive": coarse_pos,
                        "confidence": coarse_conf,
                        "model": args.model,
                        "http_status": info.get("status"),
                        "error": info.get("error"),
                    }) + "\n")
                    coarse_fp.flush()

                # Refine on positive: scan children tiles normally (with optional context)
                if coarse_pos:
                    # Ensure files opened
                    if log_all_fp is None and args.log_all:
                        log_all_fp = open(args.log_all, "a", encoding="utf-8")
                    if out_fp is None:
                        out_fp = open(args.out, "a", encoding="utf-8")

                    for j in range(f):
                        for i in range(f):
                            cx = x + i
                            cy = y + j
                            child_rel = os.path.join(str(z), str(cx), f"{cy}.jpg")
                            if args.resume and child_rel in processed_paths:
                                continue

                            # Build input for child with context if requested
                            stitched_data_url: Optional[str] = None
                            scan_type = 'base'
                            
                            if args.context_radius and args.context_radius > 0:
                                r = int(args.context_radius)
                                grid: list[list[Optional[PILImage.Image]]] = []
                                for ddy in range(-r, r+1):
                                    row: list[Optional[PILImage.Image]] = []
                                    for ddx in range(-r, r+1):
                                        px = cx + ddx
                                        py = cy + ddy
                                        neighbor_path = os.path.join(tiles_dir, str(z), str(px), f"{py}.jpg")
                                        alt_png_path = os.path.join(tiles_dir, str(z), str(px), f"{py}.png")
                                        chosen_path = neighbor_path if os.path.exists(neighbor_path) else (alt_png_path if os.path.exists(alt_png_path) else None)
                                        if chosen_path:
                                            try:
                                                row.append(PILImage.open(chosen_path).convert("RGB"))
                                            except Exception:
                                                row.append(None)
                                        else:
                                            row.append(None)
                                    grid.append(row)
                                size = (2*r + 1) * 256
                                canvas_c = PILImage.new("RGB", (size, size), (255,255,255))
                                for jj, row in enumerate(grid):
                                    for ii, img in enumerate(row):
                                        if img is None:
                                            continue
                                        img_r = img.resize((256, 256), PILImage.BILINEAR)
                                        canvas_c.paste(img_r, (ii*256, jj*256))
                                stitched_data_url = encode_pil_image_as_data_url(canvas_c, fmt="JPEG", quality=88)
                                scan_type = 'context'
                            else:
                                child_abs = os.path.join(tiles_dir, child_rel)
                                try:
                                    stitched_data_url = encode_image_as_data_url(child_abs)
                                except Exception as e:
                                    print(f"Skip unreadable image {child_rel}: {e}", file=sys.stderr)
                                    continue

                            result_c, last_call_time, info_c = call_openrouter_with_image(
                                llm=llm,
                                image_data_url=stitched_data_url,
                                min_confidence=args.min_confidence,
                                rpm=args.rpm,
                                last_call_time=last_call_time,
                                scan_type=scan_type,
                                context_radius=args.context_radius or 0,
                            )
                            sent += 1

                            # Write logs for child
                            zc, xc, yc = z, cx, cy
                            record_all: Dict[str, Any] = {
                                "path": os.path.join(str(zc), str(xc), f"{yc}.jpg"),
                                "z": zc,
                                "x": xc,
                                "y": yc,
                                "model": args.model,
                                "result_raw": result_c,
                                "http_status": info_c.get("status"),
                                "variant": info_c.get("variant"),
                                "response_excerpt": info_c.get("response_excerpt"),
                                "content_text_excerpt": info_c.get("content_text_excerpt"),
                                "error": info_c.get("error"),
                            }

                            is_positive = False
                            confidence_val: float = 0.0
                            if isinstance(result_c, dict):
                                dumpster = result_c.get("dumpster")
                                confidence = result_c.get("confidence")
                                try:
                                    confidence_val = float(confidence)
                                except (TypeError, ValueError):
                                    confidence_val = 0.0
                                is_positive = bool(dumpster) and (confidence_val >= args.min_confidence)

                            if log_all_fp is not None:
                                to_write = record_all.copy()
                                to_write["positive"] = is_positive
                                to_write["confidence"] = confidence_val
                                log_all_fp.write(json.dumps(to_write) + "\n")
                                log_all_fp.flush()

                            if is_positive:
                                out_line = {
                                    "path": record_all["path"],
                                    "z": zc,
                                    "x": xc,
                                    "y": yc,
                                    "confidence": confidence_val,
                                    "model": args.model,
                                }
                                out_fp.write(json.dumps(out_line) + "\n")
                                out_fp.flush()

                # Done with this block; move to next file
                continue

            # Regular per-tile scanning path (no coarse prefilter)
            # Build input image: either the single tile or a stitched neighborhood
            stitched_data_url: Optional[str] = None
            scan_type = 'base'
            
            if z is not None and x is not None and y is not None and args.context_radius and args.context_radius > 0:
                # Gather neighbors within [x-r..x+r], [y-r..y+r]
                r = int(args.context_radius)
                grid: list[list[Optional[PILImage.Image]]] = []
                tile_size = 256
                for dy in range(-r, r+1):
                    row: list[Optional[PILImage.Image]] = []
                    for dx in range(-r, r+1):
                        px = x + dx
                        py = y + dy
                        neighbor_path = os.path.join(tiles_dir, str(z), str(px), f"{py}.jpg")
                        alt_png_path = os.path.join(tiles_dir, str(z), str(px), f"{py}.png")
                        chosen_path = None
                        if os.path.exists(neighbor_path):
                            chosen_path = neighbor_path
                        elif os.path.exists(alt_png_path):
                            chosen_path = alt_png_path
                        if chosen_path:
                            try:
                                row.append(PILImage.open(chosen_path).convert("RGB"))
                            except Exception:
                                row.append(None)
                        else:
                            row.append(None)
                    grid.append(row)

                size = (2*r + 1) * tile_size
                canvas = PILImage.new("RGB", (size, size), (255,255,255))
                for j, row in enumerate(grid):
                    for i, img in enumerate(row):
                        if img is None:
                            continue
                        img_r = img.resize((tile_size, tile_size), PILImage.BILINEAR)
                        canvas.paste(img_r, (i*tile_size, j*tile_size))

                stitched_data_url = encode_pil_image_as_data_url(canvas, fmt="JPEG", quality=88)
                scan_type = 'context'
            else:
                try:
                    stitched_data_url = encode_image_as_data_url(image_path)
                except Exception as e:
                    print(f"Skip unreadable image {rel_path}: {e}", file=sys.stderr)
                    continue

            result, last_call_time, info = call_openrouter_with_image(
                llm=llm,
                image_data_url=stitched_data_url,
                min_confidence=args.min_confidence,
                rpm=args.rpm,
                last_call_time=last_call_time,
                scan_type=scan_type,
                context_radius=args.context_radius or 0,
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
        if coarse_fp is not None:
            coarse_fp.close()


if __name__ == "__main__":
    main()


