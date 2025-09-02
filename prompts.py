"""
Enhanced prompting templates for object detection in satellite imagery.
Supports multiple detection types: dumpsters, construction sites, etc.
Designed to reduce false positives based on analysis of common mistakes.
"""

def get_enhanced_system_prompt(detection_type: str = "dumpsters"):
    """System prompt for consistent JSON formatting."""
    if detection_type == "construction":
        return "You are an expert at analyzing satellite imagery for construction site detection and development tracking. Return ONLY valid JSON. No extra text or explanations."
    else:
        return "You are an expert at analyzing satellite imagery for commercial waste management. Return ONLY valid JSON. No extra text or explanations."


def get_base_detection_prompt():
    """Enhanced base prompt with clear dumpster criteria and common false positives."""
    return """Analyze this satellite image to detect commercial dumpsters.

DUMPSTERS are rectangular metal waste containers with these characteristics:
✓ Size: 6-40 cubic yards (larger than cars, smaller than buildings)
✓ Shape: Rectangular/box-like containers
✓ Colors: Usually dark (green, blue, black, brown, gray)
✓ Features: May have visible wheels, legs, or lifting points
✓ Location: Near commercial buildings, loading docks, behind stores
✓ Lids: May be open (showing interior) or closed

DO NOT identify these as dumpsters:
✗ Cars, trucks, SUVs, trailers, or any vehicles
✗ Residential trash cans or small bins
✗ Shadows cast by buildings or structures  
✗ Roof equipment (HVAC units, vents, satellite dishes)
✗ Storage containers, shipping containers, or sheds
✗ Parking spaces, pavement markings, or road features
✗ Construction equipment or machinery
✗ Building features like loading docks themselves

CONFIDENCE SCORING:
- 0.8-1.0: Clearly visible dumpster with multiple identifying features
- 0.6-0.7: Likely dumpster but some ambiguity (lighting, angle, partial view)
- 0.4-0.5: Uncertain - could be dumpster or similar object
- 0.2-0.3: Probably not a dumpster but has some rectangular appearance
- 0.0-0.1: Definitely not a dumpster

Only set dumpster=true if you are confident (≥0.6) AND it matches the criteria above.

Respond ONLY with JSON: {"dumpster": true|false, "confidence": number between 0 and 1}"""


def get_construction_base_detection_prompt():
    """Enhanced base prompt for construction site detection."""
    return """Analyze this satellite image to detect active construction sites.

CONSTRUCTION SITES are areas with active building/development projects with these characteristics:
✓ Excavation: Dirt piles, foundation holes, exposed ground/soil
✓ Equipment: Cranes, bulldozers, excavators, dump trucks, concrete mixers
✓ Materials: Lumber stacks, steel beams, concrete pads, gravel piles
✓ Structures: Temporary buildings, scaffolding, construction fencing
✓ Activity: Raw/disturbed earth, building frames under construction
✓ Access: Construction vehicle access roads, temporary parking areas

DO NOT identify these as construction sites:
✗ Completed buildings or parking lots
✗ Agricultural fields or farmland
✗ Landscaping or garden projects
✗ Road maintenance or small repairs
✗ Empty lots without construction activity
✗ Warehouses, industrial facilities, or storage yards
✗ Quarries or mining operations (unless building construction visible)
✗ Sports fields or recreational areas under maintenance

CONFIDENCE SCORING:
- 0.8-1.0: Clear construction activity with multiple identifying features (equipment + materials + building progress)
- 0.6-0.7: Likely construction site with some visible indicators (equipment OR significant site preparation)
- 0.4-0.5: Uncertain - could be construction or land development
- 0.2-0.3: Possibly construction-related but ambiguous (disturbed ground only)
- 0.0-0.1: Definitely not an active construction site

Only set construction_site=true if you are confident (≥0.6) AND it matches the criteria above.

Respond ONLY with JSON: {"construction_site": true|false, "confidence": number between 0 and 1}"""


def get_context_aware_prompt():
    """Enhanced prompt for context-aware scanning with tile grids."""
    return """Analyze this stitched grid of satellite map tiles for commercial dumpsters.

IMPORTANT: Focus ONLY on the CENTRAL tile region when making your decision. The surrounding tiles provide context about the area type (commercial vs residential vs industrial).

The central tile is typically outlined or highlighted. Look for dumpsters ONLY within this central area.

CONTEXT CLUES from surrounding tiles:
- Commercial/industrial areas: More likely to have dumpsters
- Residential areas: Less likely to have commercial dumpsters
- Parking lots and loading areas: Common dumpster locations
- Building density and type: Helps distinguish business vs residential

DUMPSTER IDENTIFICATION (in central tile only):
✓ Rectangular metal containers, 6-40 cubic yards
✓ Dark colors: green, blue, black, brown, gray
✓ Near commercial buildings or loading areas
✓ May have wheels, legs, or lifting mechanisms
✓ Lids may be open (showing dark interior) or closed

COMMON FALSE POSITIVES to avoid:
✗ Vehicles (cars, trucks, trailers)
✗ Shadows from buildings or structures
✗ Roof equipment (HVAC, vents)
✗ Storage sheds or small buildings
✗ Residential trash cans
✗ Construction equipment

CONFIDENCE GUIDELINES:
- 0.8-1.0: Clear dumpster visible in central tile with context support
- 0.6-0.7: Likely dumpster in central tile, some ambiguity
- 0.4-0.5: Uncertain object in central tile
- 0.2-0.3: Probably not a dumpster
- 0.0-0.1: Definitely not a dumpster

Respond ONLY with JSON: {"dumpster": true|false, "confidence": number between 0 and 1}"""


def get_construction_context_aware_prompt():
    """Enhanced prompt for context-aware construction site scanning with tile grids."""
    return """Analyze this stitched grid of satellite map tiles for active construction sites.

IMPORTANT: Focus ONLY on the CENTRAL tile region when making your decision. The surrounding tiles provide context about the area type and development patterns.

The central tile is typically outlined or highlighted. Look for construction sites ONLY within this central area.

CONTEXT CLUES from surrounding tiles:
- Urban/suburban areas: More likely to have residential/commercial construction
- Industrial areas: May have large-scale construction or facility expansion
- Rural areas: May have new development or infrastructure projects
- Existing development density: Helps identify infill construction vs. greenfield development

CONSTRUCTION SITE IDENTIFICATION (in central tile only):
✓ Active excavation or site preparation
✓ Construction equipment (cranes, excavators, dump trucks)
✓ Building materials (lumber piles, steel, concrete)
✓ Temporary structures (trailers, scaffolding, fencing)
✓ Partially completed structures or building frames
✓ Disturbed earth with organized construction activity

COMMON FALSE POSITIVES to avoid:
✗ Agricultural fields or farming operations
✗ Completed buildings or developed areas
✗ Industrial storage yards or material depots
✗ Parking lots under maintenance
✗ Natural features (rivers, ponds, rock formations)
✗ Sports facilities or recreational construction

CONFIDENCE GUIDELINES:
- 0.8-1.0: Clear construction activity in central tile with context support
- 0.6-0.7: Likely construction in central tile, some ambiguity
- 0.4-0.5: Uncertain construction-related activity in central tile
- 0.2-0.3: Possibly construction-related
- 0.0-0.1: Definitely not a construction site

Respond ONLY with JSON: {"construction_site": true|false, "confidence": number between 0 and 1}"""


def get_coarse_scanning_prompt():
    """Enhanced prompt for coarse-stage scanning of stitched tile blocks."""
    return """Analyze this stitched satellite image containing multiple tiles for potential dumpster locations.

You are performing a COARSE SCAN to identify areas that might contain dumpsters. This will be followed by detailed analysis if positive.

Look for signs of commercial/industrial activity and potential dumpster-like objects:
✓ Commercial buildings with loading areas
✓ Industrial facilities or warehouses  
✓ Retail complexes or shopping centers
✓ Dark rectangular objects that could be dumpsters
✓ Service areas behind buildings
✓ Parking lots adjacent to commercial buildings

COARSE DETECTION CRITERIA:
- Any dark rectangular objects that might be dumpsters
- Commercial/industrial context that suggests dumpster presence
- Loading docks, service areas, or waste collection points
- Be MORE permissive than detailed scanning - capture potential areas

CONFIDENCE for coarse scanning:
- 0.7-1.0: Strong signs of commercial activity with likely dumpster objects
- 0.5-0.6: Some commercial activity, possible dumpster-like objects
- 0.3-0.4: Mixed signals, uncertain commercial activity
- 0.1-0.2: Mainly residential or no obvious dumpster-like objects
- 0.0: Clearly no commercial activity or dumpster potential

Set dumpster=true if confidence ≥ 0.3 (lower threshold for coarse scanning).

Respond ONLY with JSON: {"dumpster": true|false, "confidence": number between 0 and 1}"""


def get_construction_coarse_scanning_prompt():
    """Enhanced prompt for coarse-stage construction site scanning of stitched tile blocks."""
    return """Analyze this stitched satellite image containing multiple tiles for potential construction site locations.

You are performing a COARSE SCAN to identify areas that might contain active construction sites. This will be followed by detailed analysis if positive.

Look for signs of construction and development activity:
✓ Large areas of disturbed or exposed earth
✓ Visible construction equipment or machinery
✓ Building materials or storage areas
✓ Infrastructure development or road construction
✓ New building foundations or structures under construction
✓ Construction site access roads and staging areas

COARSE DETECTION CRITERIA:
- Any visible construction equipment or machinery
- Large areas of site preparation or excavation
- New development in previously undeveloped areas
- Building projects in various stages of completion
- Be MORE permissive than detailed scanning - capture potential construction areas

CONFIDENCE for coarse construction scanning:
- 0.7-1.0: Strong signs of construction activity with visible equipment or major site work
- 0.5-0.6: Some construction indicators, possible development activity
- 0.3-0.4: Mixed signals, uncertain development activity (may be maintenance or agriculture)
- 0.1-0.2: Mainly developed areas or no obvious construction activity
- 0.0: No construction or development activity visible

Set construction_site=true if confidence ≥ 0.4 (lower threshold for coarse scanning).

Respond ONLY with JSON: {"construction_site": true|false, "confidence": number between 0 and 1}"""


def get_prompt_for_scan_type(scan_type: str, context_radius: int = 0, detection_type: str = "dumpsters") -> str:
    """
    Get the appropriate prompt based on scanning context and detection type.
    
    Args:
        scan_type: 'base', 'context', or 'coarse'
        context_radius: radius for context-aware scanning
        detection_type: 'dumpsters' or 'construction'
        
    Returns:
        Appropriate prompt string
    """
    if detection_type == "construction":
        if scan_type == 'coarse':
            return get_construction_coarse_scanning_prompt()
        elif scan_type == 'context' or context_radius > 0:
            return get_construction_context_aware_prompt()
        else:
            return get_construction_base_detection_prompt()
    else:  # dumpsters (default)
        if scan_type == 'coarse':
            return get_coarse_scanning_prompt()
        elif scan_type == 'context' or context_radius > 0:
            return get_context_aware_prompt()
        else:
            return get_base_detection_prompt()


def get_detection_response_key(detection_type: str = "dumpsters") -> str:
    """
    Get the JSON response key for a given detection type.
    
    Args:
        detection_type: 'dumpsters' or 'construction'
        
    Returns:
        The key name used in JSON responses (e.g., 'dumpster', 'construction_site')
    """
    if detection_type == "construction":
        return "construction_site"
    else:  # dumpsters (default)
        return "dumpster"


def add_hard_negatives_to_prompt(base_prompt: str, false_positive_examples: list = None) -> str:
    """
    Enhance a prompt with specific hard negative examples.
    
    Args:
        base_prompt: Base prompt to enhance
        false_positive_examples: List of common false positive patterns
        
    Returns:
        Enhanced prompt with hard negative examples
    """
    if not false_positive_examples:
        # Default hard negatives based on common satellite imagery mistakes
        false_positive_examples = [
            "Dark SUVs or pickup trucks parked behind buildings",
            "Shadows cast by building overhangs or structures",
            "HVAC equipment on building rooftops",
            "Small storage sheds or utility buildings",
            "Rectangular parking spaces or pavement markings",
            "Construction equipment or machinery",
            "Trailers or RVs in parking areas"
        ]
    
    hard_negative_section = "\nBased on previous analysis, these are commonly mistaken for dumpsters but are NOT dumpsters:\n"
    for example in false_positive_examples:
        hard_negative_section += f"✗ {example}\n"
    
    # Insert hard negatives before the confidence scoring section
    if "CONFIDENCE SCORING:" in base_prompt:
        parts = base_prompt.split("CONFIDENCE SCORING:")
        return parts[0] + hard_negative_section + "\nCONFIDENCE SCORING:" + parts[1]
    else:
        return base_prompt + hard_negative_section