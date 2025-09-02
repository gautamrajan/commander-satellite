"""
Configuration system for different object detection types.
Defines detection parameters, output configurations, and UI settings for each detection type.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class DetectionTypeConfig:
    """Configuration for a specific detection type."""
    id: str
    name: str
    description: str
    output_file_prefix: str  # e.g., 'dumpsters', 'construction_sites'
    confidence_threshold: float
    coarse_threshold: float
    ui_color: str  # CSS color for UI elements
    icon: str  # Font Awesome icon class


# Detection type configurations
DETECTION_TYPES = {
    "dumpsters": DetectionTypeConfig(
        id="dumpsters",
        name="Dumpster Detection", 
        description="Commercial waste containers and dumpsters",
        output_file_prefix="dumpsters",
        confidence_threshold=0.5,
        coarse_threshold=0.3,
        ui_color="#2196F3",
        icon="fa-dumpster"
    ),
    "construction": DetectionTypeConfig(
        id="construction",
        name="Construction Site Detection",
        description="Construction sites, equipment, and building projects",
        output_file_prefix="construction_sites", 
        confidence_threshold=0.6,
        coarse_threshold=0.4,
        ui_color="#FF9800",
        icon="fa-hard-hat"
    )
}


def get_detection_type(detection_type: str) -> DetectionTypeConfig:
    """Get configuration for a detection type."""
    if detection_type not in DETECTION_TYPES:
        raise ValueError(f"Unknown detection type: {detection_type}. Available: {list(DETECTION_TYPES.keys())}")
    return DETECTION_TYPES[detection_type]


def get_available_detection_types() -> List[str]:
    """Get list of available detection type IDs."""
    return list(DETECTION_TYPES.keys())


def get_detection_type_configs() -> Dict[str, DetectionTypeConfig]:
    """Get all detection type configurations."""
    return DETECTION_TYPES.copy()


def get_output_filename(detection_type: str, base_filename: str = None) -> str:
    """
    Generate output filename for a detection type.
    
    Args:
        detection_type: Detection type ID
        base_filename: Optional base filename (defaults to detection type prefix + .jsonl)
        
    Returns:
        Appropriate filename for the detection type
    """
    config = get_detection_type(detection_type)
    if base_filename:
        # Use provided base filename
        return base_filename
    return f"{config.output_file_prefix}.jsonl"


def validate_detection_type(detection_type: Optional[str]) -> str:
    """
    Validate and return detection type, defaulting to 'dumpsters' for backward compatibility.
    
    Args:
        detection_type: Detection type to validate
        
    Returns:
        Valid detection type ID
    """
    if not detection_type:
        return "dumpsters"  # Default for backward compatibility
    
    if detection_type not in DETECTION_TYPES:
        raise ValueError(f"Invalid detection type: {detection_type}. Available: {list(DETECTION_TYPES.keys())}")
    
    return detection_type


def get_detection_type_label(detection_type: str) -> str:
    """Get display label for detection type."""
    return get_detection_type(detection_type).name


def get_ui_config(detection_type: str) -> Dict[str, Any]:
    """
    Get UI-specific configuration for a detection type.
    
    Returns:
        Dictionary with UI configuration (color, icon, etc.)
    """
    config = get_detection_type(detection_type)
    return {
        "color": config.ui_color,
        "icon": config.icon,
        "name": config.name,
        "description": config.description
    }