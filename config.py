"""Configuration management for the Face Detection Pipeline."""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class FaceDetectionConfig:
    """Configuration class for face detection pipeline."""
    
    # Detection parameters
    scale_factor: float = 1.05
    min_neighbors: int = 3  # Reduced for demo compatibility
    min_size: tuple = (30, 30)  # Reduced for demo compatibility
    confidence_threshold: float = 0.3  # Reduced for demo compatibility
    
    # Directories
    data_dir: str = "data"
    stored_faces_dir: str = "data/stored_faces"
    output_dir: str = "output"
    
    # Models
    haar_cascade_xml: str = "haarcascade_frontalface_default.xml"
    
    # Processing options
    batch_size: int = 10
    max_faces_per_image: int = 10
    enable_gpu: bool = False
    
    # Similarity matching
    similarity_threshold: float = 0.8
    top_n_matches: int = 5
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Output options
    save_detected_faces: bool = True
    save_annotated_images: bool = True
    output_format: str = "json"  # json, csv, txt
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'FaceDetectionConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_dir = os.path.dirname(config_path)
        if config_dir:  # Only create directory if there's a directory part
            os.makedirs(config_dir, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


def get_default_config() -> FaceDetectionConfig:
    """Get default configuration."""
    return FaceDetectionConfig()


def create_sample_config(output_path: str = "config.json") -> None:
    """Create a sample configuration file."""
    config = get_default_config()
    config.save_to_file(output_path)
    print(f"Sample configuration created at: {output_path}")