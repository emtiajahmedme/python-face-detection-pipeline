"""Utility functions for the Premium Face Detection Pipeline."""

import os
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes

from config import FaceDetectionConfig
from face_detector import ImageAnalysisResult, PremiumFaceDetector
from logger import get_logger


def validate_image_file(file_path: str) -> bool:
    """Validate if file is a supported image format."""
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        return False
    
    # Check MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and not mime_type.startswith('image/'):
        return False
    
    return True


def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║              PREMIUM FACE DETECTION PIPELINE                ║
║                                                              ║
║  Professional-grade face detection and recognition system   ║
║                                                              ║
║  Features:                                                   ║
║  • Advanced face detection with confidence scoring          ║
║  • Face recognition and similarity matching                 ║
║  • Batch processing capabilities                            ║
║  • Professional CLI with comprehensive options              ║
║  • Detailed logging and progress tracking                   ║
║  • Multiple output formats (JSON, CSV, TXT)                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def format_results_table(results: List[ImageAnalysisResult]) -> str:
    """Format detection results as a table."""
    if not results:
        return "No results to display"
    
    table_lines = []
    table_lines.append("┌" + "─" * 100 + "┐")
    table_lines.append("│" + "FACE DETECTION RESULTS".center(100) + "│")
    table_lines.append("├" + "─" * 50 + "┬" + "─" * 10 + "┬" + "─" * 15 + "┬" + "─" * 23 + "┤")
    table_lines.append("│" + "Image".ljust(50) + "│" + "Faces".center(10) + "│" + "Avg Confidence".center(15) + "│" + "Processing Time (s)".center(23) + "│")
    table_lines.append("├" + "─" * 50 + "┼" + "─" * 10 + "┼" + "─" * 15 + "┼" + "─" * 23 + "┤")
    
    for result in results:
        image_name = os.path.basename(result.image_path)
        if len(image_name) > 48:
            image_name = image_name[:45] + "..."
        
        face_count = len(result.faces)
        avg_confidence = sum(face.confidence for face in result.faces) / face_count if face_count > 0 else 0
        processing_time = result.processing_time
        
        table_lines.append(
            "│" + image_name.ljust(50) + 
            "│" + str(face_count).center(10) + 
            "│" + f"{avg_confidence:.3f}".center(15) + 
            "│" + f"{processing_time:.2f}".center(23) + "│"
        )
    
    table_lines.append("└" + "─" * 50 + "┴" + "─" * 10 + "┴" + "─" * 15 + "┴" + "─" * 23 + "┘")
    
    return "\n".join(table_lines)


class OutputManager:
    """Manages output formatting and saving."""
    
    def __init__(self, config: FaceDetectionConfig):
        self.config = config
        self.logger = get_logger()
    
    def save_results(self, results: List[ImageAnalysisResult]) -> str:
        """Save results in the configured format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.output_format == "json":
            return self._save_json(results, timestamp)
        elif self.config.output_format == "csv":
            return self._save_csv(results, timestamp)
        elif self.config.output_format == "txt":
            return self._save_txt(results, timestamp)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
    
    def _save_json(self, results: List[ImageAnalysisResult], timestamp: str) -> str:
        """Save results as JSON."""
        output_path = os.path.join(self.config.output_dir, f"detection_results_{timestamp}.json")
        
        # Convert results to serializable format
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_images": len(results),
                "total_faces": sum(len(result.faces) for result in results),
                "config": self.config.to_dict()
            },
            "results": [result.to_dict() for result in results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path
    
    def _save_csv(self, results: List[ImageAnalysisResult], timestamp: str) -> str:
        """Save results as CSV."""
        output_path = os.path.join(self.config.output_dir, f"detection_results_{timestamp}.csv")
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                "image_path", "face_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
                "confidence", "face_area", "aspect_ratio", "relative_x", "relative_y",
                "processing_time", "timestamp"
            ])
            
            # Write data
            for result in results:
                for face in result.faces:
                    x, y, w, h = face.bbox
                    metadata = face.metadata or {}
                    writer.writerow([
                        result.image_path,
                        face.face_id,
                        x, y, w, h,
                        face.confidence,
                        metadata.get("area", w * h),
                        metadata.get("aspect_ratio", w / h),
                        metadata.get("relative_position", {}).get("x", x / result.image_dimensions[0]),
                        metadata.get("relative_position", {}).get("y", y / result.image_dimensions[1]),
                        result.processing_time,
                        result.timestamp
                    ])
        
        return output_path
    
    def _save_txt(self, results: List[ImageAnalysisResult], timestamp: str) -> str:
        """Save results as formatted text."""
        output_path = os.path.join(self.config.output_dir, f"detection_results_{timestamp}.txt")
        
        with open(output_path, 'w') as f:
            f.write("PREMIUM FACE DETECTION RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Total faces detected: {sum(len(result.faces) for result in results)}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Image {i}: {result.image_path}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Dimensions: {result.image_dimensions[0]}x{result.image_dimensions[1]}\n")
                f.write(f"Processing time: {result.processing_time:.2f}s\n")
                f.write(f"Faces detected: {len(result.faces)}\n")
                
                if result.faces:
                    f.write("\nFace Details:\n")
                    for j, face in enumerate(result.faces, 1):
                        x, y, w, h = face.bbox
                        f.write(f"  Face {j}:\n")
                        f.write(f"    ID: {face.face_id}\n")
                        f.write(f"    Bounding box: ({x}, {y}, {w}, {h})\n")
                        f.write(f"    Confidence: {face.confidence:.3f}\n")
                        f.write(f"    Area: {w * h} pixels\n")
                        f.write(f"    Aspect ratio: {w/h:.2f}\n")
                        if face.metadata:
                            rel_pos = face.metadata.get("relative_position", {})
                            if rel_pos:
                                f.write(f"    Relative position: ({rel_pos.get('x', 0):.2f}, {rel_pos.get('y', 0):.2f})\n")
                        f.write("\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        return output_path


class BatchProcessor:
    """Handles batch processing of multiple images."""
    
    def __init__(self, detector: PremiumFaceDetector, config: FaceDetectionConfig):
        self.detector = detector
        self.config = config
        self.logger = get_logger()
    
    def process_directory(self, directory: str, recursive: bool = False) -> List[ImageAnalysisResult]:
        """Process all images in a directory."""
        image_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    if validate_image_file(file_path):
                        image_files.append(file_path)
        else:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and validate_image_file(file_path):
                    image_files.append(file_path)
        
        self.logger.info(f"Found {len(image_files)} image files in {directory}")
        
        return self.process_files(image_files)
    
    def process_files(self, file_list: List[str]) -> List[ImageAnalysisResult]:
        """Process a list of image files."""
        results = []
        
        for i, file_path in enumerate(file_list):
            try:
                self.logger.info(f"Processing {i+1}/{len(file_list)}: {os.path.basename(file_path)}")
                result = self.detector.detect_faces_with_confidence(file_path)
                results.append(result)
                
                # Save intermediate results if configured
                if self.config.save_detected_faces:
                    self.detector.save_detected_faces(result)
                
                if self.config.save_annotated_images:
                    self.detector.create_annotated_image(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        return results


class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str):
        """End timing an operation and record the duration."""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            if operation not in self.metrics:
                self.metrics[operation] = []
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            return duration
        return None
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics."""
        stats = {}
        for operation, times in self.metrics.items():
            stats[operation] = {
                "count": len(times),
                "total": sum(times),
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times)
            }
        return stats
    
    def print_report(self):
        """Print performance report."""
        stats = self.get_statistics()
        print("\nPERFORMANCE REPORT")
        print("=" * 50)
        for operation, data in stats.items():
            print(f"\n{operation}:")
            print(f"  Count: {data['count']}")
            print(f"  Total time: {data['total']:.2f}s")
            print(f"  Average: {data['average']:.2f}s")
            print(f"  Min: {data['min']:.2f}s")
            print(f"  Max: {data['max']:.2f}s")


def create_demo_images():
    """Create demo images for testing (placeholder function)."""
    # This would create sample images with faces for demonstration
    # In a real implementation, you might download sample images or create synthetic ones
    pass


def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics."""
    import platform
    import psutil
    import cv2
    import numpy as np
    
    info = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation()
        },
        "memory": {
            "total": psutil.virtual_memory().total,
            "available": psutil.virtual_memory().available,
            "percent": psutil.virtual_memory().percent
        },
        "libraries": {
            "opencv": cv2.__version__,
            "numpy": np.__version__
        }
    }
    
    try:
        import onnxruntime as ort
        info["onnxruntime"] = {
            "version": ort.__version__,
            "providers": ort.get_available_providers()
        }
    except ImportError:
        info["onnxruntime"] = "Not installed"
    
    return info