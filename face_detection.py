"""
Face Detection Pipeline (Enhanced Legacy Version)

This is the enhanced version of the original face detection script with some premium features
while maintaining backward compatibility. For full premium features, use premium_face_detection.py

Premium features added:
- Better error handling and logging
- Configuration options
- Confidence scoring
- Progress indicators
- Multiple output formats
"""

import os
import subprocess
import sys
import time
from datetime import datetime
import json

import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Dict, Any

# Import imgbeddings with fallback
try:
    from imgbeddings import imgbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: imgbeddings not available. Face matching will be disabled.")

# Enhanced configuration
class Config:
    def __init__(self):
        self.data_dir = "data"
        self.stored_faces_dir = os.path.join(self.data_dir, "stored_faces") 
        self.test_image = "test-image.png"
        self.target_image = "solo-image.png"
        self.haar_cascade_xml = "haarcascade_frontalface_default.xml"
        
        # Enhanced parameters (relaxed for demo compatibility)
        self.scale_factor = 1.05
        self.min_neighbors = 3  # Reduced from 50 for demo
        self.min_size = (30, 30)  # Reduced from (100, 100) for demo
        self.confidence_threshold = 0.3  # Reduced from 0.7 for demo
        self.max_faces = 10
        
        # Output options
        self.save_results = True
        self.create_annotated = True
        self.verbose = True

config = Config()

# Directory structure
os.makedirs(config.stored_faces_dir, exist_ok=True)

def log_message(message: str, level: str = "INFO"):
    """Enhanced logging with timestamps."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if config.verbose:
        print(f"[{timestamp}] {level}: {message}")

def check_dependencies():
    """Check if all required dependencies are available."""
    log_message("Checking dependencies...")
    
    # Check OpenCV
    try:
        cv2.__version__
        log_message(f"✓ OpenCV {cv2.__version__} available")
    except:
        log_message("✗ OpenCV not available", "ERROR")
        return False
    
    # Check Haar cascade
    if not os.path.exists(config.haar_cascade_xml):
        log_message(f"✗ Haar cascade file not found: {config.haar_cascade_xml}", "ERROR")
        return False
    else:
        log_message(f"✓ Haar cascade model found")
    
    # Check imgbeddings
    if EMBEDDINGS_AVAILABLE:
        log_message("✓ Face embeddings available")
    else:
        log_message("⚠ Face embeddings not available - matching disabled", "WARNING")
    
    return True

# Load face detection model with error handling
try:
    haar_cascade = cv2.CascadeClassifier(config.haar_cascade_xml)
    if haar_cascade.empty():
        raise ValueError("Failed to load Haar cascade classifier")
    log_message("Face detection model loaded successfully")
except Exception as e:
    log_message(f"Failed to load face detection model: {e}", "ERROR")
    sys.exit(1)

# Initialize embeddings model
ibed = None
if EMBEDDINGS_AVAILABLE:
    try:
        ibed = imgbeddings()
        log_message("Face embeddings model initialized")
    except Exception as e:
        log_message(f"Warning: Face embeddings initialization failed: {e}", "WARNING")
        EMBEDDINGS_AVAILABLE = False

def calculate_confidence(face_region: np.ndarray, width: int, height: int) -> float:
    """Calculate confidence score for detected face (basic version)."""
    # Size score
    face_area = width * height
    max_area = config.min_size[0] * config.min_size[1] * 4
    size_score = min(face_area / max_area, 1.0)
    
    # Clarity score using Laplacian variance
    laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
    clarity_score = min(laplacian_var / 500.0, 1.0)
    
    # Aspect ratio score
    aspect_ratio = width / height
    ideal_ratio = 0.8
    aspect_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
    aspect_score = max(0.0, aspect_score)
    
    # Weighted combination
    confidence = (size_score * 0.3 + clarity_score * 0.5 + aspect_score * 0.2)
    return min(confidence, 1.0)

def detect_faces_enhanced(image_path: str) -> List[Dict[str, Any]]:
    """Enhanced face detection with confidence scoring and metadata."""
    log_message(f"Detecting faces in: {image_path}")
    
    if not os.path.exists(image_path):
        log_message(f"Image file not found: {image_path}", "ERROR")
        return []
    
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            log_message(f"Could not load image: {image_path}", "ERROR")
            return []
        
        img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width = img.shape
        
        # Detect faces
        faces_raw = haar_cascade.detectMultiScale(
            img,
            scaleFactor=config.scale_factor,
            minNeighbors=config.min_neighbors,
            minSize=config.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        log_message(f"Found {len(faces_raw)} potential faces")
        
        # Process detected faces
        face_results = []
        for i, (x, y, w, h) in enumerate(faces_raw[:config.max_faces]):
            face_region = img[y:y+h, x:x+w]
            
            # Calculate confidence
            confidence = calculate_confidence(face_region, w, h)
            
            # Filter by confidence
            if confidence < config.confidence_threshold:
                log_message(f"Face {i+1} rejected (confidence: {confidence:.3f})")
                continue
            
            face_data = {
                "id": f"face_{i+1}",
                "bbox": [int(x), int(y), int(w), int(h)],  # Convert to int
                "confidence": float(confidence),  # Convert to float
                "area": int(w * h),  # Convert to int
                "aspect_ratio": float(w / h),  # Convert to float
                "relative_position": {
                    "x": float(x / width),  # Convert to float
                    "y": float(y / height)  # Convert to float
                }
            }
            
            face_results.append(face_data)
            log_message(f"Face {i+1}: confidence={confidence:.3f}, size={w}x{h}")
        
        log_message(f"Accepted {len(face_results)} faces above confidence threshold")
        return face_results
        
    except Exception as e:
        log_message(f"Face detection failed: {e}", "ERROR")
        return []

def detect_faces(image_path: str) -> List[np.ndarray]:
    """Original function for backward compatibility."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=50, minSize=(100, 100))
        cropped_faces = [img[y:y+h, x:x+w] for x, y, w, h in faces]
        return cropped_faces
    except Exception as e:
        log_message(f"Error in detect_faces: {e}", "ERROR")
        return []

def store_faces_enhanced(face_results: List[Dict[str, Any]], image_path: str):
    """Enhanced face storage with metadata."""
    if not face_results:
        log_message("No faces to store")
        return []
    
    log_message(f"Storing {len(face_results)} detected faces...")
    
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        log_message(f"Could not load image for face extraction: {image_path}", "ERROR")
        return []
    
    saved_files = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for face_data in face_results:
        x, y, w, h = face_data["bbox"]
        face_region = img[y:y+h, x:x+w]
        
        # Generate filename with metadata
        filename = f"{base_name}_{face_data['id']}_conf_{face_data['confidence']:.2f}.jpg"
        filepath = os.path.join(config.stored_faces_dir, filename)
        
        cv2.imwrite(filepath, face_region)
        saved_files.append(filepath)
        log_message(f"Saved: {filename}")
    
    return saved_files

def store_faces(faces: List[np.ndarray]):
    """Original function for backward compatibility."""
    for i, face in enumerate(faces):
        face_path = os.path.join(config.stored_faces_dir, f"face_{i}.jpg")
        cv2.imwrite(face_path, face)

def create_annotated_image(face_results: List[Dict[str, Any]], image_path: str) -> Optional[str]:
    """Create annotated image with bounding boxes and confidence scores."""
    if not config.create_annotated or not face_results:
        return None
    
    log_message("Creating annotated image...")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Draw bounding boxes and labels
        for face_data in face_results:
            x, y, w, h = face_data["bbox"]
            confidence = face_data["confidence"]
            
            # Color based on confidence
            if confidence >= 0.9:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.7:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Add label
            label = f"{face_data['id']}: {confidence:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save annotated image
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"{base_name}_annotated.jpg"
        cv2.imwrite(output_path, img)
        log_message(f"Annotated image saved: {output_path}")
        
        return output_path
        
    except Exception as e:
        log_message(f"Failed to create annotated image: {e}", "ERROR")
        return None

def save_results_json(face_results: List[Dict[str, Any]], image_path: str, processing_time: float):
    """Save comprehensive results in JSON format."""
    if not config.save_results:
        return
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "processing_time": processing_time,
            "faces_detected": len(face_results),
            "confidence_threshold": config.confidence_threshold
        },
        "faces": face_results
    }
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{base_name}_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"Results saved: {output_path}")

def calculate_embeddings(image_list: List[np.ndarray]) -> List[np.ndarray]:
    """Calculate embeddings for a list of images (if available)."""
    if not EMBEDDINGS_AVAILABLE or not ibed:
        log_message("Embeddings not available", "WARNING")
        return []
    
    try:
        return [ibed.to_embeddings(Image.fromarray(img))[0] for img in image_list]
    except Exception as e:
        log_message(f"Failed to calculate embeddings: {e}", "ERROR")
        return []

def find_similar_faces(target_embedding: np.ndarray, stored_embeddings: List[np.ndarray], top_n: int = 3):
    """Find top-N most similar face embeddings using cosine similarity."""
    if not stored_embeddings:
        return [], []
    
    similarities = [np.dot(target_embedding, emb) / (np.linalg.norm(target_embedding) * np.linalg.norm(emb)) 
                   for emb in stored_embeddings]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [stored_embeddings[i] for i in top_indices], top_indices

def open_image(path):
    """Open image using system default viewer."""
    image_viewer_from_command_line = {
        'linux': 'xdg-open',
        'win32': 'explorer',
        'darwin': 'open'
    }[sys.platform]
    try:
        subprocess.run([image_viewer_from_command_line, path], check=False)
    except Exception as e:
        log_message(f"Could not open image: {e}", "WARNING")

def main():
    """Enhanced main function with better workflow."""
    print("=" * 60)
    print("FACE DETECTION PIPELINE (Enhanced Legacy Version)")
    print("=" * 60)
    print("For full premium features, use: python premium_face_detection.py")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        log_message("Dependency check failed. Exiting.", "ERROR")
        sys.exit(1)
    
    # Check for test image
    if not os.path.exists(config.test_image):
        log_message(f"Test image not found: {config.test_image}")
        log_message("Please provide a test image or run the demo:")
        log_message("python demo.py")
        
        # Try demo image if available
        if os.path.exists("demo_faces.jpg"):
            config.test_image = "demo_faces.jpg"
            log_message(f"Using demo image: {config.test_image}")
        else:
            return
    
    start_time = time.time()
    
    # Step 1: Enhanced face detection
    log_message("Step 1: Enhanced face detection")
    face_results = detect_faces_enhanced(config.test_image)
    
    if not face_results:
        log_message("No faces detected. Try lowering the confidence threshold.", "WARNING")
        return
    
    # Step 2: Store detected faces
    log_message("Step 2: Storing detected faces")
    saved_files = store_faces_enhanced(face_results, config.test_image)
    
    # Step 3: Create annotated image
    log_message("Step 3: Creating annotated image")
    annotated_path = create_annotated_image(face_results, config.test_image)
    
    processing_time = time.time() - start_time
    
    # Step 4: Save results
    log_message("Step 4: Saving results")
    save_results_json(face_results, config.test_image, processing_time)
    
    # Step 5: Face matching (if embeddings available and target image exists)
    if EMBEDDINGS_AVAILABLE and os.path.exists(config.target_image):
        log_message("Step 5: Face matching")
        
        # Get stored face images for embeddings
        stored_face_images = [cv2.imread(filepath, cv2.IMREAD_GRAYSCALE) for filepath in saved_files]
        stored_embeddings = calculate_embeddings(stored_face_images)
        
        if stored_embeddings:
            # Detect face in target image
            target_results = detect_faces_enhanced(config.target_image)
            if target_results:
                # Use original detect_faces for embeddings calculation
                target_faces = detect_faces(config.target_image)
                if target_faces:
                    target_embedding = calculate_embeddings([target_faces[0]])[0]
                    similar_faces, indices = find_similar_faces(target_embedding, stored_embeddings, top_n=1)
                    
                    if indices:
                        log_message(f"Found similar face match (index: {indices[0]})")
                        best_match_path = saved_files[indices[0]]
                        log_message(f"Best match: {best_match_path}")
                        open_image(best_match_path)
                    else:
                        log_message("No similar faces found")
            else:
                log_message(f"No face detected in target image: {config.target_image}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"✓ Faces detected: {len(face_results)}")
    print(f"✓ Processing time: {processing_time:.2f} seconds")
    print(f"✓ Faces saved: {len(saved_files)}")
    if annotated_path:
        print(f"✓ Annotated image: {annotated_path}")
    
    # Show face details
    print(f"\nFace Details:")
    for face_data in face_results:
        print(f"  {face_data['id']}: confidence={face_data['confidence']:.3f}, "
              f"size={face_data['bbox'][2]}x{face_data['bbox'][3]}")
    
    print(f"\n🚀 Upgrade to Premium Features:")
    print(f"   python premium_face_detection.py detect {config.test_image}")

if __name__ == "__main__":
    main()
