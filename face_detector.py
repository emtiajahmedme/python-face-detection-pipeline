"""Advanced face detection utilities with premium features."""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Optional, Any
import json
from dataclasses import dataclass
from datetime import datetime
import hashlib

from imgbeddings import imgbeddings
from config import FaceDetectionConfig
from logger import get_logger, ProgressLogger

# Make imgbeddings optional
try:
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class FaceDetectionResult:
    """Enhanced face detection result with metadata."""
    
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    embedding: Optional[np.ndarray] = None
    face_id: Optional[str] = None
    landmarks: Optional[List[Tuple[int, int]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        result = {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "face_id": self.face_id,
            "landmarks": self.landmarks,
            "metadata": self.metadata or {}
        }
        if self.embedding is not None:
            result["embedding"] = self.embedding.tolist()
        return result


@dataclass
class ImageAnalysisResult:
    """Complete analysis result for an image."""
    
    image_path: str
    faces: List[FaceDetectionResult]
    processing_time: float
    image_dimensions: Tuple[int, int]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "faces": [face.to_dict() for face in self.faces],
            "face_count": len(self.faces),
            "processing_time": self.processing_time,
            "image_dimensions": self.image_dimensions,
            "timestamp": self.timestamp
        }


class PremiumFaceDetector:
    """Enhanced face detector with premium features."""
    
    def __init__(self, config: FaceDetectionConfig):
        self.config = config
        self.logger = get_logger()
        self.haar_cascade = None
        self.ibed = None
        
        # Initialize models
        self._initialize_models()
        
        # Create directories
        self._setup_directories()
    
    def _initialize_models(self):
        """Initialize face detection models."""
        try:
            if not os.path.exists(self.config.haar_cascade_xml):
                raise FileNotFoundError(f"Haar cascade file not found: {self.config.haar_cascade_xml}")
            
            self.haar_cascade = cv2.CascadeClassifier(self.config.haar_cascade_xml)
            if self.haar_cascade.empty():
                raise ValueError("Failed to load Haar cascade classifier")
            
            # Initialize imgbeddings only if needed and available
            try:
                self.ibed = imgbeddings()
                self.logger.info("Face detection models initialized successfully (with embeddings)")
            except Exception as e:
                self.logger.warning(f"Embeddings model failed to initialize: {e}")
                self.logger.info("Face detection models initialized successfully (without embeddings)")
                self.ibed = None
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.config.data_dir,
            self.config.stored_faces_dir,
            self.config.output_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Directories setup completed")
    
    def detect_faces_with_confidence(
        self, 
        image_path: str, 
        return_embeddings: bool = True
    ) -> ImageAnalysisResult:
        """Detect faces with confidence scores and metadata."""
        start_time = datetime.now()
        
        try:
            # Load and validate image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img_gray.shape
            
            # Detect faces
            faces_raw = self.haar_cascade.detectMultiScale(
                img_gray,
                scaleFactor=self.config.scale_factor,
                minNeighbors=self.config.min_neighbors,
                minSize=self.config.min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Process detected faces
            face_results = []
            for i, (x, y, w, h) in enumerate(faces_raw[:self.config.max_faces_per_image]):
                # Extract face region
                face_region = img_gray[y:y+h, x:x+w]
                
                # Generate face ID
                face_hash = hashlib.md5(face_region.tobytes()).hexdigest()[:12]
                face_id = f"face_{face_hash}"
                
                # Calculate confidence (basic estimation based on face size and clarity)
                confidence = self._calculate_confidence(face_region, w, h)
                
                # Skip low confidence faces
                if confidence < self.config.confidence_threshold:
                    continue
                
                # Generate embedding if requested
                embedding = None
                if return_embeddings and self.ibed is not None:
                    try:
                        pil_face = Image.fromarray(face_region)
                        embedding = self.ibed.to_embeddings(pil_face)[0]
                    except Exception as e:
                        self.logger.warning(f"Failed to generate embedding for face {i}: {e}")
                elif return_embeddings and self.ibed is None:
                    self.logger.debug("Embeddings requested but model not available")
                
                # Create metadata
                metadata = {
                    "area": w * h,
                    "aspect_ratio": w / h,
                    "position": {"x": x, "y": y},
                    "relative_position": {
                        "x": x / width,
                        "y": y / height
                    }
                }
                
                face_result = FaceDetectionResult(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    embedding=embedding,
                    face_id=face_id,
                    metadata=metadata
                )
                
                face_results.append(face_result)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ImageAnalysisResult(
                image_path=image_path,
                faces=face_results,
                processing_time=processing_time,
                image_dimensions=(width, height),
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info(
                f"Detected {len(face_results)} faces in {image_path} "
                f"(processing time: {processing_time:.2f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Face detection failed for {image_path}: {e}")
            raise
    
    def _calculate_confidence(self, face_region: np.ndarray, width: int, height: int) -> float:
        """Calculate confidence score for detected face."""
        # Basic confidence calculation based on:
        # 1. Face size (larger faces generally more reliable)
        # 2. Image clarity (using Laplacian variance)
        # 3. Aspect ratio (faces should be roughly rectangular)
        
        # Size score (0-1)
        face_area = width * height
        max_area = self.config.min_size[0] * self.config.min_size[1] * 4  # Reasonable max
        size_score = min(face_area / max_area, 1.0)
        
        # Clarity score (0-1) using Laplacian variance
        laplacian_var = cv2.Laplacian(face_region, cv2.CV_64F).var()
        clarity_score = min(laplacian_var / 500.0, 1.0)  # Normalize to 0-1
        
        # Aspect ratio score (0-1)
        aspect_ratio = width / height
        ideal_ratio = 0.8  # Typical face aspect ratio
        aspect_score = 1.0 - abs(aspect_ratio - ideal_ratio) / ideal_ratio
        aspect_score = max(0.0, aspect_score)
        
        # Weighted combination
        confidence = (size_score * 0.3 + clarity_score * 0.5 + aspect_score * 0.2)
        return min(confidence, 1.0)
    
    def save_detected_faces(
        self, 
        result: ImageAnalysisResult, 
        base_filename: Optional[str] = None
    ) -> List[str]:
        """Save detected faces to disk."""
        if not self.config.save_detected_faces:
            return []
        
        saved_files = []
        base_name = base_filename or os.path.splitext(os.path.basename(result.image_path))[0]
        
        # Load original image
        img = cv2.imread(result.image_path)
        
        for i, face in enumerate(result.faces):
            x, y, w, h = face.bbox
            face_region = img[y:y+h, x:x+w]
            
            # Generate filename
            filename = f"{base_name}_face_{i}_conf_{face.confidence:.2f}_{face.face_id}.jpg"
            filepath = os.path.join(self.config.stored_faces_dir, filename)
            
            # Save face
            cv2.imwrite(filepath, face_region)
            saved_files.append(filepath)
        
        self.logger.info(f"Saved {len(saved_files)} detected faces")
        return saved_files
    
    def create_annotated_image(
        self, 
        result: ImageAnalysisResult, 
        output_path: Optional[str] = None
    ) -> str:
        """Create annotated image with bounding boxes and metadata."""
        if not self.config.save_annotated_images:
            return ""
        
        # Load image
        img = cv2.imread(result.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for i, face in enumerate(result.faces):
            x, y, w, h = face.bbox
            
            # Color based on confidence
            if face.confidence >= 0.9:
                color = "green"
            elif face.confidence >= 0.7:
                color = "orange"
            else:
                color = "red"
            
            # Draw bounding box
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            
            # Draw label
            label = f"Face {i+1}\nConf: {face.confidence:.2f}\nID: {face.face_id}"
            
            # Background for text
            text_bbox = draw.textbbox((x, y - 60), label, font=small_font)
            draw.rectangle(text_bbox, fill=color, outline=color)
            draw.text((x, y - 60), label, fill="white", font=small_font)
        
        # Add summary information
        summary = f"Faces Detected: {len(result.faces)} | Processing Time: {result.processing_time:.2f}s"
        draw.text((10, 10), summary, fill="white", font=font, stroke_width=2, stroke_fill="black")
        
        # Save annotated image
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(result.image_path))[0]
            output_path = os.path.join(self.config.output_dir, f"{base_name}_annotated.jpg")
        
        # Convert back to RGB and save
        pil_img.save(output_path, "JPEG", quality=95)
        
        self.logger.info(f"Saved annotated image: {output_path}")
        return output_path


class FaceMatcher:
    """Premium face matching with advanced similarity search."""
    
    def __init__(self, config: FaceDetectionConfig):
        self.config = config
        self.logger = get_logger()
        self.stored_embeddings = []
        self.stored_metadata = []
    
    def add_face_embedding(self, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add a face embedding to the database."""
        self.stored_embeddings.append(embedding)
        self.stored_metadata.append(metadata)
    
    def find_similar_faces(
        self, 
        target_embedding: np.ndarray, 
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find similar faces with detailed similarity scores."""
        if not self.stored_embeddings:
            return []
        
        threshold = threshold or self.config.similarity_threshold
        
        similarities = []
        for i, stored_embedding in enumerate(self.stored_embeddings):
            # Calculate cosine similarity
            similarity = np.dot(target_embedding, stored_embedding) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= threshold:
                similarities.append({
                    "index": i,
                    "similarity": float(similarity),
                    "metadata": self.stored_metadata[i],
                    "match_quality": self._classify_match_quality(similarity)
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top N matches
        return similarities[:self.config.top_n_matches]
    
    def _classify_match_quality(self, similarity: float) -> str:
        """Classify match quality based on similarity score."""
        if similarity >= 0.95:
            return "excellent"
        elif similarity >= 0.85:
            return "good"
        elif similarity >= 0.75:
            return "fair"
        else:
            return "poor"