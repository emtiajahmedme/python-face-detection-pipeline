#!/usr/bin/env python3
"""
Premium Face Detection Pipeline CLI
A professional-grade face detection and recognition system.
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional
import glob

from config import FaceDetectionConfig, create_sample_config
from logger import setup_logger, ProgressLogger
from face_detector import PremiumFaceDetector, FaceMatcher, ImageAnalysisResult
from utils import (
    OutputManager, 
    BatchProcessor, 
    validate_image_file,
    print_banner,
    format_results_table
)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Premium Face Detection Pipeline - Professional face detection and recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s detect image.jpg                    # Basic face detection
  %(prog)s detect *.jpg --batch               # Batch process multiple images
  %(prog)s match target.jpg reference.jpg     # Compare faces between images
  %(prog)s config --create                    # Create sample configuration
  %(prog)s detect image.jpg --confidence 0.8  # Custom confidence threshold
        """
    )
    
    # Global options
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.json",
        help="Configuration file path (default: config.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log to file instead of console"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Detect command
    detect_parser = subparsers.add_parser(
        "detect",
        help="Detect faces in images"
    )
    detect_parser.add_argument(
        "images",
        nargs="+",
        help="Input image file(s) or pattern"
    )
    detect_parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Enable batch processing mode"
    )
    detect_parser.add_argument(
        "--confidence",
        type=float,
        help="Minimum confidence threshold (0.0-1.0)"
    )
    detect_parser.add_argument(
        "--max-faces",
        type=int,
        help="Maximum number of faces to detect per image"
    )
    detect_parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results"
    )
    detect_parser.add_argument(
        "--format",
        choices=["json", "csv", "txt"],
        help="Output format"
    )
    detect_parser.add_argument(
        "--no-save-faces",
        action="store_true",
        help="Don't save detected face images"
    )
    detect_parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Don't create annotated images"
    )
    
    # Match command
    match_parser = subparsers.add_parser(
        "match",
        help="Match faces between images"
    )
    match_parser.add_argument(
        "target",
        help="Target image containing face to match"
    )
    match_parser.add_argument(
        "reference",
        nargs="+",
        help="Reference image(s) to search in"
    )
    match_parser.add_argument(
        "--threshold",
        type=float,
        help="Similarity threshold (0.0-1.0)"
    )
    match_parser.add_argument(
        "--top-n",
        type=int,
        help="Number of top matches to return"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management"
    )
    config_parser.add_argument(
        "--create",
        action="store_true",
        help="Create sample configuration file"
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration"
    )
    config_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration file"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show system information"
    )
    
    return parser


def handle_detect_command(args, config: FaceDetectionConfig, logger):
    """Handle face detection command."""
    # Override config with command line arguments
    if args.confidence is not None:
        config.confidence_threshold = args.confidence
    if args.max_faces is not None:
        config.max_faces_per_image = args.max_faces
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.format is not None:
        config.output_format = args.format
    if args.no_save_faces:
        config.save_detected_faces = False
    if args.no_annotate:
        config.save_annotated_images = False
    
    # Expand image patterns and validate files
    image_files = []
    for pattern in args.images:
        if "*" in pattern or "?" in pattern:
            # Handle glob patterns
            matches = glob.glob(pattern)
            if not matches:
                logger.warning(f"No files matched pattern: {pattern}")
            image_files.extend(matches)
        else:
            if os.path.exists(pattern):
                image_files.append(pattern)
            else:
                logger.error(f"File not found: {pattern}")
                continue
    
    if not image_files:
        logger.error("No valid image files found")
        return False
    
    # Validate image files
    valid_files = []
    for img_file in image_files:
        if validate_image_file(img_file):
            valid_files.append(img_file)
        else:
            logger.warning(f"Invalid image file: {img_file}")
    
    if not valid_files:
        logger.error("No valid image files to process")
        return False
    
    logger.info(f"Processing {len(valid_files)} image(s)")
    
    # Initialize detector
    detector = PremiumFaceDetector(config)
    output_manager = OutputManager(config)
    
    # Process images
    all_results = []
    
    if args.batch and len(valid_files) > 1:
        # Batch processing
        processor = BatchProcessor(detector, config)
        with ProgressLogger(logger, "Batch face detection", len(valid_files)) as progress:
            for i, img_file in enumerate(valid_files):
                try:
                    result = detector.detect_faces_with_confidence(img_file)
                    all_results.append(result)
                    
                    # Save individual results
                    if config.save_detected_faces:
                        detector.save_detected_faces(result)
                    
                    if config.save_annotated_images:
                        detector.create_annotated_image(result)
                    
                    progress.update(1, f"Processed {os.path.basename(img_file)}")
                    
                except Exception as e:
                    progress.log_error(f"Failed to process {img_file}: {e}")
                    continue
    else:
        # Single image or individual processing
        for img_file in valid_files:
            try:
                logger.info(f"Processing: {img_file}")
                result = detector.detect_faces_with_confidence(img_file)
                all_results.append(result)
                
                # Save results
                if config.save_detected_faces:
                    saved_faces = detector.save_detected_faces(result)
                    logger.info(f"Saved {len(saved_faces)} face images")
                
                if config.save_annotated_images:
                    annotated_path = detector.create_annotated_image(result)
                    if annotated_path:
                        logger.info(f"Created annotated image: {annotated_path}")
                
                # Show immediate results for single images
                if len(valid_files) == 1:
                    print(format_results_table([result]))
                
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")
                continue
    
    # Save comprehensive results
    if all_results:
        output_path = output_manager.save_results(all_results)
        logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        total_faces = sum(len(result.faces) for result in all_results)
        avg_processing_time = sum(result.processing_time for result in all_results) / len(all_results)
        
        print(f"\n{'='*60}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Images processed: {len(all_results)}")
        print(f"Total faces detected: {total_faces}")
        print(f"Average processing time: {avg_processing_time:.2f}s per image")
        print(f"Results saved to: {output_path}")
        
        if len(all_results) > 1:
            print(format_results_table(all_results))
    
    return True


def handle_match_command(args, config: FaceDetectionConfig, logger):
    """Handle face matching command."""
    # Override config with command line arguments
    if args.threshold is not None:
        config.similarity_threshold = args.threshold
    if args.top_n is not None:
        config.top_n_matches = args.top_n
    
    # Validate files
    if not validate_image_file(args.target):
        logger.error(f"Invalid target image: {args.target}")
        return False
    
    reference_files = []
    for ref_file in args.reference:
        if validate_image_file(ref_file):
            reference_files.append(ref_file)
        else:
            logger.warning(f"Invalid reference image: {ref_file}")
    
    if not reference_files:
        logger.error("No valid reference images found")
        return False
    
    # Initialize detector and matcher
    detector = PremiumFaceDetector(config)
    matcher = FaceMatcher(config)
    
    logger.info(f"Matching faces from {args.target} against {len(reference_files)} reference images")
    
    # Extract target face
    with ProgressLogger(logger, "Target face extraction") as progress:
        target_result = detector.detect_faces_with_confidence(args.target)
        progress.update(1, f"Found {len(target_result.faces)} faces in target")
    
    if not target_result.faces:
        logger.error(f"No faces found in target image: {args.target}")
        return False
    
    if len(target_result.faces) > 1:
        logger.warning(f"Multiple faces found in target image. Using the first face (confidence: {target_result.faces[0].confidence:.2f})")
    
    target_embedding = target_result.faces[0].embedding
    if target_embedding is None:
        logger.error("Failed to generate embedding for target face")
        return False
    
    # Process reference images and build face database
    with ProgressLogger(logger, "Reference image processing", len(reference_files)) as progress:
        for ref_file in reference_files:
            try:
                ref_result = detector.detect_faces_with_confidence(ref_file)
                
                for face in ref_result.faces:
                    if face.embedding is not None:
                        metadata = {
                            "source_image": ref_file,
                            "face_id": face.face_id,
                            "confidence": face.confidence,
                            "bbox": face.bbox,
                            "metadata": face.metadata
                        }
                        matcher.add_face_embedding(face.embedding, metadata)
                
                progress.update(1, f"Processed {os.path.basename(ref_file)} - {len(ref_result.faces)} faces")
                
            except Exception as e:
                progress.log_error(f"Failed to process {ref_file}: {e}")
                continue
    
    # Find matches
    logger.info("Searching for similar faces...")
    matches = matcher.find_similar_faces(target_embedding)
    
    if matches:
        print(f"\n{'='*80}")
        print(f"FACE MATCHING RESULTS")
        print(f"{'='*80}")
        print(f"Target: {args.target}")
        print(f"Found {len(matches)} similar face(s) (threshold: {config.similarity_threshold})")
        print(f"{'='*80}")
        
        for i, match in enumerate(matches, 1):
            metadata = match["metadata"]
            print(f"\nMatch #{i}:")
            print(f"  Source: {metadata['source_image']}")
            print(f"  Face ID: {metadata['face_id']}")
            print(f"  Similarity: {match['similarity']:.3f}")
            print(f"  Quality: {match['match_quality']}")
            print(f"  Detection confidence: {metadata['confidence']:.3f}")
            print(f"  Bounding box: {metadata['bbox']}")
    else:
        print(f"\nNo similar faces found (threshold: {config.similarity_threshold})")
        print("Try lowering the similarity threshold with --threshold")
    
    return True


def handle_config_command(args, config: FaceDetectionConfig, logger):
    """Handle configuration command."""
    if args.create:
        create_sample_config(args.config if hasattr(args, 'config') else "config.json")
        return True
    
    if args.show:
        print(f"\n{'='*60}")
        print("CURRENT CONFIGURATION")
        print(f"{'='*60}")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            print(f"{key:25}: {value}")
        return True
    
    if args.validate:
        try:
            # Try to load and validate config
            test_config = FaceDetectionConfig.load_from_file(args.config)
            print(f"✓ Configuration file '{args.config}' is valid")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    # If no specific action, show help
    print("Use --create, --show, or --validate with the config command")
    return True


def handle_info_command(args, config: FaceDetectionConfig, logger):
    """Handle info command."""
    import cv2
    import numpy as np
    from imgbeddings import imgbeddings
    
    print(f"\n{'='*60}")
    print("SYSTEM INFORMATION")
    print(f"{'='*60}")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Platform: {sys.platform}")
    
    # Check GPU availability
    gpu_info = "Not available"
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            gpu_info = "CUDA GPU available"
        elif 'DmlExecutionProvider' in providers:
            gpu_info = "DirectML GPU available"
    except:
        pass
    
    print(f"GPU acceleration: {gpu_info}")
    
    # Model information
    print(f"\nMODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Haar cascade: {config.haar_cascade_xml}")
    print(f"Haar cascade exists: {os.path.exists(config.haar_cascade_xml)}")
    
    # Test basic functionality
    print(f"\nFUNCTIONALITY TEST")
    print(f"{'='*60}")
    try:
        # Test imgbeddings
        ibed = imgbeddings()
        print("✓ Image embeddings model loaded")
        
        # Test OpenCV cascade
        cascade = cv2.CascadeClassifier(config.haar_cascade_xml)
        if not cascade.empty():
            print("✓ Haar cascade classifier loaded")
        else:
            print("✗ Haar cascade classifier failed to load")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
    
    return True


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Print banner for main commands
    if args.command in ["detect", "match"]:
        print_banner()
    
    # Handle special case: no command provided
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO"
    logger = setup_logger(
        level=log_level,
        log_file=args.log_file,
        console_output=not args.quiet
    )
    
    # Load configuration
    try:
        if os.path.exists(args.config):
            config = FaceDetectionConfig.load_from_file(args.config)
            if args.verbose:
                logger.debug(f"Loaded configuration from: {args.config}")
        else:
            config = FaceDetectionConfig()
            if args.verbose:
                logger.debug("Using default configuration")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Handle commands
    success = False
    try:
        if args.command == "detect":
            success = handle_detect_command(args, config, logger)
        elif args.command == "match":
            success = handle_match_command(args, config, logger)
        elif args.command == "config":
            success = handle_config_command(args, config, logger)
        elif args.command == "info":
            success = handle_info_command(args, config, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())