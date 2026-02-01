"""
Upper-Body Clothing Detection with YOLOv8
==========================================
Detects 6 upper-body clothing categories from DeepFashion2:
- Short sleeve top
- Long sleeve top
- Short sleeve outwear
- Long sleeve outwear
- Vest
- Sling

Usage:
    python detect_top_clothes.py
    python detect_top_clothes.py --source video.mp4
    python detect_top_clothes.py --source 0 --conf 0.6 --scale 0.75
"""

import cv2
import gc
import argparse
from ultralytics import YOLO


# =============================================================================
# CLASS CONFIGURATION - DeepFashion2 Upper-Body Categories
# =============================================================================
# These are the 6 upper-body clothing classes we want to detect.
# Modify this dictionary if your model uses different class indices.
# Format: {class_index: "class_name"}

TARGET_CLASSES = {
    0: "short sleeve top",
    1: "long sleeve top",
    2: "short sleeve outwear",
    3: "long sleeve outwear",
    4: "vest",
    5: "sling"
}

# List of class indices to filter (only detect these)
TARGET_CLASS_IDS = list(TARGET_CLASSES.keys())

# Colors for each class (BGR format)
CLASS_COLORS = {
    0: (255, 100, 100),   # Light blue - short sleeve top
    1: (255, 0, 0),       # Blue - long sleeve top
    2: (100, 255, 100),   # Light green - short sleeve outwear
    3: (0, 255, 0),       # Green - long sleeve outwear
    4: (100, 100, 255),   # Light red - vest
    5: (0, 255, 255)      # Yellow - sling
}


# =============================================================================
# MODEL PATH CONFIGURATION
# =============================================================================
# Change this to your custom DeepFashion2 trained weights
MODEL_PATH = "yolov8n.pt"  # <-- CHANGE THIS to your custom weights


def detect_upper_body_clothes(source=0, conf_threshold=0.5, frame_scale=0.5):
    """
    Real-time upper-body clothing detection using YOLOv8.
    
    Args:
        source: 0 for webcam, or path to video file
        conf_threshold: Minimum confidence for detections (0.0 to 1.0)
        frame_scale: Scale factor for processing (lower = faster, less accurate)
    """
    # Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"✅ Model loaded. Classes: {model.names}")
    
    # Open video source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("❌ Cannot open video source")
        return
    
    # Get original frame dimensions
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🎥 Video source: {orig_width}x{orig_height}")
    print(f"🔧 Processing scale: {frame_scale} ({int(orig_width*frame_scale)}x{int(orig_height*frame_scale)})")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ---------------------------------------------------------------------
        # FRAME DOWNSCALING FOR EFFICIENCY
        # ---------------------------------------------------------------------
        if frame_scale != 1.0:
            proc_frame = cv2.resize(frame, None, fx=frame_scale, fy=frame_scale)
        else:
            proc_frame = frame.copy()
        
        # ---------------------------------------------------------------------
        # YOLOV8 INFERENCE
        # ---------------------------------------------------------------------
        results = model.predict(
            proc_frame,
            conf=conf_threshold,
            imgsz=320,           # Small input size for speed
            max_det=20,          # Max detections per frame
            verbose=False        # Suppress console output
        )
        
        # ---------------------------------------------------------------------
        # CLASS FILTERING - Only process TARGET_CLASSES
        # ---------------------------------------------------------------------
        # This is where we filter detections to only our 6 clothing categories
        
        detections = results[0].boxes
        detection_count = 0
        
        for box in detections:
            # Get class ID and confidence
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # =================================================================
            # CLASS FILTER: Skip classes not in TARGET_CLASS_IDS
            # To change which classes are detected, modify TARGET_CLASSES dict
            # at the top of this file.
            # =================================================================
            if class_id not in TARGET_CLASS_IDS:
                continue  # Skip this detection
            
            detection_count += 1
            
            # Get bounding box coordinates (scaled back to original frame)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if frame_scale != 1.0:
                x1 = int(x1 / frame_scale)
                y1 = int(y1 / frame_scale)
                x2 = int(x2 / frame_scale)
                y2 = int(y2 / frame_scale)
            
            # Get class name and color
            class_name = TARGET_CLASSES.get(class_id, f"Class {class_id}")
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            
            # -----------------------------------------------------------------
            # DRAW BOUNDING BOX
            # -----------------------------------------------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # -----------------------------------------------------------------
            # DRAW LABEL WITH CONFIDENCE
            # -----------------------------------------------------------------
            label = f"{class_name}: {confidence:.2f}"
            
            # Label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w + 5, y1), 
                color, 
                -1  # Filled
            )
            
            # Label text
            cv2.putText(
                frame, 
                label, 
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255),  # White text
                2
            )
        
        # ---------------------------------------------------------------------
        # DISPLAY STATUS OVERLAY
        # ---------------------------------------------------------------------
        status = f"Detections: {detection_count} | Frame: {frame_count}"
        cv2.rectangle(frame, (10, 10), (350, 40), (0, 0, 0), -1)
        cv2.putText(frame, status, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Upper-Body Clothing Detection", frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Memory cleanup every 100 frames
        if frame_count % 100 == 0:
            gc.collect()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final cleanup
    del model
    gc.collect()
    print(f"✅ Detection stopped. Total frames: {frame_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upper-Body Clothing Detection with YOLOv8")
    parser.add_argument("--source", type=str, default="0", 
                        help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--conf", type=float, default=0.5, 
                        help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--scale", type=float, default=0.5, 
                        help="Frame scale for processing (0.25-1.0)")
    
    args = parser.parse_args()
    
    # Convert source to int if it's a webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    
    print("=" * 60)
    print("Upper-Body Clothing Detection with YOLOv8")
    print("=" * 60)
    print(f"Target classes: {list(TARGET_CLASSES.values())}")
    print("=" * 60)
    
    detect_upper_body_clothes(
        source=source,
        conf_threshold=args.conf,
        frame_scale=args.scale
    )
