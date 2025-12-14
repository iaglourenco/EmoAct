"""Activity classification based on pose landmarks and image classification."""

import math
from typing import Optional
from emoact.types import Landmark
from ultralytics.models import YOLO
import numpy as np

# Load YOLOv11 classification model for image-based activity classification
# Model path follows the same pattern as pose.py and objects.py
try:
    classification_model = YOLO("models/yolo11n-cls.pt")
except Exception as e:
    print(f"Warning: Could not load YOLOv11 classification model: {e}")
    print("Image-based classification will be unavailable. Only pose-based classification will work.")
    classification_model = None


def calculate_angle(p1: Landmark, p2: Landmark, p3: Landmark) -> Optional[float]:
    """
    Calculate the angle between three points (p1-p2-p3).
    Returns angle in degrees, or None if landmarks have low confidence.
    """
    min_confidence = 0.3
    if (
        p1["confidence"] < min_confidence
        or p2["confidence"] < min_confidence
        or p3["confidence"] < min_confidence
    ):
        return None

    # Vector from p2 to p1
    v1_x = p1["x"] - p2["x"]
    v1_y = p1["y"] - p2["y"]

    # Vector from p2 to p3
    v2_x = p3["x"] - p2["x"]
    v2_y = p3["y"] - p2["y"]

    # Calculate angle using dot product
    dot_product = v1_x * v2_x + v1_y * v2_y
    mag1 = math.sqrt(v1_x**2 + v1_y**2)
    mag2 = math.sqrt(v2_x**2 + v2_y**2)

    if mag1 == 0 or mag2 == 0:
        return None

    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]

    angle = math.acos(cos_angle)
    return math.degrees(angle)


def get_landmark_by_name(landmarks: list[Landmark], name: str) -> Optional[Landmark]:
    """Get a landmark by name from the list."""
    for landmark in landmarks:
        if landmark["name"] == name:
            return landmark
    return None


def is_arms_raised(landmarks: list[Landmark]) -> bool:
    """Check if one or both arms are raised above the shoulders."""
    left_wrist = get_landmark_by_name(landmarks, "left wrist")
    right_wrist = get_landmark_by_name(landmarks, "right wrist")
    left_shoulder = get_landmark_by_name(landmarks, "left shoulder")
    right_shoulder = get_landmark_by_name(landmarks, "right shoulder")

    if not all([left_shoulder, right_shoulder]):
        return False

    arms_raised = False

    # Check left arm
    if left_wrist and left_shoulder and left_wrist["confidence"] > 0.3:
        if left_wrist["y"] < left_shoulder["y"]:
            arms_raised = True

    # Check right arm
    if right_wrist and right_shoulder and right_wrist["confidence"] > 0.3:
        if right_wrist["y"] < right_shoulder["y"]:
            arms_raised = True

    return arms_raised


def is_sitting(landmarks: list[Landmark]) -> bool:
    """Check if the person is sitting based on hip-knee-ankle angles."""
    left_hip = get_landmark_by_name(landmarks, "left hip")
    left_knee = get_landmark_by_name(landmarks, "left knee")
    left_ankle = get_landmark_by_name(landmarks, "left ankle")

    right_hip = get_landmark_by_name(landmarks, "right hip")
    right_knee = get_landmark_by_name(landmarks, "right knee")
    right_ankle = get_landmark_by_name(landmarks, "right ankle")

    # Check if we have enough landmarks
    if not all([left_hip, left_knee, right_hip, right_knee]):
        return False

    # Calculate knee angles (hip-knee-ankle)
    angles = []
    if left_ankle and left_hip and left_knee:
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if left_angle:
            angles.append(left_angle)

    if right_ankle and right_hip and right_knee:
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        if right_angle:
            angles.append(right_angle)

    # Sitting typically has knee angles between 60 and 120 degrees
    if angles:
        avg_angle = sum(angles) / len(angles)
        return 50 <= avg_angle <= 130

    return False


def is_standing(landmarks: list[Landmark]) -> bool:
    """Check if the person is standing (legs relatively straight)."""
    left_hip = get_landmark_by_name(landmarks, "left hip")
    left_knee = get_landmark_by_name(landmarks, "left knee")
    left_ankle = get_landmark_by_name(landmarks, "left ankle")

    right_hip = get_landmark_by_name(landmarks, "right hip")
    right_knee = get_landmark_by_name(landmarks, "right knee")
    right_ankle = get_landmark_by_name(landmarks, "right ankle")

    # Check if we have enough landmarks
    if not all([left_hip, left_knee, right_hip, right_knee]):
        return False

    # Calculate knee angles
    angles = []
    if left_ankle and left_hip and left_knee:
        left_angle = calculate_angle(left_hip, left_knee, left_ankle)
        if left_angle:
            angles.append(left_angle)

    if right_ankle and right_hip and right_knee:
        right_angle = calculate_angle(right_hip, right_knee, right_ankle)
        if right_angle:
            angles.append(right_angle)

    # Standing typically has knee angles greater than 140 degrees (nearly straight)
    if angles:
        avg_angle = sum(angles) / len(angles)
        return avg_angle > 140

    return False


def is_waving(landmarks: list[Landmark]) -> bool:
    """Check if person is waving (hand raised and elbow bent)."""
    left_shoulder = get_landmark_by_name(landmarks, "left shoulder")
    left_elbow = get_landmark_by_name(landmarks, "left elbow")
    left_wrist = get_landmark_by_name(landmarks, "left wrist")

    right_shoulder = get_landmark_by_name(landmarks, "right shoulder")
    right_elbow = get_landmark_by_name(landmarks, "right elbow")
    right_wrist = get_landmark_by_name(landmarks, "right wrist")

    # Check left arm
    if left_shoulder and left_elbow and left_wrist:
        if left_wrist["y"] < left_shoulder["y"]:  # Hand above shoulder
            elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            if elbow_angle and 60 <= elbow_angle <= 150:
                return True

    # Check right arm
    if right_shoulder and right_elbow and right_wrist:
        if right_wrist["y"] < right_shoulder["y"]:  # Hand above shoulder
            elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            if elbow_angle and 60 <= elbow_angle <= 150:
                return True

    return False


def classify_image(image: np.ndarray, confidence_threshold: float = 0.3) -> tuple[Optional[str], float, str]:
    """
    Classify the activity in an image using YOLOv11 classification model.

    Args:
        image: The image to classify (numpy array)
        confidence_threshold: Minimum confidence for classification

    Returns:
        tuple: (class_name or None, confidence, details_string)
    """
    if classification_model is None or image is None or image.size == 0:
        return None, 0.0, "Image classification unavailable (model not loaded or invalid image)"

    try:
        # Run inference
        results = classification_model(image, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            # Get top prediction
            probs = result.probs
            if probs is not None:
                class_idx = probs.top1
                class_name = result.names[class_idx]
                confidence = float(probs.top1conf)
                
                # Get top 3 predictions for details
                top5_indices = probs.top5
                top5_conf = probs.top5conf
                details_list = []
                for i in range(min(3, len(top5_indices))):
                    idx = top5_indices[i]
                    conf = top5_conf[i]
                    details_list.append(f"{result.names[idx]}({conf:.2f})")
                details = "Image-based: " + ", ".join(details_list)
                
                if confidence >= confidence_threshold:
                    return class_name, confidence, details
                else:
                    return None, confidence, f"{details} (below threshold {confidence_threshold})"
    except Exception as e:
        return None, 0.0, f"Image classification failed: {str(e)}"

    return None, 0.0, "Image classification returned no results"


def classify_activity(landmarks: list[Landmark], image: Optional[np.ndarray] = None) -> dict:
    """
    Classify the activity based on pose landmarks and image classification.
    Returns detailed information combining both methods for LLM processing.

    Args:
        landmarks: List of pose landmarks
        image: Optional image for image-based classification

    Returns:
        dict: ActivityInfo with keys:
            - label: Primary activity label
            - pose_based: Activity from pose landmarks
            - image_based: Activity from image classification (None if unavailable)
            - confidence: Overall confidence score (0.0 to 1.0)
            - details: Detailed information for LLM processing
    """
    pose_activity = "unknown"
    pose_confidence = 0.0
    pose_details = []
    
    # First, try pose-based classification
    if landmarks and len(landmarks) > 0:
        # Priority order: more specific activities first
        if is_waving(landmarks):
            pose_activity = "waving"
            pose_confidence = 0.9  # High confidence for specific gestures
            pose_details.append("Arms raised above shoulders with bent elbows")
        elif is_arms_raised(landmarks):
            pose_activity = "raising_hand"
            pose_confidence = 0.85
            pose_details.append("One or both hands raised above shoulders")
        elif is_sitting(landmarks):
            pose_activity = "sitting"
            pose_confidence = 0.8
            # Calculate knee angles for detail
            left_hip = get_landmark_by_name(landmarks, "left hip")
            left_knee = get_landmark_by_name(landmarks, "left knee")
            left_ankle = get_landmark_by_name(landmarks, "left ankle")
            if left_hip and left_knee and left_ankle:
                angle = calculate_angle(left_hip, left_knee, left_ankle)
                if angle:
                    pose_details.append(f"Knee angle: {angle:.1f}° (sitting range: 50-130°)")
        elif is_standing(landmarks):
            pose_activity = "standing"
            pose_confidence = 0.75
            pose_details.append("Legs relatively straight (knee angle > 140°)")
        else:
            pose_details.append("No specific pose pattern detected")
    else:
        pose_details.append("No pose landmarks available")

    # Try image-based classification if available
    image_activity = None
    image_confidence = 0.0
    image_details = ""
    
    if image is not None:
        image_activity, image_confidence, image_details = classify_image(image)

    # Combine results with detailed information
    # Prioritize pose-based for known activities, blend with image-based info
    if pose_activity != "unknown" and image_activity is not None:
        # Both methods have results - provide blended information
        if pose_activity == image_activity:
            # Agreement between methods - high confidence
            final_label = pose_activity
            final_confidence = min(0.95, (pose_confidence + image_confidence) / 2 + 0.1)
            details = f"Pose-based: {pose_activity} ({pose_confidence:.2f}). {' '.join(pose_details)}. {image_details}. Both methods agree."
        else:
            # Disagreement - prefer pose but include both
            final_label = pose_activity
            final_confidence = pose_confidence * 0.8  # Reduce confidence due to disagreement
            details = f"Pose-based: {pose_activity} ({pose_confidence:.2f}). {' '.join(pose_details)}. {image_details}. Methods disagree - using pose-based."
    elif pose_activity != "unknown":
        # Only pose-based available
        final_label = pose_activity
        final_confidence = pose_confidence
        details = f"Pose-based: {pose_activity} ({pose_confidence:.2f}). {' '.join(pose_details)}. {image_details if image_details else 'No image classification available.'}"
    elif image_activity is not None:
        # Only image-based available
        final_label = image_activity
        final_confidence = image_confidence
        details = f"Pose-based: unknown. {' '.join(pose_details)}. {image_details}. Using image-based classification."
    else:
        # No classification available
        final_label = "unknown"
        final_confidence = 0.0
        details = f"Pose-based: unknown. {' '.join(pose_details)}. {image_details if image_details else 'No image classification available.'}"

    return {
        "label": final_label,
        "pose_based": pose_activity,
        "image_based": image_activity,
        "confidence": final_confidence,
        "details": details
    }
