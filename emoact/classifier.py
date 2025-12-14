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


def classify_image(image: np.ndarray, confidence_threshold: float = 0.3) -> Optional[str]:
    """
    Classify the activity in an image using YOLOv11 classification model.

    Args:
        image: The image to classify (numpy array)
        confidence_threshold: Minimum confidence for classification

    Returns:
        str: The detected activity class or None if confidence is too low
    """
    if classification_model is None or image is None or image.size == 0:
        return None

    try:
        # Run inference
        results = classification_model(image, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            # Get top prediction
            probs = result.probs
            if probs is not None and probs.top1conf >= confidence_threshold:
                # Get class name
                class_idx = probs.top1
                class_name = result.names[class_idx]
                return class_name
    except Exception as e:
        print(f"Warning: Image classification failed: {e}")
        return None

    return None


def classify_activity(landmarks: list[Landmark], image: Optional[np.ndarray] = None) -> str:
    """
    Classify the activity based on pose landmarks and optionally image classification.

    Args:
        landmarks: List of pose landmarks
        image: Optional image crop for image-based classification

    Returns:
        str: The detected activity (e.g., "sitting", "standing", "raising_hand", "waving", "unknown")
    """
    pose_activity = "unknown"
    
    # First, try pose-based classification
    if landmarks and len(landmarks) > 0:
        # Priority order: more specific activities first
        if is_waving(landmarks):
            pose_activity = "waving"
        elif is_arms_raised(landmarks):
            pose_activity = "raising_hand"
        elif is_sitting(landmarks):
            pose_activity = "sitting"
        elif is_standing(landmarks):
            pose_activity = "standing"

    # Try image-based classification if available
    image_activity = None
    if image is not None:
        image_activity = classify_image(image)

    # Combine results: prioritize pose-based for known activities,
    # fall back to image classification if pose is unknown
    if pose_activity != "unknown":
        return pose_activity
    elif image_activity is not None:
        return image_activity
    else:
        return "unknown"
