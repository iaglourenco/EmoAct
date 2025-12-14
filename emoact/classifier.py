"""Activity data collection - gathers raw pose and image data for LLM processing."""

import math
from typing import Optional
from emoact.types import Landmark, ActivityRawData, PoseAngles, ImagePrediction
from ultralytics.models import YOLO
import numpy as np

# Image classification constants
TOP_PREDICTIONS_COUNT = 5  # Number of top predictions to return

# Load YOLOv11 classification model for image-based activity classification
# Model path follows the same pattern as pose.py and objects.py
try:
    classification_model = YOLO("models/yolo11n-cls.pt")
except Exception as e:
    print(f"Warning: Could not load YOLOv11 classification model: {e}")
    print("Image-based classification will be unavailable.")
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


def calculate_all_joint_angles(landmarks: list[Landmark]) -> PoseAngles:
    """
    Calculate all major joint angles from pose landmarks.
    Returns raw angle measurements without any interpretation.

    Args:
        landmarks: List of pose landmarks

    Returns:
        PoseAngles: Dictionary of joint angles in degrees (None if cannot be calculated)
    """
    angles: PoseAngles = {
        "left_elbow": None,
        "right_elbow": None,
        "left_knee": None,
        "right_knee": None,
        "left_hip": None,
        "right_hip": None,
        "left_shoulder": None,
        "right_shoulder": None,
    }

    if not landmarks or len(landmarks) == 0:
        return angles

    # Calculate left elbow angle (shoulder-elbow-wrist)
    left_shoulder = get_landmark_by_name(landmarks, "left shoulder")
    left_elbow = get_landmark_by_name(landmarks, "left elbow")
    left_wrist = get_landmark_by_name(landmarks, "left wrist")
    if left_shoulder and left_elbow and left_wrist:
        angles["left_elbow"] = calculate_angle(left_shoulder, left_elbow, left_wrist)

    # Calculate right elbow angle
    right_shoulder = get_landmark_by_name(landmarks, "right shoulder")
    right_elbow = get_landmark_by_name(landmarks, "right elbow")
    right_wrist = get_landmark_by_name(landmarks, "right wrist")
    if right_shoulder and right_elbow and right_wrist:
        angles["right_elbow"] = calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

    # Calculate left knee angle (hip-knee-ankle)
    left_hip = get_landmark_by_name(landmarks, "left hip")
    left_knee = get_landmark_by_name(landmarks, "left knee")
    left_ankle = get_landmark_by_name(landmarks, "left ankle")
    if left_hip and left_knee and left_ankle:
        angles["left_knee"] = calculate_angle(left_hip, left_knee, left_ankle)

    # Calculate right knee angle
    right_hip = get_landmark_by_name(landmarks, "right hip")
    right_knee = get_landmark_by_name(landmarks, "right knee")
    right_ankle = get_landmark_by_name(landmarks, "right ankle")
    if right_hip and right_knee and right_ankle:
        angles["right_knee"] = calculate_angle(right_hip, right_knee, right_ankle)

    # Calculate left hip angle (shoulder-hip-knee)
    if left_shoulder and left_hip and left_knee:
        angles["left_hip"] = calculate_angle(left_shoulder, left_hip, left_knee)

    # Calculate right hip angle
    if right_shoulder and right_hip and right_knee:
        angles["right_hip"] = calculate_angle(right_shoulder, right_hip, right_knee)

    # Calculate left shoulder angle (elbow-shoulder-hip)
    if left_elbow and left_shoulder and left_hip:
        angles["left_shoulder"] = calculate_angle(left_elbow, left_shoulder, left_hip)

    # Calculate right shoulder angle
    if right_elbow and right_shoulder and right_hip:
        angles["right_shoulder"] = calculate_angle(
            right_elbow, right_shoulder, right_hip
        )

    return angles


def get_image_predictions(image: np.ndarray) -> list[ImagePrediction]:
    """
    Get raw image classification predictions from YOLOv11 model.
    Returns top predictions without any inference or filtering.

    Args:
        image: The image to classify (numpy array)

    Returns:
        list[ImagePrediction]: List of top predictions with class names and confidence scores
    """
    if classification_model is None or image is None or image.size == 0:
        return []

    try:
        # Run inference
        results = classification_model(image, verbose=False)

        if len(results) > 0:
            result = results[0]
            probs = result.probs
            if probs is not None:
                # Get top N predictions
                top_indices = probs.top5
                top_conf = probs.top5conf

                predictions = []
                for i in range(min(TOP_PREDICTIONS_COUNT, len(top_indices))):
                    idx = top_indices[i]
                    conf = top_conf[i]
                    predictions.append(
                        {"class_name": result.names[idx], "confidence": float(conf)}
                    )
                return predictions
    except Exception as e:
        print(f"Warning: Image classification failed: {e}")
        return []

    return []


def collect_activity_data(
    landmarks: list[Landmark], image: Optional[np.ndarray] = None
) -> ActivityRawData:
    """
    Collect raw activity data from pose landmarks and image classification.
    No inference or interpretation - just raw measurements for LLM processing.

    Args:
        landmarks: List of pose landmarks
        image: Optional image for image-based classification

    Returns:
        ActivityRawData: Dictionary with raw pose and image data
    """
    # Collect pose data
    pose_available = landmarks is not None and len(landmarks) > 0
    pose_angles = calculate_all_joint_angles(landmarks if pose_available else [])

    # Collect image classification data
    image_predictions = get_image_predictions(image) if image is not None else []
    image_classification_available = len(image_predictions) > 0

    return {
        "pose_landmarks": landmarks if pose_available else [],
        "pose_angles": pose_angles,
        "image_predictions": image_predictions,
        "pose_available": pose_available,
        "image_classification_available": image_classification_available,
    }
