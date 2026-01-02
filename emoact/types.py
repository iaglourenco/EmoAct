from numpy import ndarray
from typing import TypedDict


class Landmark(TypedDict):
    name: str  # nose, left_shoulder, right_shoulder, etc.
    x: float
    y: float
    confidence: float


class Pose(TypedDict):
    landmarks: list[Landmark]


class PoseAngles(TypedDict):
    left_elbow: float | None  # Angle in degrees
    right_elbow: float | None
    left_knee: float | None
    right_knee: float | None
    left_hip: float | None
    right_hip: float | None
    left_shoulder: float | None
    right_shoulder: float | None


class ImagePrediction(TypedDict):
    class_name: str
    confidence: float


class ActivityRawData(TypedDict):
    """Raw data for LLM processing - no inference, just measurements"""

    pose_landmarks: list[Landmark]  # Raw pose landmark data
    pose_angles: PoseAngles  # Calculated joint angles
    image_predictions: list[ImagePrediction]  # Top predictions from YOLO classifier
    pose_available: bool  # Whether pose data is available
    image_classification_available: (
        bool  # Whether image classification ran successfully
    )


class PersonInfo(TypedDict):
    person_id: str
    gender: int  # 0 for female, 1 for male,
    age: int  # in years
    face_embedding: ndarray  # 512-d vector or None if not computed
    face_location: tuple[
        int, int, int, int, float
    ]  # (left, top, right, bottom, confidence)
    image: ndarray  # cropped face image
    emotions: list[str]  # list of detected emotions
    pose: Pose  # body pose information
    activity: ActivityRawData  # Raw activity data for LLM processing


class SceneObject(TypedDict):
    label: str  # laptop, chair etc.
    bbox: tuple[int, int, int, int]  # (left, top, right, bottom)
    confidence: float


class FrameInfo(TypedDict):
    image: ndarray
    persons: list[PersonInfo]
    objects: list[SceneObject]


class PipelineState(TypedDict):
    video_path: str
    output_path: str
    transcription: str
    fps: float
    frames: list[FrameInfo]
    summary: str

    object_conf_threshold: float  # Confidence threshold for object detection
    pose_conf_threshold: float  # Confidence threshold for pose landmarks
