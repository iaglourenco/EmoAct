from numpy import ndarray
from typing import TypedDict


class Landmark(TypedDict):
    name: str  # nose, left_shoulder, right_shoulder, etc.
    x: float
    y: float
    confidence: float


class Pose(TypedDict):
    landmarks: list[Landmark]


class ActivityInfo(TypedDict):
    label: str  # Primary activity label (e.g., "standing", "sitting", "raising_hand", "unknown")
    pose_based: str  # Activity detected from pose landmarks
    image_based: str | None  # Activity detected from image classification (None if unavailable)
    confidence: float  # Overall confidence score (0.0 to 1.0)
    details: str  # Additional details for LLM processing


class PersonInfo(TypedDict):
    person_id: str
    gender: int  # 0 for female, 1 for male,
    age: int  # in years
    face_embedding: ndarray  # 128-d vector or None if not computed
    face_location: tuple[
        int, int, int, int, float
    ]  # (left, top, right, bottom, confidence)
    image: ndarray  # cropped face image
    emotions: list[str]  # list of detected emotions
    pose: Pose  # body pose information
    activity: str | ActivityInfo  # detected activity (legacy: str, enhanced: ActivityInfo dict)


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
