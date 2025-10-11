from numpy import imag, ndarray
from typing_extensions import TypedDict


class Landmark(TypedDict):
    name: str  # nose, left_shoulder, right_shoulder, etc.
    x: float
    y: float
    z: float
    visibility: float


class Pose(TypedDict):
    landmarks: list[Landmark]


class PersonInfo(TypedDict):
    person_id: str
    face_embedding: ndarray | None  # 128-d vector or None if not computed
    face_location: tuple[
        int, int, int, int, float
    ]  # (top, right, bottom, left, confidence)
    image: ndarray  # cropped face image
    emotions: list[str]  # list of detected emotions
    pose: Pose  # body pose information


class SceneObject(TypedDict):
    label: str  # laptop, chair etc.
    bbox: tuple[int, int, int, int]  # (top, right, bottom, left)
    confidence: float


class FrameInfo(TypedDict):
    image: ndarray
    persons: list[PersonInfo]
    objects: list[SceneObject]


class PipelineState(TypedDict):
    video_path: str
    output_path: str
    fps: float
    frames: list[FrameInfo]
    summary: str
