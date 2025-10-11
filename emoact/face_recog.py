from typing import List
from deepface.modules import representation
import numpy as np


# recog model Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet
# detector Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yolov11n', 'yolov11s', 'yolov11m', 'centerface' or 'skip'
def detect_faces(
    frames: List[np.ndarray],
    model_name="VGG-Face",
    detector_backend: str = "mtcnn",
) -> List[List[tuple[int, int, int, int, np.ndarray, float]]]:
    """Detect faces and embeddings in an image using DeepFace."""

    face_objs = representation.represent(
        frames,
        model_name=model_name,
        detector_backend=detector_backend,
        align=True,
        enforce_detection=False,
    )

    # Get positions in (top, right, bottom, left, embedding, confidence) format
    # Return list (list of faces per image) of list (faces per image) of tuples (top, right, bottom, left, embedding, confidence)
    faces = [
        [
            (
                int(face_obj["facial_area"]["x"]),  # type: ignore
                int(face_obj["facial_area"]["x"]) + int(face_obj["facial_area"]["w"]),  # type: ignore
                int(face_obj["facial_area"]["y"]) + int(face_obj["facial_area"]["h"]),  # type: ignore
                int(face_obj["facial_area"]["y"]),  # type: ignore
                np.array(face_obj["embedding"]),  # type: ignore
                float(face_obj.get("face_confidence", 1.0)),  # type: ignore
            )
            for face_obj in frame_faces
        ]
        for frame_faces in face_objs
    ]

    return faces
