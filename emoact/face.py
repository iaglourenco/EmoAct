import numpy as np
import insightface
import sys, os


class SuppressStdout:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._stdout


# Supress insightface loading messages
with SuppressStdout():
    app = insightface.app.FaceAnalysis(
        "buffalo_l", root="./", providers=["CUDAExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))


def detect_faces(frame: np.ndarray, threshold: float = 0.5):
    """
    Detect faces in a frame.

    Returns:
        List of tuples: (left, top, right, bottom, embedding, confidence, gender, age)
    """
    faces = app.get(frame)
    face_locations = []
    for face in faces:
        if face.det_score >= threshold:
            bbox = face.bbox.astype(int)
            gender = face.gender.astype(int)
            # InsightFace returns [x1, y1, x2, y2] which is [left, top, right, bottom]
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            face_locations.append(
                (
                    left,
                    top,
                    right,
                    bottom,
                    face.embedding,
                    face.det_score,
                    gender,
                    face.age,
                )
            )
    return face_locations
