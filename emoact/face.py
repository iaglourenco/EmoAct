import numpy as np
import insightface

app = insightface.app.FaceAnalysis("buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_faces(
    frame: np.ndarray, threshold: float = 0.5
) -> list[tuple[int, int, int, int, np.ndarray, float]]:
    """
    Detect faces in a frame.

    Returns:
        List of tuples: (left, top, right, bottom, embedding, confidence)
    """
    faces = app.get(frame)
    face_locations = []
    for face in faces:
        if face.det_score >= threshold:
            bbox = face.bbox.astype(int)
            # InsightFace returns [x1, y1, x2, y2] which is [left, top, right, bottom]
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            face_locations.append(
                (left, top, right, bottom, face.embedding, face.det_score)
            )
    return face_locations
