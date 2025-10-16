import numpy as np
import insightface

app = insightface.app.FaceAnalysis("buffalo_l", providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_faces(frame: np.ndarray, threshold: float = 0.5):
    """Detect faces for each frame and return their locations in (top, right, bottom, left) format."""
    faces = app.get(frame)
    face_locations = []
    for face in faces:
        if face.det_score >= threshold:
            bbox = face.bbox.astype(int)
            top, left, bottom, right = bbox[1], bbox[0], bbox[3], bbox[2]
            face_locations.append((top, right, bottom, left))
    return face_locations


if __name__ == "__main__":
    import cv2

    # Example usage
    video_path = "/home/prime/projects/FIAP-TC4/input_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_locations = detect_faces(frame)
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
