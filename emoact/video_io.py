import cv2


def load_video(video_path: str) -> tuple[list, float]:
    if video_path is None:
        raise ValueError("No video path provided")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def save_video(frames, output_path: str, fps: float):
    if not frames:
        raise ValueError("No frames to save")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    for frame in frames:
        out.write(frame)

    out.release()
