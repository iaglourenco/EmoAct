from ultralytics.models import YOLO

model = YOLO("models/yolo11n-pose.pt")

pose_landmark_names = [
    "nose",
    "left eye",
    "right eye",
    "left ear",
    "right ear",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
]


def detect_poses_in_frame(frame):
    """
    Detect poses in a frame.

    Returns:
        List of tuples: (bbox, keypoints)
        where bbox is (left, top, right, bottom)
        and keypoints is a flattened array of [x, y, confidence] values
    """
    results = model(frame)
    poses = []
    for result in results:
        keypoints = result.keypoints
        boxes = result.boxes
        for box, kps in zip(boxes, keypoints):
            # YOLO xyxy format: [left, top, right, bottom]
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            kps = kps.data.cpu().numpy().flatten()
            poses.append((bbox, kps))

    return poses


if __name__ == "__main__":
    video_path = "cropped.mp4"
    import cv2

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        poses = detect_poses_in_frame(frame)
        for bbox, keypoints in poses:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for i in range(0, len(keypoints), 3):
                x, y, conf = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
