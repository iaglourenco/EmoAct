from tabnanny import verbose
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
    results = model(frame, verbose=False)
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
