#!/usr/bin/env python3
"""
Real-time webcam detection system using the EmoAct pipeline.
Detects faces, poses, emotions, and objects in real-time from webcam feed.
"""

import cv2
import numpy as np
from emoact.types import PersonInfo, FrameInfo
from emoact import face, pose, objects, emotions
from emoact.utils import draw_bounding_boxes, draw_text, draw_pose_skeleton


# Define consistent color scheme
COLORS = {
    "face_bbox": (255, 100, 100),  # Light red/pink for face boxes
    "face_text": (255, 150, 150),  # Lighter red for face text
    "emotion_text": (100, 255, 255),  # Yellow for emotions
    "object_bbox": (100, 255, 100),  # Green for object boxes
    "object_text": (100, 255, 100),  # Green for object text
    "info_text": (255, 255, 255),  # White for info text
}


def process_frame(image: np.ndarray) -> FrameInfo:
    """
    Process a single frame and detect faces, poses, and objects.

    Args:
        image: The input frame from webcam

    Returns:
        FrameInfo with detected persons and objects
    """
    frame_info: FrameInfo = {"image": image.copy(), "persons": [], "objects": []}

    # 1. Detect faces
    face_locations = face.detect_faces(image)
    for left, top, right, bottom, embedding, confidence, gender, age in face_locations:
        person_info: PersonInfo = {
            "face_location": (left, top, right, bottom, confidence),
            "face_embedding": embedding,
            "image": image[top:bottom, left:right],
            "gender": gender,
            "age": age,
            "emotions": [],
            "pose": {"landmarks": []},
            "person_id": "",
        }
        frame_info["persons"].append(person_info)

    # 2. Detect poses and match to persons
    poses = pose.detect_poses_in_frame(image)
    for bbox, keypoints in poses:
        left, top, right, bottom = bbox

        # Find the closest person by IoU or center distance
        best_person = None
        best_iou = 0.0

        if len(frame_info["persons"]) == 1:
            best_person = frame_info["persons"][0]
            best_iou = 1.0
        else:
            for person in frame_info["persons"]:
                if person["face_location"]:
                    face_left, face_top, face_right, face_bottom, _ = person[
                        "face_location"
                    ]
                    # Compute IoU
                    ix1 = max(face_left, left)
                    iy1 = max(face_top, top)
                    ix2 = min(face_right, right)
                    iy2 = min(face_bottom, bottom)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    intersection = iw * ih
                    union = (
                        (face_right - face_left) * (face_bottom - face_top)
                        + (right - left) * (bottom - top)
                        - intersection
                    )
                    iou = intersection / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_person = person

        if best_person and best_iou > 0.01:  # IoU threshold
            best_person["pose"]["landmarks"] = [
                {
                    "x": float(keypoints[i]),
                    "y": float(keypoints[i + 1]),
                    "confidence": float(keypoints[i + 2]),
                    "name": pose.pose_landmark_names[i // 3],
                }
                for i in range(0, len(keypoints), 3)
            ]

    # 3. Detect objects
    detected_objects = objects.detect_objects_in_frame(image)
    for bbox, cls, conf in detected_objects:
        left, top, right, bottom = bbox
        if conf > 0.5:
            frame_info["objects"].append(
                {
                    "bbox": (left, top, right, bottom),
                    "label": objects.model.names[cls],
                    "confidence": conf,
                }
            )

    # 4. Detect emotions
    for person in frame_info["persons"]:
        if person["face_location"]:
            left, top, right, bottom, _ = person["face_location"]
            face_img = image[top:bottom, left:right]
            if face_img.size > 0:
                emotion = emotions.detect_emotion(face_img)
                person["emotions"].append(emotion)

    return frame_info


def draw_detections(image: np.ndarray, frame_info: FrameInfo) -> np.ndarray:
    """
    Draw all detections on the image.

    Args:
        image: The input image
        frame_info: Detection results

    Returns:
        Image with drawn detections
    """
    for person in frame_info["persons"]:
        # Draw face bounding box
        if person["face_location"]:
            left, top, right, bottom, confidence = person["face_location"]

            # Draw face box
            draw_bounding_boxes(
                image,
                [(left, top, right, bottom)],
                color=COLORS["face_bbox"],
                thickness=2,
            )

            # Draw person ID and confidence
            person_id = person["person_id"] if person["person_id"] else "Unknown"
            text = f"ID: {person_id} ({confidence:.2f})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Background rectangle for text
            cv2.rectangle(
                image,
                (left, top - text_size[1] - 8),
                (left + text_size[0] + 4, top - 2),
                COLORS["face_bbox"],
                -1,
            )

            draw_text(
                image,
                text,
                position=(left + 2, top - 5),
                font_scale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

        # Draw pose skeleton
        if person["pose"]["landmarks"]:
            landmarks = person["pose"]["landmarks"]
            draw_pose_skeleton(image, landmarks, confidence_threshold=0.3)

        # Draw emotions
        if person["emotions"]:
            left, top, right, bottom, _ = person["face_location"]
            emotions_text = ", ".join(person["emotions"])
            text = f"Emotions: {emotions_text}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # Background rectangle
            cv2.rectangle(
                image,
                (left, bottom + 5),
                (left + text_size[0] + 4, bottom + text_size[1] + 13),
                COLORS["emotion_text"],
                -1,
            )

            draw_text(
                image,
                text,
                position=(left + 2, bottom + text_size[1] + 8),
                font_scale=0.5,
                color=(0, 0, 0),
                thickness=1,
            )
        # Draw gender and age
        if person["gender"] is not None and person["age"] is not None:
            gender_text = "Male" if person["gender"] == 1 else "Female"
            age_text = f", Age: {person['age']}"
            text = f"{gender_text}{age_text}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            # Background rectangle for text
            cv2.rectangle(
                image,
                (left, bottom + 30),
                (left + text_size[0] + 4, bottom + 30 + text_size[1] + 8),
                COLORS["face_text"],
                -1,
            )
            draw_text(
                image,
                text,
                position=(left + 2, bottom + 30 + text_size[1] + 3),
                font_scale=0.5,
                color=(255, 255, 255),
                thickness=1,
            )

    # Draw detected objects
    for obj in frame_info["objects"]:
        left, top, right, bottom = obj["bbox"]

        # Draw bounding box
        draw_bounding_boxes(
            image,
            [(left, top, right, bottom)],
            color=COLORS["object_bbox"],
            thickness=2,
        )

        # Prepare label text
        text = f"{obj['label']} ({obj['confidence']:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Background rectangle for text
        cv2.rectangle(
            image,
            (left, top - text_size[1] - 8),
            (left + text_size[0] + 4, top - 2),
            COLORS["object_bbox"],
            -1,
        )

        draw_text(
            image,
            text,
            position=(left + 2, top - 5),
            font_scale=0.5,
            color=(0, 0, 0),
            thickness=1,
        )

    return image


def main():
    """
    Main function to run real-time webcam detection.
    """
    print("Starting webcam detection...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")

    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set camera resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    fps_display = 0
    fps_start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Process the frame
        frame_info = process_frame(frame)

        # Draw detections
        output_frame = draw_detections(frame_info["image"], frame_info)

        # Calculate and display FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps_end_time = cv2.getTickCount()
            time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
            fps_display = 10 / time_diff
            fps_start_time = fps_end_time

        # Draw FPS and stats
        stats_text = f"FPS: {fps_display:.1f} | Persons: {len(frame_info['persons'])} | Objects: {len(frame_info['objects'])}"
        text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        # Background for stats
        cv2.rectangle(
            output_frame,
            (10, 10),
            (10 + text_size[0] + 10, 10 + text_size[1] + 10),
            (0, 0, 0),
            -1,
        )

        draw_text(
            output_frame,
            stats_text,
            position=(15, 10 + text_size[1]),
            font_scale=0.7,
            color=COLORS["info_text"],
            thickness=2,
        )

        # Display the frame
        cv2.imshow("EmoAct - Real-time Detection", output_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("s"):
            filename = f"snapshot_{frame_count}.jpg"
            cv2.imwrite(filename, output_frame)
            print(f"Saved snapshot to {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam detection stopped.")


if __name__ == "__main__":
    main()
