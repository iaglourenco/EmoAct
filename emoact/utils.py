import cv2
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))


def draw_graph(graph, filename="graph"):
    """Draws the state graph to a file."""
    graph_bytes = graph.get_graph().draw_mermaid_png()
    with open(f"{filename}.png", "wb") as f:
        f.write(graph_bytes)


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on the image.
    Args:
        image: The image to draw on.
        boxes: List of bounding boxes, each defined as (left, top, right, bottom).
        color: Color of the bounding box.
        thickness: Thickness of the bounding box lines.
    Returns:
        The image with bounding boxes drawn.
    """
    for left, top, right, bottom in boxes:
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
    return image


def draw_landmarks(image, landmarks, color=(0, 0, 255), radius=3):
    for landmark in landmarks:
        x, y = int(landmark["x"]), int(landmark["y"])
        cv2.circle(image, (x, y), radius, color, -1)
    return image


def draw_pose_skeleton(image, landmarks, confidence_threshold=0.5):
    """Draw pose skeleton with connections between landmarks.

    Args:
        image: The image to draw on.
        landmarks: List of landmarks with 'x', 'y', 'confidence', and 'name' fields.
        confidence_threshold: Minimum confidence to draw landmarks and connections.

    Returns:
        The image with pose skeleton drawn.
    """
    # Define pose connections (COCO-style keypoint connections)
    connections = [
        # Face
        ("nose", "left eye"),
        ("nose", "right eye"),
        ("left eye", "left ear"),
        ("right eye", "right ear"),
        # Upper body
        ("left shoulder", "right shoulder"),
        ("left shoulder", "left elbow"),
        ("right shoulder", "right elbow"),
        ("left elbow", "left wrist"),
        ("right elbow", "right wrist"),
        # Torso
        ("left shoulder", "left hip"),
        ("right shoulder", "right hip"),
        ("left hip", "right hip"),
        # Lower body
        ("left hip", "left knee"),
        ("right hip", "right knee"),
        ("left knee", "left ankle"),
        ("right knee", "right ankle"),
    ]

    # Create a dictionary for quick landmark lookup
    landmark_dict = {lm["name"]: lm for lm in landmarks}

    # Define colors for different body parts
    color_map = {
        "face": (255, 200, 100),  # Light blue for face
        "upper": (100, 255, 100),  # Green for upper body
        "torso": (255, 100, 255),  # Magenta for torso
        "lower": (100, 200, 255),  # Orange for lower body
    }

    # Draw connections
    for start_name, end_name in connections:
        if start_name in landmark_dict and end_name in landmark_dict:
            start_lm = landmark_dict[start_name]
            end_lm = landmark_dict[end_name]

            # Only draw if both landmarks have sufficient confidence
            if (
                start_lm["confidence"] > confidence_threshold
                and end_lm["confidence"] > confidence_threshold
            ):

                start_point = (int(start_lm["x"]), int(start_lm["y"]))
                end_point = (int(end_lm["x"]), int(end_lm["y"]))

                # Determine color based on body part
                if "eye" in start_name or "ear" in start_name or "nose" in start_name:
                    color = color_map["face"]
                elif (
                    "shoulder" in start_name
                    or "elbow" in start_name
                    or "wrist" in start_name
                ):
                    color = color_map["upper"]
                elif "hip" in start_name and "hip" in end_name:
                    color = color_map["torso"]
                elif "shoulder" in start_name and "hip" in end_name:
                    color = color_map["torso"]
                else:
                    color = color_map["lower"]

                cv2.line(image, start_point, end_point, color, 2, cv2.LINE_AA)

    # Draw landmarks on top of connections
    for landmark in landmarks:
        if landmark["confidence"] > confidence_threshold:
            x, y = int(landmark["x"]), int(landmark["y"])
            # Draw outer circle (border)
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
            # Draw inner circle (landmark)
            cv2.circle(image, (x, y), 3, (0, 100, 255), -1)

    return image


def draw_text(
    image,
    text,
    position=(10, 30),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(255, 255, 255),
    thickness=2,
):
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image
