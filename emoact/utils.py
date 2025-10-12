import cv2
import numpy as np


def calculate_bbox_center_distance(bbox1, bbox2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.

    Args:
        bbox1: tuple (top, right, bottom, left) for first bounding box
        bbox2: tuple (top, right, bottom, left) for second bounding box

    Returns:
        float: Euclidean distance between the centers
    """
    top1, right1, bottom1, left1 = bbox1
    top2, right2, bottom2, left2 = bbox2

    center_x1 = (left1 + right1) / 2
    center_y1 = (top1 + bottom1) / 2
    center_x2 = (left2 + right2) / 2
    center_y2 = (top2 + bottom2) / 2

    return np.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)


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
    for top, right, bottom, left, _ in boxes:
        cv2.rectangle(image, (top, left), (right, bottom), color, thickness)
    return image


def draw_landmarks(image, landmarks, color=(0, 0, 255), radius=3):
    for landmark in landmarks:
        x, y = int(landmark["x"]), int(landmark["y"])
        cv2.circle(image, (x, y), radius, color, -1)
    return image


def draw_text(
    image,
    text,
    position=(10, 30),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    color=(255, 255, 255),
    thickness=2,
):
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image
