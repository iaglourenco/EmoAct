import cv2


def draw_graph(graph, filename="graph"):
    """Draws the state graph to a file."""
    graph_bytes = graph.get_graph().draw_mermaid_png()
    with open(f"{filename}.png", "wb") as f:
        f.write(graph_bytes)


def draw_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    for top, right, bottom, left, _ in boxes:
        cv2.rectangle(image, (left, top), (right, bottom), color, thickness)
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
