from emoact.types import PersonInfo, PipelineState
from langgraph.graph import StateGraph, START, END

from emoact.utils import draw_graph


def load_video(state: PipelineState):
    from emoact import video_io

    video_path = state["video_path"]
    frames, state["fps"] = video_io.load_video(video_path)
    state["frames"] = [
        {"image": frame, "persons": [], "objects": []}
        for frame in frames  # [:100]  # limit frames for testing
    ]
    return state


def detect_faces(state: PipelineState):
    from emoact import face

    for frame_info in state["frames"]:
        image = frame_info["image"]
        face_locations = face.detect_faces(image)
        for left, top, right, bottom, embedding, confidence in face_locations:
            person_info: PersonInfo = {
                "face_location": (left, top, right, bottom, confidence),
                "face_embedding": embedding,
                "image": image[top:bottom, left:right],
                "emotions": [],
                "pose": {"landmarks": []},
                "person_id": "",
            }
            frame_info["persons"].append(person_info)

    return state


def detect_poses(state: PipelineState):
    from emoact import pose

    for frame_info in state["frames"]:
        image = frame_info["image"]
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
    return state


def detect_objects(state: PipelineState):
    from emoact import objects

    for frame_info in state["frames"]:
        image = frame_info["image"]
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
    return state


def detect_emotions(state: PipelineState):
    from emoact import emotions

    for frame_info in state["frames"]:
        image = frame_info["image"]
        for person in frame_info["persons"]:
            if person["face_location"]:
                left, top, right, bottom, _ = person["face_location"]
                face_img = image[top:bottom, left:right]
                if face_img.size > 0:
                    emotion = emotions.detect_emotion(face_img)
                    person["emotions"].append(emotion)
    return state


def draw(state: PipelineState):
    from emoact.utils import draw_bounding_boxes, draw_text, draw_pose_skeleton
    import cv2

    # Define consistent color scheme
    COLORS = {
        "face_bbox": (255, 100, 100),  # Light red/pink for face boxes
        "face_text": (255, 150, 150),  # Lighter red for face text
        "emotion_text": (100, 255, 255),  # Yellow for emotions
        "object_bbox": (100, 255, 100),  # Green for object boxes
        "object_text": (100, 255, 100),  # Green for object text
        "background": (0, 0, 0),  # Black background for text
    }

    for frame_info in state["frames"]:
        image = frame_info["image"]

        for person in frame_info["persons"]:
            # Draw face bounding box with enhanced appearance
            if person["face_location"]:
                left, top, right, bottom, confidence = person["face_location"]

                # Draw rounded rectangle effect (thick border)
                draw_bounding_boxes(
                    image,
                    [(left, top, right, bottom)],
                    color=COLORS["face_bbox"],
                    thickness=2,
                )

                # Draw semi-transparent background for text
                person_id = person["person_id"] if person["person_id"] else "Unknown"
                text = f"ID: {person_id} ({confidence:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Background rectangle for better text visibility
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

            # Draw pose skeleton with connections
            if person["pose"]["landmarks"]:
                landmarks = person["pose"]["landmarks"]
                draw_pose_skeleton(image, landmarks, confidence_threshold=0.3)

            # Draw emotions with enhanced appearance
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

        # Draw objects with consistent styling
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

        frame_info["image"] = image

    return state


def track_faces(state: PipelineState) -> PipelineState:
    # Placeholder for face tracking logic
    # TODO: Track faces across frames and assign consistent person_id

    return state


def summarize(state: PipelineState):
    total_frames = len(state["frames"])
    total_persons = sum(len(frame_info["persons"]) for frame_info in state["frames"])
    state["summary"] = (
        f"Processed {total_frames} frames with {total_persons} detected persons."
    )
    # TODO: Add more detailed summary information if needed by calling a language model
    return state


def save_video(state: PipelineState):
    from emoact import video_io

    frames = [frame_info["image"] for frame_info in state["frames"]]
    output_path = state["output_path"]
    fps = state["fps"]
    video_io.save_video(frames, output_path, fps)
    return state


graph_builder = StateGraph(PipelineState)


def has_persons(state: PipelineState):
    return (
        "has_faces"
        if any(len(f["persons"]) > 0 for f in state["frames"])
        else "no_faces"
    )


# Nós
graph_builder.add_node("load_video", load_video)
graph_builder.add_node("detect_faces", detect_faces)
graph_builder.add_node("detect_poses", detect_poses)
graph_builder.add_node("detect_objects", detect_objects)
graph_builder.add_node("detect_emotions", detect_emotions)
graph_builder.add_node("track_faces", track_faces)
graph_builder.add_node("draw", draw)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("save_video", save_video)

# Arestas
graph_builder.add_edge(START, "load_video")
graph_builder.add_edge("load_video", "detect_faces")

# Após detect_faces, executar em paralelo quando há faces
# Usamos add_edge para cada nó para criar fan-out
graph_builder.add_edge("detect_faces", "detect_poses")
graph_builder.add_edge("detect_faces", "detect_emotions")
graph_builder.add_edge("detect_faces", "detect_objects")

# Fan-in: Todos convergem para track_faces
graph_builder.add_edge("detect_poses", "track_faces")
graph_builder.add_edge("detect_emotions", "track_faces")
graph_builder.add_edge("detect_objects", "track_faces")

graph_builder.add_edge("track_faces", "draw")
graph_builder.add_edge("draw", "summarize")
graph_builder.add_edge("summarize", "save_video")

graph = graph_builder.compile()
draw_graph(graph)

if __name__ == "__main__":
    initial_state: PipelineState = {
        "video_path": "input_video.mp4",
        "output_path": "output.mp4",
        "fps": 0.0,
        "frames": [],
        "summary": "",
    }
    for event in graph.stream(initial_state):
        if event:
            node_name = list(event.keys())[0]
            state = event[node_name]
            print(f"Node '{node_name}' completed.")
