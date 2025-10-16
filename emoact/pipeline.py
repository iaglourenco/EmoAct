from emoact.types import PersonInfo, PipelineState
from langgraph.graph import StateGraph, START, END

from emoact.utils import calculate_bbox_center_distance, cosine_similarity, draw_graph


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

    return state


def draw(state: PipelineState):
    from emoact.utils import draw_bounding_boxes, draw_text

    for frame_info in state["frames"]:
        image = frame_info["image"]
        for person in frame_info["persons"]:
            # Draw face bounding box and ID
            if person["face_location"]:
                top, right, bottom, left, confidence = person["face_location"]
                draw_bounding_boxes(image, [(top, right, bottom, left, confidence)])
                draw_text(
                    image, f"ID: {person['person_id']}", position=(top - 10, left)
                )
            # Draw emotions
            if person["emotions"]:
                top, right, bottom, left, _ = person["face_location"]
                emotions_text = ", ".join(person["emotions"])
                draw_text(
                    image,
                    f"Emotions: {emotions_text}",
                    position=(bottom + 20, left),
                    color=(255, 0, 0),
                )

        # Draw objects
        for obj in frame_info["objects"]:
            top, right, bottom, left = obj["bbox"]
            draw_bounding_boxes(
                image,
                [(top, right, bottom, left, obj["confidence"])],
                color=(0, 255, 0),
            )
            draw_text(
                image,
                f"{obj['label']} ({obj['confidence']:.2f})",
                position=(top - 10, left),
                color=(0, 255, 0),
            )

        frame_info["image"] = image

    return state


def detect_poses(state: PipelineState):
    # Placeholder for pose detection logic
    # TODO: This function should update the 'pose' field in each PersonInfo
    return state


def detect_objects(state: PipelineState):
    # Placeholder for object detection logic
    # TODO: This function should update the 'objects' field in each FrameInfo
    return state


def detect_emotions(state: PipelineState):
    # Placeholder for emotion detection logic
    # TODO: This function should update the 'emotions' field in each PersonInfo
    return state


def track_faces(state: PipelineState) -> PipelineState:

    # histórico global de embeddings e IDs
    known_persons = (
        []
    )  # list of dicts: {"person_id": str, "embedding": np.ndarray, "last_bbox": tuple}
    next_person_id = 0
    similarity_threshold: float = 0.6

    for frame_info in state["frames"]:
        for person in frame_info["persons"]:
            best_match_id = None
            best_score = -1.0

            for kp in known_persons:
                score = cosine_similarity(person["face_embedding"], kp["embedding"])
                # opcional: penaliza muito se bbox estiver longe (melhora tracking em movimento rápido)
                top, right, bottom, left, _ = person["face_location"]
                prev_top, prev_right, prev_bottom, prev_left, _ = kp["last_bbox"]
                center_dist = calculate_bbox_center_distance(
                    (top, right, bottom, left),
                    (prev_top, prev_right, prev_bottom, prev_left),
                )
                # normaliza distância em pixels (opcional, ajustável)
                score -= center_dist * 0.001

                if score > best_score:
                    best_score = score
                    best_match_id = kp["person_id"]

            if best_score >= similarity_threshold and best_match_id is not None:
                # match encontrado
                person["person_id"] = best_match_id
                # atualiza embedding e bbox histórico
                for kp in known_persons:
                    if kp["person_id"] == best_match_id:
                        kp["embedding"] = person["face_embedding"]
                        kp["last_bbox"] = person["face_location"]
                        break
            else:
                # nova pessoa
                person_id = f"person_{next_person_id}"
                next_person_id += 1
                person["person_id"] = person_id
                known_persons.append(
                    {
                        "person_id": person_id,
                        "embedding": person["face_embedding"],
                        "last_bbox": person["face_location"],
                    }
                )

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
graph_builder.add_conditional_edges(
    "detect_faces",
    has_persons,
    {"has_faces": "detect_poses", "no_faces": "detect_objects"},
)
graph_builder.add_edge("detect_poses", "detect_emotions")

graph_builder.add_edge("detect_emotions", "detect_objects")
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
