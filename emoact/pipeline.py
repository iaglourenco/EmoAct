from emoact.types import PersonInfo, PipelineState
from langgraph.graph import StateGraph, START, END

from emoact.utils import draw_graph


def load_video(state: PipelineState):
    from emoact import video_io

    video_path = state["video_path"]
    frames, state["fps"] = video_io.load_video(video_path)
    state["frames"] = [
        {"image": frame, "persons": [], "objects": []} for frame in frames
    ]
    return state


def detect_faces(state: PipelineState):
    from emoact import face_recog

    frames = [frame_info["image"] for frame_info in state["frames"]]
    results = face_recog.detect_faces(frames)
    for frame_info, faces in zip(state["frames"], results):
        for i, (top, right, bottom, left, embedding, confidence) in enumerate(faces):
            person = PersonInfo(
                person_id=f"person_{i+1}",
                face_embedding=embedding,
                face_location=(top, right, bottom, left, confidence),
                image=frame_info["image"][top:bottom, left:right],
                emotions=[],
                pose={"landmarks": []},
            )
            if confidence > 0.95:  # Only consider faces with high confidence
                frame_info["persons"].append(person)

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
                    image, f"ID: {person['person_id']}", position=(left, top - 10)
                )
            # Draw emotions
            if person["emotions"]:
                top, right, bottom, left, _ = person["face_location"]
                emotions_text = ", ".join(person["emotions"])
                draw_text(
                    image,
                    f"Emotions: {emotions_text}",
                    position=(left, bottom + 20),
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
                position=(left, top - 10),
                color=(0, 255, 0),
            )

        frame_info["image"] = image

    return state


def detect_poses(state: PipelineState):
    # Placeholder for pose detection logic
    # This function should update the 'pose' field in each PersonInfo
    return state


def detect_objects(state: PipelineState):
    # Placeholder for object detection logic
    # This function should update the 'objects' field in each FrameInfo
    return state


def detect_emotions(state: PipelineState):
    # Placeholder for emotion detection logic
    # This function should update the 'emotions' field in each PersonInfo
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


def has_persons(state: PipelineState) -> bool:
    return any(len(frame_info["persons"]) > 0 for frame_info in state["frames"])


# Nós
graph_builder.add_node("load_video", load_video)
graph_builder.add_node("detect_faces", detect_faces)
graph_builder.add_node("detect_poses", detect_poses)
graph_builder.add_node("detect_objects", detect_objects)
graph_builder.add_node("detect_emotions", detect_emotions)
graph_builder.add_node("draw", draw)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("save_video", save_video)

# Arestas
graph_builder.add_edge(START, "load_video")
graph_builder.add_edge("load_video", "detect_faces")
graph_builder.add_conditional_edges(
    "detect_faces", has_persons, ["detect_poses", "detect_emotions"]
)
graph_builder.add_edge("detect_faces", "detect_objects")
graph_builder.add_edge(["detect_poses", "detect_emotions", "detect_objects"], "draw")
graph_builder.add_edge("draw", "summarize")
graph_builder.add_edge("summarize", "save_video")

graph = graph_builder.compile()
draw_graph(graph)

exit()

if __name__ == "__main__":
    initial_state: PipelineState = {
        "video_path": "cropped.mp4",
        "output_path": "output.mp4",
        "fps": 0.0,
        "frames": [],
        "summary": "",
    }
    final_state = graph.invoke(initial_state)
    print("Pipeline completed.")
