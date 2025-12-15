from emoact.types import PersonInfo, PipelineState
from langgraph.graph import StateGraph, START

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
        for (
            left,
            top,
            right,
            bottom,
            embedding,
            confidence,
            gender,
            age,
        ) in face_locations:
            person_info: PersonInfo = {
                "face_location": (left, top, right, bottom, confidence),
                "face_embedding": embedding,
                "gender": gender,
                "age": age,
                "image": image[top:bottom, left:right],
                "emotions": [],
                "pose": {"landmarks": []},
                "person_id": "",
                "activity": {
                    "pose_landmarks": [],
                    "pose_angles": {
                        "left_elbow": None,
                        "right_elbow": None,
                        "left_knee": None,
                        "right_knee": None,
                        "left_hip": None,
                        "right_hip": None,
                        "left_shoulder": None,
                        "right_shoulder": None,
                    },
                    "image_predictions": [],
                    "pose_available": False,
                    "image_classification_available": False,
                },
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
            if best_person and best_iou > 0.01:
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

            if conf > state["object_conf_threshold"]:
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

    # Define color scheme
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
                text = f"ID: {person_id}"
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
                draw_pose_skeleton(
                    image, landmarks, confidence_threshold=state["pose_conf_threshold"]
                )

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
    """Track faces across frames and assign consistent person_id."""
    from emoact.tracker import track_faces_in_video

    # Track faces across all frames
    state["frames"] = track_faces_in_video(
        state["frames"],
        similarity_threshold=0.6,
        max_distance_threshold=200.0,
        max_frames_missing=30,
    )

    return state


def classify_activities(state: PipelineState):
    """
    Collect raw activity data from pose landmarks and YOLOv11-cls model.
    No inference - just raw data collection for LLM processing.
    """
    from emoact.classifier import collect_activity_data

    for frame_info in state["frames"]:
        image = frame_info["image"]
        for person in frame_info["persons"]:
            # Collect raw data from both pose landmarks and image
            # Pass full frame for image classification (not just face crop)
            activity_data = collect_activity_data(
                person["pose"]["landmarks"], image=image
            )
            person["activity"] = activity_data

    return state


def transcribe_audio(state: PipelineState):
    from emoact.audio import transcribe_video

    video_path = state["video_path"]
    transcription = transcribe_video(video_path, model="base")
    state["transcription"] = str(transcription)
    return state


def summarize(state: PipelineState):
    from emoact.llm import generate_video_summary, prepare_frame_data_summary

    total_frames = len(state["frames"])
    total_persons = sum(len(frame_info["persons"]) for frame_info in state["frames"])

    # Calculate video duration
    video_duration = total_frames / state["fps"] if state["fps"] > 0 else 0

    # Prepare condensed frame data summary
    frame_data_summary = prepare_frame_data_summary(state["frames"])

    # Generate comprehensive summary using LLM
    transcription = state.get("transcription", "No audio transcription available.")

    # Get output directory from state or use current directory
    import os

    output_dir = os.path.dirname(state.get("output_path", ".")) or "."

    llm_summary = generate_video_summary(
        transcription=transcription,
        frame_data_summary=frame_data_summary,
        video_duration=video_duration,
        output_dir=output_dir,
    )

    state["summary"] = llm_summary
    return state


def save_video(state: PipelineState):
    from emoact import video_io

    frames = [frame_info["image"] for frame_info in state["frames"]]
    output_path = state["output_path"]
    fps = state["fps"]
    video_io.save_video(frames, output_path, fps)
    return state


def export_summary(state: PipelineState):
    """Export the LLM-generated summary to a text file."""
    import os

    output_path = state["output_path"]
    # Create summary file path (same name as video, but .txt extension)
    base_name = os.path.splitext(output_path)[0]
    summary_path = f"{base_name}_summary.txt"

    # Write summary to file
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("VIDEO ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Video: {state['video_path']}\n")
        f.write(f"Duration: {len(state['frames']) / state['fps']:.1f} seconds\n")
        f.write(f"Total Frames: {len(state['frames'])}\n")
        f.write(f"FPS: {state['fps']:.1f}\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(state["summary"])
        f.write("\n\n" + "=" * 80 + "\n")

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
graph_builder.add_node("classify_activities", classify_activities)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("save_video", save_video)
graph_builder.add_node("export_summary", export_summary)

# Arestas
graph_builder.add_edge(START, "load_video")
graph_builder.add_edge("load_video", "detect_faces")
graph_builder.add_conditional_edges(
    "detect_faces",
    has_persons,
    {"has_faces": "detect_poses", "no_faces": "detect_objects"},
)
graph_builder.add_edge("detect_poses", "detect_emotions")
graph_builder.add_edge("detect_emotions", "track_faces")
graph_builder.add_edge("track_faces", "classify_activities")

graph_builder.add_edge("classify_activities", "detect_objects")
graph_builder.add_edge("detect_objects", "transcribe_audio")
graph_builder.add_edge("transcribe_audio", "draw")
graph_builder.add_edge("draw", "summarize")
graph_builder.add_edge("summarize", "save_video")
graph_builder.add_edge("save_video", "export_summary")


print()
graph = graph_builder.compile()
draw_graph(graph)
if __name__ == "__main__":
    from tqdm import tqdm

    initial_state: PipelineState = {
        "video_path": "input/input_video.mp4",
        "output_path": "output.mp4",
        "fps": 0.0,
        "transcription": "",
        "frames": [],
        "summary": "",
        "object_conf_threshold": 0.5,
        "pose_conf_threshold": 0.3,
    }

    # Create progress bar
    with tqdm(
        total=len(graph.nodes) - 2,
        desc="Processing video",
        unit="step",
    ) as pbar:
        for event in graph.stream(initial_state):
            if event:
                node_name = list(event.keys())[0]
                state = event[node_name]

                # Update progress bar with current node
                pbar.update(1)

                # Show additional info for certain nodes
                if node_name == "load_video":
                    frames_count = len(state.get("frames", []))
                    pbar.write(
                        f"  → Loaded {frames_count} frames at {state['fps']:.1f} FPS"
                    )
                if node_name == "detect_poses":
                    total_poses = sum(
                        1
                        for f in state["frames"]
                        for p in f["persons"]
                        if p["pose"]["landmarks"]
                    )
                    pbar.write(f"  → Detected {total_poses} poses across all frames")
                if node_name == "detect_emotions":
                    total_emotions = sum(
                        len(p["emotions"])
                        for f in state["frames"]
                        for p in f["persons"]
                    )
                    pbar.write(
                        f"  → Detected {total_emotions} emotions across all faces"
                    )
                elif node_name == "detect_faces":
                    total_faces = sum(len(f["persons"]) for f in state["frames"])
                    pbar.write(f"  → Detected {total_faces} faces across all frames")
                elif node_name == "track_faces":
                    unique_persons = len(
                        set(
                            p["person_id"]
                            for f in state["frames"]
                            for p in f["persons"]
                            if p.get("person_id")
                        )
                    )
                    pbar.write(f"  → Tracked {unique_persons} unique person(s)")
                elif node_name == "transcribe_audio":
                    if state.get("transcription"):
                        words = len(state["transcription"].split())
                        pbar.write(f"  → Transcribed {words} words")
                elif node_name == "save_video":
                    pbar.write(f"  → Saved video to {state['output_path']}")
                elif node_name == "export_summary":
                    pbar.write(f"  → Summary exported successfully")

    print("\n✓ Pipeline completed successfully!")
