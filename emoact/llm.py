"""
LLM utilities for generating video summaries and analysis.
Uses a local LLM server running at port 1234 with OpenAI-like API.
"""

import requests
from pathlib import Path
from typing import Any, Sequence
from datetime import datetime


LLM_BASE_URL = "http://localhost:1234/v1"
MAX_TOKENS_PER_CHUNK = 2000  # Reduced limit to fit 8000 token context window
MAX_OUTPUT_TOKENS = 1000  # Maximum tokens for LLM response


def call_llm_chat(
    messages: list[dict[str, str]],
    model: str = "local-model",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> str:
    """
    Call the local LLM chat completion endpoint.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (defaults to local-model)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        The LLM's response text
    """
    url = f"{LLM_BASE_URL}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        error_msg = (
            f"HTTP Error calling LLM: {e.response.status_code}\n{e.response.text}"
        )
        print(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.Timeout:
        error_msg = "LLM request timed out after 300 seconds"
        print(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.ConnectionError:
        error_msg = "Could not connect to LLM server. Is it running on localhost:1234?"
        print(error_msg)
        return f"Error: {error_msg}"
    except requests.exceptions.RequestException as e:
        error_msg = f"Error calling LLM: {str(e)}"
        print(error_msg)
        return f"Error: {error_msg}"
    except (KeyError, IndexError) as e:
        error_msg = f"Unexpected response format from LLM: {str(e)}"
        print(error_msg)
        return f"Error: {error_msg}"


def prepare_frame_data_summary(frames: Sequence[Any]) -> str:
    """
    Create a condensed summary of frame data for LLM processing.
    Aggregates and samples frame data to fit within context limits.

    Args:
        frames: List of FrameInfo dicts

    Returns:
        Condensed text summary of frame data
    """
    total_frames = len(frames)

    # Sample frames more aggressively to avoid context overflow
    sample_rate = max(1, total_frames // 30)  # Keep only ~30 frames max
    sampled_frames = frames[::sample_rate]

    # Aggregate person statistics
    person_ids = set()
    emotion_counts = {}
    activity_data = []
    object_counts = {}
    object_confidences = []
    person_demographics = {}  # person_id -> {gender, ages, positions}
    people_per_frame = []

    for idx, frame in enumerate(sampled_frames):
        frame_num = idx * sample_rate
        frame_person_count = len(frame.get("persons", []))
        people_per_frame.append(frame_person_count)

        for person in frame.get("persons", []):
            person_id = person.get("person_id", "Unknown")
            person_ids.add(person_id)

            # Track demographics per person
            if person_id not in person_demographics:
                person_demographics[person_id] = {
                    "gender": person.get("gender"),
                    "ages": [],
                    "face_positions": [],  # Track face position across frames
                }

            # Collect age and position data
            age = person.get("age")
            if age:
                person_demographics[person_id]["ages"].append(age)

            face_loc = person.get("face_location")
            if face_loc and len(face_loc) >= 5:
                left, top, right, bottom, conf = face_loc
                # Calculate relative position (normalized to 0-1)
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                person_demographics[person_id]["face_positions"].append(
                    {"x": center_x, "y": center_y, "confidence": conf}
                )

            # Count emotions
            for emotion in person.get("emotions", []):
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Collect activity data
            activity = person.get("activity", {})
            if activity.get("pose_available") or activity.get(
                "image_classification_available"
            ):
                activity_entry = {
                    "frame": frame_num,
                    "person_id": person_id,
                    "gender": (
                        "Male"
                        if person.get("gender") == 1
                        else "Female" if person.get("gender") == 0 else "Unknown"
                    ),
                    "age": person.get("age"),
                }

                # Add top activity predictions
                if activity.get("image_predictions"):
                    top_predictions = activity["image_predictions"][:3]  # Top 3
                    activity_entry["activities"] = [
                        f"{pred['class_name']} ({pred['confidence']:.2f})"
                        for pred in top_predictions
                    ]

                # Add pose angles if available
                if activity.get("pose_angles"):
                    angles = activity["pose_angles"]
                    activity_entry["pose_angles"] = {
                        k: round(v, 1) if v is not None else None
                        for k, v in angles.items()
                        if v is not None
                    }

                # Add key pose landmarks (sample only important ones)
                if activity.get("pose_landmarks"):
                    key_landmarks = [
                        "nose",
                        "left_shoulder",
                        "right_shoulder",
                        "left_hip",
                        "right_hip",
                        "left_wrist",
                        "right_wrist",
                    ]
                    selected_landmarks = [
                        lm
                        for lm in activity["pose_landmarks"]
                        if lm.get("name") in key_landmarks
                        and lm.get("confidence", 0) > 0.5
                    ]
                    if selected_landmarks:
                        activity_entry["key_landmarks"] = [
                            f"{lm['name']}({lm['confidence']:.2f})"
                            for lm in selected_landmarks[:5]  # Max 5
                        ]

                activity_data.append(activity_entry)

        # Count objects with confidence tracking
        for obj in frame.get("objects", []):
            label = obj.get("label", "unknown")
            confidence = obj.get("confidence", 0)
            object_counts[label] = object_counts.get(label, 0) + 1
            object_confidences.append(confidence)

    # Build summary text
    summary_parts = [
        f"Total frames analyzed: {total_frames}",
        f"Frames sampled: {len(sampled_frames)}",
        f"Unique persons detected: {len(person_ids)}",
        f"Person IDs: {', '.join(sorted(person_ids)) if person_ids else 'None'}",
        "",
        "Scene Density:",
        (
            f"  - Average people per frame: {sum(people_per_frame) / len(people_per_frame):.1f}"
            if people_per_frame
            else "  - No data"
        ),
        (
            f"  - Max people in frame: {max(people_per_frame)}"
            if people_per_frame
            else "  - No data"
        ),
        "",
        "Demographics per Person:",
    ]

    # Add demographic info for each person
    for person_id in sorted(person_demographics.keys()):
        demo = person_demographics[person_id]
        gender_str = (
            "Male"
            if demo["gender"] == 1
            else "Female" if demo["gender"] == 0 else "Unknown"
        )
        avg_age = sum(demo["ages"]) / len(demo["ages"]) if demo["ages"] else "Unknown"
        avg_age_str = f"{avg_age:.0f}" if isinstance(avg_age, float) else avg_age

        # Determine typical position (center, left, right, top, bottom)
        if demo["face_positions"]:
            avg_x = sum(p["x"] for p in demo["face_positions"]) / len(
                demo["face_positions"]
            )
            avg_y = sum(p["y"] for p in demo["face_positions"]) / len(
                demo["face_positions"]
            )
            avg_conf = sum(p["confidence"] for p in demo["face_positions"]) / len(
                demo["face_positions"]
            )

            # Simple position description (assuming normalized coords or pixel coords)
            pos_desc = "center"
            if avg_x < 0.33:
                pos_desc = "left"
            elif avg_x > 0.67:
                pos_desc = "right"

            summary_parts.append(
                f"  - {person_id}: {gender_str}, Age ~{avg_age_str}, "
                f"Typically in {pos_desc} (face conf: {avg_conf:.2f})"
            )
        else:
            summary_parts.append(f"  - {person_id}: {gender_str}, Age ~{avg_age_str}")

    summary_parts.append("")
    summary_parts.append("Emotion Distribution:")

    for emotion, count in sorted(
        emotion_counts.items(), key=lambda x: x[1], reverse=True
    ):
        summary_parts.append(f"  - {emotion}: {count} occurrences")

    summary_parts.append("")
    summary_parts.append("Objects Detected:")

    # Add average confidence for context
    if object_confidences:
        avg_obj_conf = sum(object_confidences) / len(object_confidences)
        summary_parts.append(
            f"  Average object detection confidence: {avg_obj_conf:.2f}"
        )

    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        summary_parts.append(f"  - {obj}: {count} occurrences")

    summary_parts.append("")
    summary_parts.append("Activity Data (sampled frames):")

    # Group activity data by person
    activities_by_person = {}
    for act in activity_data:
        pid = act["person_id"]
        if pid not in activities_by_person:
            activities_by_person[pid] = []
        activities_by_person[pid].append(act)

    for person_id, activities in activities_by_person.items():
        summary_parts.append(f"\nPerson {person_id}:")
        if activities:
            first_activity = activities[0]
            summary_parts.append(f"  Gender: {first_activity.get('gender', 'Unknown')}")
            summary_parts.append(f"  Age: {first_activity.get('age', 'Unknown')}")

        # Sample activities for this person (max 5 to avoid context overflow)
        sampled_activities = activities[
            :: max(1, len(activities) // 5)
        ]  # Max 5 samples per person
        for act in sampled_activities:
            frame_info = f"  Frame {act['frame']}:"
            details = []

            if "activities" in act:
                # Only show top 2 activities
                top_activities = act["activities"][:2]
                details.append(f"Activities: {', '.join(top_activities)}")

            # Remove pose angles and landmarks to save space
            # Keep only most relevant info

            if details:
                summary_parts.append(f"{frame_info} {' | '.join(details)}")

    return "\n".join(summary_parts)


def chunk_text(text: str, max_chars: int = MAX_TOKENS_PER_CHUNK * 4) -> list[str]:
    """
    Split text into chunks that fit within context limits.
    Rough approximation: 1 token ≈ 4 characters.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for newline

        if current_length + line_length > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def save_raw_data_to_file(
    transcription: str,
    frame_data_summary: str,
    video_duration: float,
    output_dir: str = ".",
) -> str:
    """
    Save raw data to a file when LLM processing fails.

    Args:
        transcription: Audio transcription text
        frame_data_summary: Condensed frame analysis data
        video_duration: Duration in seconds
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"raw_data_{timestamp}.txt"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Video Duration: {video_duration:.1f} seconds\n")
        f.write("=" * 80 + "\n\n")
        f.write("TRANSCRIPTION:\n")
        f.write(transcription)
        f.write("\n\n" + "=" * 80 + "\n\n")
        f.write("FRAME DATA SUMMARY:\n")
        f.write(frame_data_summary)

    return str(filepath)


def generate_video_summary(
    transcription: str,
    frame_data_summary: str,
    video_duration: float,
    output_dir: str = ".",
) -> str:
    """
    Generate a comprehensive video summary using the LLM.
    Handles long content by chunking and using divide-and-conquer approach.
    Saves raw data to file if LLM processing fails.

    Args:
        transcription: Audio transcription text
        frame_data_summary: Condensed frame analysis data
        video_duration: Duration in seconds
        output_dir: Directory to save raw data if processing fails

    Returns:
        Comprehensive video summary from the LLM
    """
    system_prompt = """Você é um analista especializado em vídeos, com expertise em comportamento humano, emoções e reconhecimento de atividades.
Sua tarefa é analisar dados de vídeo incluindo transcrição de áudio, emoções detectadas, atividades identificadas e objetos de cena para gerar um resumo abrangente e coerente.

INSTRUÇÕES CRÍTICAS:
- Escreva TODO o resumo em PORTUGUÊS BRASILEIRO
- Use APENAS texto simples - NÃO use tabelas Markdown, formatação especial ou listas complexas
- NUNCA invente ou alucinacione informações que não estão nos dados fornecidos
- Se não tiver certeza sobre algo ou se os dados forem insuficientes, DIGA CLARAMENTE "Não foi possível determinar..." ou "Os dados não indicam..."
- Escreva SOMENTE sobre o que realmente ocorreu no vídeo baseado nos dados fornecidos
- Se alguma informação estiver ausente ou não clara, mencione explicitamente essa limitação

Foque em:
1. Narrativa geral e eventos-chave no vídeo (baseado SOMENTE nos dados)
2. Estados emocionais e mudanças das pessoas ao longo do vídeo
3. Atividades e comportamentos observados
4. Interações entre pessoas e objetos
5. Padrões ou insights notáveis

Forneça um resumo bem estruturado em texto simples que conte a história do que aconteceu no vídeo, sempre baseado estritamente nos dados fornecidos."""

    # Prepare the analysis data
    duration_str = f"{video_duration:.1f} seconds ({video_duration/60:.1f} minutes)"

    # Truncate data more aggressively to fit in context
    max_transcription_chars = 3000  # ~750 tokens
    max_frame_data_chars = 4000  # ~1000 tokens

    transcription_truncated = transcription[:max_transcription_chars]
    if len(transcription) > max_transcription_chars:
        transcription_truncated += "\n... (transcription truncated)"

    frame_data_truncated = frame_data_summary[:max_frame_data_chars]
    if len(frame_data_summary) > max_frame_data_chars:
        frame_data_truncated += "\n... (frame data truncated)"

    # Check if we need to chunk the content
    transcription_chunks = chunk_text(transcription_truncated)
    frame_chunks = chunk_text(frame_data_truncated)

    # Try to process with LLM
    try:
        # If everything fits in one request
        if len(transcription_chunks) == 1 and len(frame_chunks) == 1:
            user_prompt = f"""Analise estes dados de vídeo e gere um resumo abrangente em PORTUGUÊS BRASILEIRO:

DURAÇÃO DO VÍDEO: {duration_str}

TRANSCRIÇÃO DE ÁUDIO:
{transcription_truncated}

ANÁLISE VISUAL:
{frame_data_truncated}

Forneça um resumo detalhado em texto simples cobrindo os eventos-chave, emoções, atividades e narrativa geral do vídeo. Escreva SOMENTE sobre o que está presente nos dados acima. Se algo não estiver claro ou disponível, mencione explicitamente."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            result = call_llm_chat(messages, max_tokens=MAX_OUTPUT_TOKENS)

            # Check if result is an error
            if result.startswith("Error:"):
                print("\nLLM processing failed. Saving raw data...")
                raw_file = save_raw_data_to_file(
                    transcription, frame_data_summary, video_duration, output_dir
                )
                return f"Failed to generate summary due to LLM error. Raw data saved to: {raw_file}\n\n{result}"

            return result

        # Divide and conquer approach for long videos
        else:
            # First, summarize transcription chunks
            transcription_summaries = []
            for i, chunk in enumerate(transcription_chunks):
                chunk_prompt = f"""Resuma os pontos-chave desta porção da transcrição do vídeo em PORTUGUÊS BRASILEIRO (Parte {i+1}/{len(transcription_chunks)}):

{chunk}

Foque nos principais tópicos, eventos e diálogos. Escreva SOMENTE o que está presente no texto acima. Não invente informações."""

                messages = [
                    {
                        "role": "system",
                        "content": "Você é um especialista em resumos. Crie resumos concisos de trechos de transcrição em português brasileiro. Nunca invente informações que não estão no texto.",
                    },
                    {"role": "user", "content": chunk_prompt},
                ]

                summary = call_llm_chat(messages, max_tokens=300)
                if summary.startswith("Error:"):
                    print(f"\nError processing chunk {i+1}. Saving raw data...")
                    raw_file = save_raw_data_to_file(
                        transcription, frame_data_summary, video_duration, output_dir
                    )
                    return f"Failed to generate summary. Raw data saved to: {raw_file}\n\n{summary}"

                transcription_summaries.append(summary)

            # Combine transcription summaries
            combined_transcription = "\n\n".join(
                [
                    f"Parte {i+1}: {summary}"
                    for i, summary in enumerate(transcription_summaries)
                ]
            )

            # Final comprehensive summary
            final_prompt = f"""Analise estes dados de vídeo e gere um resumo abrangente em PORTUGUÊS BRASILEIRO:

DURAÇÃO DO VÍDEO: {duration_str}

RESUMO DA TRANSCRIÇÃO DE ÁUDIO:
{combined_transcription}

ANÁLISE VISUAL:
{frame_data_truncated}

Forneça um resumo detalhado e coerente em texto simples que conte a história completa do vídeo, incluindo eventos-chave, emoções, atividades e insights. Escreva SOMENTE sobre o que está presente nos dados acima. Se algo não estiver claro, limitado ou indisponível, mencione explicitamente essa limitação."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ]

            result = call_llm_chat(messages, max_tokens=MAX_OUTPUT_TOKENS)

            if result.startswith("Error:"):
                print("\nFinal LLM call failed. Saving raw data...")
                raw_file = save_raw_data_to_file(
                    transcription, frame_data_summary, video_duration, output_dir
                )
                return f"Failed to generate summary. Raw data saved to: {raw_file}\n\n{result}"

            return result

    except Exception as e:
        print(f"\nUnexpected error during LLM processing: {str(e)}")
        raw_file = save_raw_data_to_file(
            transcription, frame_data_summary, video_duration, output_dir
        )
        return f"Unexpected error during summary generation. Raw data saved to: {raw_file}\n\nError: {str(e)}"


def get_available_models() -> list[str]:
    """
    Get list of available models from the local LLM server.

    Returns:
        List of model names
    """
    url = f"{LLM_BASE_URL}/models"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        return [model["id"] for model in data.get("data", [])]

    except requests.exceptions.RequestException as e:
        print(f"Error getting models: {str(e)}")
        return []
