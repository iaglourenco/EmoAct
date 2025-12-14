"""
LLM utilities for generating video summaries and analysis.
Uses a local LLM server running at port 1234 with OpenAI-like API.
"""

import requests
from typing import Any, Sequence


LLM_BASE_URL = "http://localhost:1234/v1"
MAX_TOKENS_PER_CHUNK = 8000  # Conservative limit for context window


def call_llm_chat(
    messages: list[dict[str, str]],
    model: str = "local-model",
    temperature: float = 0.7,
    max_tokens: int = 2000,
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

    except requests.exceptions.RequestException as e:
        return f"Error calling LLM: {str(e)}"


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

    # Sample frames if too many (every Nth frame)
    sample_rate = max(1, total_frames // 100)  # Keep ~100 frames max
    sampled_frames = frames[::sample_rate]

    # Aggregate person statistics
    person_ids = set()
    emotion_counts = {}
    activity_data = []
    object_counts = {}

    for idx, frame in enumerate(sampled_frames):
        frame_num = idx * sample_rate

        for person in frame.get("persons", []):
            person_id = person.get("person_id", "Unknown")
            person_ids.add(person_id)

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

                activity_data.append(activity_entry)

        # Count objects
        for obj in frame.get("objects", []):
            label = obj.get("label", "unknown")
            object_counts[label] = object_counts.get(label, 0) + 1

    # Build summary text
    summary_parts = [
        f"Total frames analyzed: {total_frames}",
        f"Frames sampled: {len(sampled_frames)}",
        f"Unique persons detected: {len(person_ids)}",
        f"Person IDs: {', '.join(sorted(person_ids)) if person_ids else 'None'}",
        "",
        "Emotion Distribution:",
    ]

    for emotion, count in sorted(
        emotion_counts.items(), key=lambda x: x[1], reverse=True
    ):
        summary_parts.append(f"  - {emotion}: {count} occurrences")

    summary_parts.append("")
    summary_parts.append("Objects Detected:")
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

        # Sample activities for this person
        sampled_activities = activities[
            :: max(1, len(activities) // 10)
        ]  # Max 10 samples per person
        for act in sampled_activities:
            frame_info = f"  Frame {act['frame']}:"
            if "activities" in act:
                summary_parts.append(f"{frame_info} {', '.join(act['activities'])}")

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


def generate_video_summary(
    transcription: str,
    frame_data_summary: str,
    video_duration: float,
) -> str:
    """
    Generate a comprehensive video summary using the LLM.
    Handles long content by chunking and using divide-and-conquer approach.

    Args:
        transcription: Audio transcription text
        frame_data_summary: Condensed frame analysis data
        video_duration: Duration in seconds

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

    # Check if we need to chunk the content
    transcription_chunks = chunk_text(transcription)
    frame_chunks = chunk_text(frame_data_summary)

    # If everything fits in one request
    if len(transcription_chunks) == 1 and len(frame_chunks) == 1:
        user_prompt = f"""Analise estes dados de vídeo e gere um resumo abrangente em PORTUGUÊS BRASILEIRO:

DURAÇÃO DO VÍDEO: {duration_str}

TRANSCRIÇÃO DE ÁUDIO:
{transcription}

ANÁLISE VISUAL:
{frame_data_summary}

Forneça um resumo detalhado em texto simples cobrindo os eventos-chave, emoções, atividades e narrativa geral do vídeo. Escreva SOMENTE sobre o que está presente nos dados acima. Se algo não estiver claro ou disponível, mencione explicitamente."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return call_llm_chat(messages, max_tokens=2000)

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

            summary = call_llm_chat(messages, max_tokens=500)
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
{frame_data_summary}

Forneça um resumo detalhado e coerente em texto simples que conte a história completa do vídeo, incluindo eventos-chave, emoções, atividades e insights. Escreva SOMENTE sobre o que está presente nos dados acima. Se algo não estiver claro, limitado ou indisponível, mencione explicitamente essa limitação."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ]

        return call_llm_chat(messages, max_tokens=2000)


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
