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
        error_msg = "LLM request timed out after 300 seconds."
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


def _select_event_frames(frames: Sequence[Any], max_frames: int = 8) -> list[int]:
    """
    Select frames based on events (changes in people count, activity).
    Returns indices of selected frames. Aggressive sampling for token efficiency.
    """
    if len(frames) <= max_frames:
        return list(range(len(frames)))

    selected = {0, len(frames) - 1}  # First and last only

    prev_people_count = 0

    for idx, frame in enumerate(frames):
        if len(selected) >= max_frames:
            break

        people_count = len(frame.get("persons", []))

        # Select only significant people count changes
        if abs(people_count - prev_people_count) >= 2:
            selected.add(idx)
            prev_people_count = people_count

    # Uniform sampling if needed
    if len(selected) < max_frames and len(frames) > max_frames:
        step = len(frames) // max_frames
        for i in range(0, len(frames), step):
            if len(selected) >= max_frames:
                break
            selected.add(i)

    return sorted(selected)


def prepare_frame_data_summary(frames: Sequence[Any]) -> str:
    """
    Create ultra-compact structured data for LLM processing.
    Output is dense, predictable, token-efficient.
    """
    total_frames = len(frames)

    # Aggressive event-based sampling
    selected_indices = _select_event_frames(frames, max_frames=16)
    sampled_frames = [frames[i] for i in selected_indices]

    # Aggregate only
    person_ids = set()
    emotion_counts = {}
    activity_counts = {}
    object_counts = {}
    people_per_frame = []
    person_confidence = {}
    pose_angles_data = {}
    pose_availability = {}
    image_classification_availability = {}

    for frame in sampled_frames:
        people_per_frame.append(len(frame.get("persons", [])))

        for person in frame.get("persons", []):
            pid = person.get("person_id", "Unknown")
            person_ids.add(pid)

            # Emotions
            for emotion in person.get("emotions", []):
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Activities
            activity = person.get("activity", {})

            # Track pose and image classification availability
            if pid not in pose_availability:
                pose_availability[pid] = []
                image_classification_availability[pid] = []

            pose_availability[pid].append(activity.get("pose_available", False))
            image_classification_availability[pid].append(
                activity.get("image_classification_available", False)
            )

            # Collect pose angles
            if activity.get("pose_angles"):
                angles = activity["pose_angles"]
                if pid not in pose_angles_data:
                    pose_angles_data[pid] = {}

                for joint, angle in angles.items():
                    if angle is not None:
                        if joint not in pose_angles_data[pid]:
                            pose_angles_data[pid][joint] = []
                        pose_angles_data[pid][joint].append(angle)

            if activity.get("image_predictions"):
                preds = activity["image_predictions"][:2]

                if pid not in activity_counts:
                    activity_counts[pid] = {}

                for pred in preds:
                    act = pred["class_name"]
                    activity_counts[pid][act] = activity_counts[pid].get(act, 0) + 1

                    if pid not in person_confidence:
                        person_confidence[pid] = []
                    person_confidence[pid].append(pred["confidence"])

        # Objects
        for obj in frame.get("objects", []):
            label = obj.get("label", "unknown")
            object_counts[label] = object_counts.get(label, 0) + 1

    # Anomaly metrics
    anomalies = []
    for pid in person_ids:
        confs = person_confidence.get(pid, [])
        if len(confs) >= 2:
            avg = sum(confs) / len(confs)
            if avg < 0.45:
                anomalies.append(f"{pid}:{avg:.2f}")

    # Ultra-compact format
    avg_ppl = sum(people_per_frame) / len(people_per_frame) if people_per_frame else 0

    out = [
        f"frm:{total_frames} smp:{len(sampled_frames)} ppl:{len(person_ids)} avg:{avg_ppl:.1f}"
    ]

    # Top emotions only
    top_emo = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_emo:
        out.append("emo:" + ",".join([f"{e}({c})" for e, c in top_emo]))

    # Top objects only
    top_obj = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    if top_obj:
        out.append("obj:" + ",".join([f"{o}({c})" for o, c in top_obj]))

    # Activities only for persons with data
    if activity_counts:
        act_lines = []
        for pid in sorted(activity_counts.keys())[:3]:  # Max 3 persons
            acts = activity_counts[pid]
            top_acts = sorted(acts.items(), key=lambda x: x[1], reverse=True)[:2]
            act_lines.append(f"{pid}:" + ",".join([f"{a}({c})" for a, c in top_acts]))
        if act_lines:
            out.append("act:" + " ".join(act_lines))

    # Pose angles summary
    if pose_angles_data:
        angle_lines = []
        for pid in sorted(pose_angles_data.keys())[:3]:  # Max 3 persons
            joints = pose_angles_data[pid]
            joint_summaries = []
            for joint, angles in sorted(joints.items())[:4]:  # Max 4 joints per person
                if angles:
                    avg = sum(angles) / len(angles)
                    joint_summaries.append(f"{joint}:{avg:.0f}")
            if joint_summaries:
                angle_lines.append(f"{pid}:" + ",".join(joint_summaries))
        if angle_lines:
            out.append("ang:" + " ".join(angle_lines))

    # Data availability summary
    if pose_availability:
        pose_avail = sum(sum(avail) for avail in pose_availability.values()) / sum(
            len(avail) for avail in pose_availability.values()
        )
        img_avail = sum(
            sum(avail) for avail in image_classification_availability.values()
        ) / sum(len(avail) for avail in image_classification_availability.values())
        out.append(f"avail:pose={pose_avail:.2f},img={img_avail:.2f}")

    # Anomalies
    if anomalies:
        out.append(f"anom:{len(anomalies)} " + ",".join(anomalies[:3]))
    else:
        out.append("anom:0")

    return " | ".join(out)


def save_raw_data_to_file(
    transcription: str,
    frame_data_summary: str,
    video_duration: float,
    output_dir: str = ".",
) -> str:
    """
    Save raw data to a file when LLM processing fails.

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


def save_prompt_to_file(
    system_prompt: str,
    user_prompt: str,
    video_duration: float,
    output_dir: str = ".",
) -> str:
    """
    Save the complete prompt sent to LLM for debugging and analysis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_prompt_{timestamp}.txt"
    filepath = Path(output_dir) / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"LLM PROMPT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video Duration: {video_duration:.1f} seconds\n")
        f.write("=" * 80 + "\n\n")
        f.write("SYSTEM PROMPT:\n")
        f.write(system_prompt)
        f.write("\n\n" + "=" * 80 + "\n\n")
        f.write("USER PROMPT:\n")
        f.write(user_prompt)
        f.write("\n\n" + "=" * 80 + "\n")

    return str(filepath)


def generate_video_summary(
    transcription: str,
    frame_data_summary: str,
    video_duration: float,
    output_dir: str = ".",
) -> str:
    """
    Generate a comprehensive video summary using the LLM.

    """
    system_prompt = """ROLE: video_analyst
LANG: pt-BR

OBJETIVO:
Gerar um resumo técnico, factual e coeso a partir de dados de áudio e visão computacional estruturados.
Não inventar informações, não inferir atributos não observáveis, não explicar causas.

FONTE DOS DADOS:
- ÁUDIO: transcrição textual do vídeo; pode conter narração descritiva ou interpretativa.
- VISUAL: resumo estrutural compacto gerado por visão computacional; é parcial e baseado em amostragem.

PRIORIDADE DE CONFIANÇA:
- Dados VISUAIS estruturados têm prioridade sobre interpretações narrativas do áudio.
- O áudio deve ser tratado como contextual, não como evidência visual direta.

SINTAXE E SIGNIFICADO DOS CAMPOS VISUAIS:
- frm: total de frames do vídeo. Usar apenas como escala temporal global.
- smp: número de frames efetivamente analisados. As observações visuais são parciais.
- ppl: número total de indivíduos distintos detectados ao longo dos frames amostrados.
  Não representa pessoas simultâneas.
- avg: média de pessoas por frame amostrado. Indica densidade visual do ambiente.
- emo: contagens relativas de emoções detectadas nos frames amostrados.
  Não são porcentagens, não são estados emocionais absolutos.
- obj: contagens relativas de objetos detectados. Usar apenas para contextualizar o ambiente físico.
- act: rótulos de classificação visual associados a indivíduos específicos.
  São hipóteses de classificação de imagem e podem estar incorretas.
  Não representam ações reais confirmadas.
- ang: ângulos médios das articulações corporais (em graus) por indivíduo.
  Refletem postura corporal observada nos frames amostrados.
  Úteis para inferir posições corporais (sentado, em pé, braços levantados, etc).
- avail: proporção de frames onde dados de pose e classificação de imagem estavam disponíveis.
  Valores próximos de 1.0 indicam cobertura completa, valores baixos indicam dados parciais.
- anom: número de indivíduos com baixa confiança média de classificação.
  Os valores indicam frequência e magnitude, não a causa do problema.

REGRAS DE INTERPRETAÇÃO:
- Não converter contagens em porcentagens.
- Não assumir simultaneidade entre indivíduos.
- Não assumir causalidade entre emoção, atividade e áudio.
- Não inferir idade, etnia, gênero, intenção ou contexto social não explicitamente fornecido.
- Emoções e atividades devem ser descritas como tendências observadas, não como estados definitivos.

ANOMALIAS:
- Se anom:0, declarar explicitamente ausência de anomalias.
- Se houver anomalias, relatar apenas a quantidade e recorrência.
- Nunca explicar causas ou consequências das anomalias.

ESTILO DO TEXTO:
- Texto corrido, técnico e objetivo.
- Parágrafos curtos.
- Linguagem descritiva, não promocional.
- Sem metáforas, sem adjetivação excessiva.

FORMATO FINAL:
- Um único resumo integrado (não separar áudio e visual em listas).
- Máximo de 300 palavras.
- Incluir uma frase curta indicando que as observações visuais se baseiam em amostragem limitada de frames representativos.
- Focar em padrões, recorrências e coerência entre áudio e visual.
- Descrever o que o vídeo aparenta retratar com base exclusiva nos dados fornecidos, sem extrapolações.
- Faça o resumo como se fosse para um relatório técnico.
"""

    # Prepare the analysis data
    duration_str = f"{video_duration:.1f} seconds ({video_duration/60:.1f} minutes)"

    # Try to process with LLM
    try:
        user_prompt = f"""Vídeo {duration_str}

ÁUDIO: {transcription}

VISUAL: {frame_data_summary}

Resumo 150 pal."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Save prompt to file
        try:
            prompt_file = save_prompt_to_file(
                system_prompt, user_prompt, video_duration, output_dir
            )
            print(f"\nPrompt saved to: {prompt_file}")
        except Exception as e:
            print(f"\nWarning: Could not save prompt to file: {str(e)}")

        result = call_llm_chat(messages, max_tokens=300)

        # Check if result is an error
        if result.startswith("Error:"):
            print("\nLLM processing failed. Saving raw data...")
            raw_file = save_raw_data_to_file(
                transcription, frame_data_summary, video_duration, output_dir
            )
            return f"Failed to generate summary due to LLM error. Raw data saved to: {raw_file}\n\n{result}"

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
