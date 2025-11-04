import whisper
import torch
import os


def transcribe_video(video_path, model="base"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the Whisper model
    model = whisper.load_model(model, device=device, download_root="./models/whisper")

    # Transcribe the video file
    result = model.transcribe(video_path)

    text = result["text"]

    return text
