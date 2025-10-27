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


if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video file path
    video_path = os.path.abspath(video_path)
    transcription = transcribe_video(video_path, model="base")
    print("Transcription:")
    print(transcription)
