import mediapipe as mp
from mediapipe.tasks import python
import cv2

model_path = "models\\pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
)


def draw_pose_landmarks(frame, pose_landmarks):
    """Desenha os pose landmarks no frame"""
    if pose_landmarks:
        height, width, _ = frame.shape

        # Conecta os pontos dos landmarks formando o esqueleto
        connections = [
            # Rosto
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 7),
            (0, 4),
            (4, 5),
            (5, 6),
            (6, 8),
            # Tronco
            (9, 10),
            (11, 12),
            (11, 13),
            (13, 15),
            (15, 17),
            (15, 19),
            (15, 21),
            (17, 19),
            (12, 14),
            (14, 16),
            (16, 18),
            (16, 20),
            (16, 22),
            (18, 20),
            (11, 23),
            (12, 24),
            (23, 24),
            # Pernas
            (23, 25),
            (25, 27),
            (27, 29),
            (27, 31),
            (29, 31),
            (24, 26),
            (26, 28),
            (28, 30),
            (28, 32),
            (30, 32),
        ]

        # Desenha as conexões
        for connection in connections:
            if connection[0] < len(pose_landmarks) and connection[1] < len(
                pose_landmarks
            ):
                start_point = pose_landmarks[connection[0]]
                end_point = pose_landmarks[connection[1]]

                start_x = int(start_point.x * width)
                start_y = int(start_point.y * height)
                end_x = int(end_point.x * width)
                end_y = int(end_point.y * height)

                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        # Desenha os pontos dos landmarks
        for landmark in pose_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    return frame


# load video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(0)

# Verifica se o vídeo foi carregado corretamente
if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

print("Vídeo carregado com sucesso!")
print("Pressione 'q' ou 'ESC' para sair")

# Cria a janela para exibição
cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)

frame_count = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()

        if not ret:
            # Se chegou ao fim do vídeo, reinicia
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue

        frame_count += 1

        # Converte o frame para RGB (MediaPipe usa RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Usa frame_count como timestamp se não conseguir obter do vídeo
        try:
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            if frame_timestamp_ms <= 0:
                frame_timestamp_ms = frame_count * 33  # Assumindo ~30 FPS
        except:
            frame_timestamp_ms = frame_count * 33

        # Cria a imagem MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detecta os pose landmarks
        try:
            pose_landmarker_result = landmarker.detect_for_video(
                mp_image, frame_timestamp_ms
            )

            # Desenha os landmarks no frame
            if pose_landmarker_result.pose_landmarks:
                for pose_landmarks in pose_landmarker_result.pose_landmarks:
                    frame = draw_pose_landmarks(frame, pose_landmarks)
        except Exception as e:
            print(f"Erro na detecção do frame {frame_count}: {e}")
            # Continue mesmo com erro na detecção

        # Exibe o frame (com ou sem landmarks)
        cv2.imshow("Pose Detection", frame)

        # Pressione 'q' para sair ou 'ESC'
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == 27:
            break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
print("Aplicação encerrada!")
