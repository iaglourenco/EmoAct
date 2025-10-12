from emoact.face_recog import detect_faces
import cv2


frame = cv2.imread("photo.jpg")  # Replace with your image path
if frame is None:
    raise ValueError("Image not found or unable to load.")

# Detect faces in the frame
faces = detect_faces([frame, frame])

# Draw bounding boxes around detected faces
for top, right, bottom, left, embedding, confidence in faces[0]:
    if confidence > 0.95:  # Only draw boxes for high-confidence detections
        cv2.rectangle(frame, (top, left), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Conf: {confidence:.2f}",
            (top - 10, left),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

# Display the resulting frame
cv2.imshow("Face Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
