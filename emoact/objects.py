from ultralytics.models import YOLO

model = YOLO("models/yolo11n.pt")


def detect_objects_in_frame(frame):
    """
    Detect objects in a frame.

    Returns:
        List of tuples: (bbox, class_id, confidence)
        where bbox is (left, top, right, bottom)
    """
    results = model(frame)
    objects = []

    for result in results:
        # Acessa o tensor de dados diretamente: [x1, y1, x2, y2, conf, cls]
        # YOLO retorna xyxy que é [left, top, right, bottom]
        for data in result.boxes.data:
            # Desempacota os dados do tensor
            left, top, right, bottom, conf, cls = data.cpu().numpy()

            # Monta a tupla de retorno: (left, top, right, bottom)
            bbox = (int(left), int(top), int(right), int(bottom))
            confidence = float(conf)
            class_id = int(cls)
            # Remove 'person' class (class_id 0)
            if class_id != 0:
                objects.append((bbox, class_id, confidence))

    return objects
