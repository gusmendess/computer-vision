from ultralytics import YOLO
from pathlib import Path


BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
EPOCHS = 10
IMAGE_SIZE = 320
BATCH_SIZE = 8
MODEL_NAME = "yolo11n.pt"
DATASET_PATH = str(ROOT_DIR / "datasets/person_dataset/data.yaml")


def load_model(model_name: str) -> YOLO:
    return YOLO(model_name)


def train(model: YOLO, dataset_path: str, epochs: int, image_size: int, batch_size: int):
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        pretrained=True,
        project=str(BASE_DIR / "runs/detect"),
        name="person",
        verbose=True,
    )
    return results


def validate(model: YOLO):
    metrics = model.val()
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    return metrics


def predict(model: YOLO, source: str):
    results = model.predict(source=source, save=True)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            coordinates = box.xyxy[0].tolist()
            print(f"Class: {result.names[class_id]} | Confidence: {confidence:.4f} | Box: {coordinates}")
    return results


def main():
    model = load_model(MODEL_NAME)
    train(model, DATASET_PATH, EPOCHS, IMAGE_SIZE, BATCH_SIZE)
    validate(model)


if __name__ == "__main__":
    main()
