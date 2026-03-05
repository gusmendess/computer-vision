from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolo11n.pt")

    results = model.predict(
        source='test.mp4', # or 'test.jpg' or 0 for webcam
        conf=0.5,
        save=True,
        #show=True, # if not using WSL
        #stream=True # for webcam or video
    )

if __name__ == "__main__":
    main()