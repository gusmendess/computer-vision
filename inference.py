from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    
    # Para processar uma imagem
    results = model.predict(source='foto.jpeg', conf=0.7, save=True)
    
    # Para usar webcam (sem mostrar janela, apenas salva frames)
    # model.predict(source=0, conf=0.25, save=True, stream=True)

if __name__ == "__main__":
    main()
