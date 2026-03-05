# Computer Vision

Repositório com dois projetos de visão computacional: **Classificação de Imagens** e **Detecção de Objetos**.

## Estrutura do Projeto

```
computer-vision/
├── datasets/
│   ├── cat_dog_dataset/          # Dataset de classificação (cat/dog)
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── person_dataset/           # Dataset de detecção (Persons)
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
│
├── classification/
│   └── train.py                  # Treino com ResNet50 (PyTorch)
│
├── detection/
│   ├── train.py                  # Treino com YOLOv11
│   └── inference.py              # Inferência em imagem/vídeo
│
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Classificação de Imagens

Treina um modelo **ResNet50** para classificar imagens de gato e cachorro usando PyTorch.

### Executar treino

```bash
python classification/train.py
```

O script executa as seguintes etapas:
- Carrega o dataset `cat_dog_dataset` (train/valid/test)
- Treina um ResNet50 pré-treinado com fine-tuning
- Exibe métricas por epoch (loss, accuracy, precision, recall)
- Gera gráficos de validação (`classification/validation_metrics.png`)
- Avalia no dataset de teste com confusion matrix (`classification/test_metrics.png`)

### Hiperparâmetros

| Parâmetro     | Valor  |
|---------------|--------|
| Batch Size    | 32     |
| Epochs        | 10     |
| Learning Rate | 1e-5   |
| Image Size    | 224x224|
| Modelo        | ResNet50 |

## Detecção de Objetos

Treina um modelo **YOLOv11** para detecção de pessoas usando Ultralytics.

### Executar treino

```bash
python detection/train.py
```

O script executa as seguintes etapas:
- Carrega o modelo YOLOv11 pré-treinado (`yolo11n.pt`)
- Treina com o dataset `person_dataset`
- Exibe métricas de detecção (mAP50, mAP50-95, Precision, Recall)
- Salva resultados em `detection/runs/detect/person/`

### Hiperparâmetros

| Parâmetro   | Valor     |
|-------------|-----------|
| Batch Size  | 8         |
| Epochs      | 10        |
| Image Size  | 320       |
| Modelo      | yolo11n.pt|

### Inferência

```bash
python detection/inference.py
```

Executa inferência em uma imagem ou vídeo. Para alterar a fonte, edite o parâmetro `source` no arquivo:

- Imagem: `source='test.jpeg'`
- Vídeo: `source='test.mp4'`
- Webcam: `source=0`. OBS: É o mais legal, use tiver webcam!
