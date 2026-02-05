# Computer Vision

Repositório para treinamento e inferência de modelos de visão computacional.

## Instalação das Dependências

1. Instale as dependências usando pip:

```bash
pip install -r requirements.txt
```

## Estrutura do Dataset

O dataset deve estar organizado da seguinte forma:

```
dataset/
├── train/
│   ├── classe1/
│   └── classe2/
├── valid/
│   ├── classe1/
│   └── classe2/
└── test/
    ├── classe1/
    └── classe2/
```

## Treinamento

Para treinar o modelo:

```bash
python train.py
```

O script irá:
- Treinar um modelo ResNet50
- Gerar gráficos de métricas (`validation_metrics.png` e `test_metrics.png`)
- Mostrar as métricas finais no console

## Inferência

Para executar inferência em uma imagem:

1. Coloque a imagem que deseja processar na raiz do projeto com o nome `foto.jpeg`
2. Execute:

```bash
python inference.py
```

Os resultados serão salvos automaticamente.

Para usar a webcam (comentado no código), descomente a linha 10 do arquivo `inference.py`.
