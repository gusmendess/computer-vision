import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

# Hyperparâmetros
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-5
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformações
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset e Dataloader
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_dataset = datasets.ImageFolder("dataset/valid", transform=transform)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

test_dataset = datasets.ImageFolder("dataset/test", transform=transform)
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

print(f"Classes detectadas: {train_dataset.classes}")
print(f"Número de imagens de treino: {len(train_dataset)}")
print(f"Número de imagens de validação: {len(val_dataset)}")
print(f"Número de imagens de teste: {len(test_dataset)}")

# Modelo
model = models.resnet50(weights="IMAGENET1K_V2")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss e Otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def calculate_metrics(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predictions = torch.max(outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall

# Histórico de métricas
history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_precision': [],
    'val_recall': []
}

# Treinamento
for epoch in range(EPOCHS):
    # Treinamento
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    
    # Validação
    val_loss, val_acc, val_prec, val_rec = calculate_metrics(model, val_loader, DEVICE)
    
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    history['val_precision'].append(val_prec)
    history['val_recall'].append(val_rec)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val Prec: {val_prec:.4f} - Val Rec: {val_rec:.4f}")

# Plotar métricas de validação ao longo dos epochs
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

epochs = range(1, len(history['train_loss']) + 1)

axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
axes[0, 0].set_title('Loss ao longo dos Epochs')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(epochs, history['val_accuracy'], 'g-', label='Accuracy')
axes[0, 1].set_title('Validação - Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(epochs, history['val_precision'], 'm-', label='Precision')
axes[1, 0].set_title('Validação - Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(epochs, history['val_recall'], 'c-', label='Recall')
axes[1, 1].set_title('Validação - Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('validation_metrics.png')
plt.show()

print("\n" + "="*50)
print("Avaliação no Dataset de Teste")
print("="*50)


# Avaliação final no dataset de teste
test_loss, test_acc, test_prec, test_rec = calculate_metrics(model, test_loader, DEVICE)

print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy : {test_acc*100:.2f}%")
print(f"Precision: {test_prec*100:.2f}%")
print(f"Recall   : {test_rec*100:.2f}%")

# Calcular confusion matrix para o teste
model.eval()
all_test_predictions = []
all_test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        all_test_predictions.extend(predictions.cpu().numpy())
        all_test_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_test_labels, all_test_predictions)

# Plotar métricas do teste
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras com métricas
metrics = ['Accuracy', 'Precision', 'Recall']
values = [test_acc, test_prec, test_rec]
colors = ['#2ecc71', '#3498db', '#9b59b6']

bars = axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
axes[0].set_ylim([0, 1])
axes[0].set_ylabel('Score')
axes[0].set_title('Métricas no Dataset de Teste')
axes[0].grid(True, axis='y', alpha=0.3)

for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{value*100:.2f}%',
                ha='center', va='bottom', fontweight='bold')

# Confusion Matrix
im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[1].figure.colorbar(im, ax=axes[1])
axes[1].set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=train_dataset.classes,
           yticklabels=train_dataset.classes,
           title='Confusion Matrix - Teste',
           ylabel='True Label',
           xlabel='Predicted Label')

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[1].text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('test_metrics.png')
plt.show()