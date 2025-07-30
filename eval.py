import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ======= CONFIGURAÇÕES =======
IMG_HEIGHT = 75
IMG_WIDTH = 100
TRAIN_PATH = "data/train"
EVAL_PATH = "data/eval"
EPOCHS = 10
BATCH_SIZE = 32

# ======= FUNÇÃO PARA CARREGAR IMAGENS =======
def carregar_imagens(pasta_base):
    X, y = [], []
    label_names = sorted(os.listdir(pasta_base))
    label_map = {label: idx for idx, label in enumerate(label_names)}

    for label in label_names:
        pasta_classe = os.path.join(pasta_base, label)
        for nome_arquivo in os.listdir(pasta_classe):
            caminho_img = os.path.join(pasta_classe, nome_arquivo)
            img = load_img(caminho_img, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img) / 255.0
            X.append(img)
            y.append(label_map[label])
    
    return np.array(X), np.array(y), label_map, label_names

# ======= CARREGANDO DADOS =======
print("Carregando imagens de treino e avaliação...")

X_train, y_train, label_map, label_names = carregar_imagens(TRAIN_PATH)
X_test, y_test, _, _ = carregar_imagens(EVAL_PATH)

y_train_cat = to_categorical(y_train, num_classes=len(label_names))
y_test_cat = to_categorical(y_test, num_classes=len(label_names))

# ======= DEFINIÇÃO DO MODELO =======
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_names), activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ======= TREINAMENTO =======
print("Treinando modelo...")
start_time = time.time()
history = model.fit(X_train, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
training_time = time.time() - start_time
print(f"Tempo de treinamento: {training_time:.2f} segundos")

# ======= INFERÊNCIA =======
print("Fazendo predições no conjunto de teste...")
start_time = time.time()
y_pred_probs = model.predict(X_test)
inference_time = time.time() - start_time
y_pred = np.argmax(y_pred_probs, axis=1)

acc = np.mean(y_pred == y_test)
print(f"\nAcurácia total no teste: {acc*100:.2f}%")
print(f"Tempo de inferência: {inference_time:.2f} segundos para {len(X_test)} imagens")

# ======= MATRIZ DE CONFUSÃO =======
cm = confusion_matrix(y_test, y_pred, labels=range(len(label_names)))
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(12, 10))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão (% de acertos por classe)")
plt.tight_layout()
plt.show()

# ======= RELATÓRIO OPCIONAL =======
print("\nRelatório detalhado:")
print(classification_report(y_test, y_pred, target_names=label_names))