# ALPR - Automatic License Plate Recognition

> A high-performance CNN for vehicle license plate character recognition optimized for edge computing

## 📋 Overview

This project implements a complete **Automatic License Plate Recognition (ALPR)** solution specifically focused on individual character recognition using a custom Convolutional Neural Network. The main objective was to develop a model that combines **high accuracy** with **extreme computational efficiency**.

### 🎯 Results Achieved

| Metric | Original Model (FP32) | Quantized Model (INT8) | Improvement |
|---------|------------------------|--------------------------|----------|
| **Accuracy** | 99.80% | 99.78% | -0.02% (negligible) |
| **Inference Time** | 70.41ms | 0.49ms | **143x faster** |
| **Model Size** | ~2.4MB | ~0.6MB | **4x smaller** |

## 🏗️ Model Architecture

### Custom CNN (LeNet-inspired)
```
Input: (30, 40, 1) - Grayscale images
│
├─ Conv2D(32, 3x3) → BatchNorm → MaxPool2D(2x2)
├─ Conv2D(64, 3x3) → BatchNorm → MaxPool2D(2x2)
│
├─ Flatten
├─ Dense(128) → BatchNorm → Dropout(0.4)
└─ Dense(35) → Softmax
```

### Recognized Classes (35 total)
- **Digits**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Letters**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, P, Q, R, S, T, U, V, W, X, Y, Z

## 🔄 Processing Pipeline

### 1. Preprocessing
```
Original Data → Resizing (40x30) → Binarization (Otsu) → Cleaning → Data Augmentation
```

- **Resizing**: 100x75 --> 40x30 (6.25x reduction in pixel count)
- **Binarization**: Otsu method for optimal segmentation
- **Automatic cleaning**: Removal of low-quality samples (~10% of dataset)
- **Data augmentation**: 10x increase in sample count

### 2. Data Structure
```
data/
├── train/           # Original training data
├── eval/            # Original test data
├── resized/         # Resized images
├── thresholded/     # Binarized images
├── cleaned/         # Clean data
├── augmented_data/  # Augmented data (final training)
└── processed_eval/  # Processed evaluation data
```

## ⚙️ Optimization and Training

### Systematic Methodology
1. **Hyperparameter search**: Learning rate, model capacity, dropout
2. **Refined training**: Advanced callbacks with adaptive LR reduction
3. **Final polishing**: Fine-tuning with ultra-low learning rate
4. **Quantization**: Conversion to INT8 using TensorFlow Lite

## 🚀 Installation and Usage

### Prerequisites
```bash
Python 3.8+
CUDA (optional, for GPU training)
```

### Installation
```bash
git clone https://github.com/racciatti/alpr.git
cd alpr
python -m venv venv
activate virtual environment (OS dependent)
pip install -r requirements.txt
```

### Basic Usage
```python
import tensorflow as tf

# Load quantized model
interpreter = tf.lite.Interpreter(model_path='models/final_model_quant.tflite')
interpreter.allocate_tensors()

# Make prediction on preprocessed image
# ... preprocessing code ...
interpreter.set_tensor(input_details['index'], processed_image)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details['index'])
```

## 📊 Project Structure

```
alpr/
├── data.ipynb              # Data processing pipeline
├── model.ipynb             # Model training and optimization
├── challenge_report.md     # Competition report
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔬 Key Technologies

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Image processing
- **Albumentations**: Data augmentation
- **TensorFlow Lite**: Optimization and quantization
- **scikit-learn**: Metrics and evaluation

## 📈 Performance

The model was specifically optimized for:
- **Real-time applications**
- **Resource-constrained devices**
- **Edge computing**
- **Low power consumption**

### Testing Environment
- **Hardware**: Google Colab GPU T4
- **Dataset**: 251,471 training images, 10,650 validation images

## 📄 License

This project is licensed under [LICENSE](LICENSE).

## 🤝 Contributions

Contributions are welcome! Please open an issue or pull request to discuss significant changes.

---

**Developed over ~6 hours for a FCT UNESP competition.**

---

# ALPR - Automatic License Plate Recognition

> Uma CNN de alto desempenho para reconhecimento de caracteres de placas veiculares otimizada para edge computing

## 📋 Visão Geral

Este projeto implementa uma solução completa de **Reconhecimento Automático de Placas (ALPR)** focada especificamente no reconhecimento de caracteres individuais usando uma Rede Neural Convolucional customizada. O objetivo principal foi desenvolver um modelo que combine **alta precisão** com **extrema eficiência computacional**.

### 🎯 Resultados Alcançados

| Métrica | Modelo Original (FP32) | Modelo Quantizado (INT8) | Melhoria |
|---------|------------------------|--------------------------|----------|
| **Precisão** | 99,80% | 99,78% | -0,02% (negligível) |
| **Tempo de Inferência** | 70,41ms | 0,49ms | **143x mais rápido** |
| **Tamanho do Modelo** | ~2,4MB | ~0,6MB | **4x menor** |

## 🏗️ Arquitetura do Modelo

### CNN Customizada (Inspirada na LeNet)
```
Entrada: (30, 40, 1) - Imagens em escala de cinza
│
├─ Conv2D(32, 3x3) → BatchNorm → MaxPool2D(2x2)
├─ Conv2D(64, 3x3) → BatchNorm → MaxPool2D(2x2)
│
├─ Flatten
├─ Dense(128) → BatchNorm → Dropout(0.4)
└─ Dense(35) → Softmax
```

### Classes Reconhecidas (35 total)
- **Dígitos**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Letras**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, P, Q, R, S, T, U, V, W, X, Y, Z

## 🔄 Pipeline de Processamento

### 1. Pré-processamento
```
Dados Originais → Redimensionamento (40x30) → Binarização (Otsu) → Limpeza → Aumento de Dados
```

- **Redimensionamento**: 100x75 --> 40x30 (redução de 6.25x no número de pixels)
- **Binarização**: Método de Otsu para segmentação ótima
- **Limpeza automática**: Remoção das amostras de baixa qualidade (~10% do dataset)
- **Data augmentation**: Aumento de 10x do número de amostras

### 2. Estrutura dos Dados
```
data/
├── train/           # Dados originais de treinamento
├── eval/            # Dados originais de teste
├── resized/         # Imagens redimensionadas
├── thresholded/     # Imagens limizarizadas
├── cleaned/         # Dados limpos
├── augmented_data/  # Dados aumentados (treinamento final)
└── processed_eval/  # Dados de avaliação processados
```

## ⚙️ Otimização e Treinamento

### Metodologia Sistemática
1. **Busca de hiperparâmetros**: Taxa de aprendizado, capacidade do modelo, dropout
2. **Treinamento refinado**: Callbacks avançados com redução adaptativa de LR
3. **Polimento final**: Fine-tuning com taxa de aprendizado ultra-baixa
4. **Quantização**: Conversão para INT8 usando TensorFlow Lite

## 🚀 Instalação e Uso

### Pré-requisitos
```bash
Python 3.8+
CUDA (opcional, para treinamento com GPU)
```

### Instalação
```bash
git clone https://github.com/racciatti/alpr.git
cd alpr
python -m venv venv
ativar ambiente virtual (depende do OS)
pip install -r requirements.txt
```

### Uso Básico
```python
import tensorflow as tf

# Carregar modelo quantizado
interpreter = tf.lite.Interpreter(model_path='models/final_model_quant.tflite')
interpreter.allocate_tensors()

# Fazer predição em imagem preprocessada
# ... código de preprocessing ...
interpreter.set_tensor(input_details['index'], processed_image)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details['index'])
```

## 📊 Estrutura do Projeto

```
alpr/
├── data.ipynb              # Pipeline de processamento de dados
├── model.ipynb             # Treinamento e otimização do modelo
├── challenge_report.md     # Relatório para competição
├── requirements.txt        # Dependências
└── README.md              # Este arquivo
```

## 🔬 Principais Tecnologias

- **TensorFlow/Keras**: Framework de deep learning
- **OpenCV**: Processamento de imagens
- **Albumentations**: Data augmentation
- **TensorFlow Lite**: Otimização e quantização
- **scikit-learn**: Métricas e avaliação

## 📈 Performance

O modelo foi otimizado especificamente para:
- **Aplicações em tempo real**
- **Dispositivos com recursos limitados**
- **Edge computing**
- **Baixo consumo energético**

### Ambiente de Teste
- **Hardware**: Google Colab GPU T4
- **Dataset**: 251.471 imagens de treinamento, 10.650 de validação

## 📄 Licença

Este projeto está licenciado sob [LICENSE](LICENSE).

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request para discutir mudanças significativas.

---

**Desenvolvido ao longo de ~6 horas para uma competição da FCT UNESP.**
