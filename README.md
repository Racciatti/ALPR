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
- **Letras**: A, B, C, D, E, F, G, H, J, K, L, M, N, P, Q, R, S, T, U, V, W, X, Y, Z
  
*Nota: Letras I e O não são utilizadas no padrão brasileiro de placas*

## 🔄 Pipeline de Processamento

### 1. Pré-processamento
```
Dados Originais → Redimensionamento (40x30) → Binarização (Otsu) → Limpeza → Aumento de Dados
```

- **Redimensionamento**: 40x30 pixels usando `cv2.INTER_AREA`
- **Binarização**: Método de Otsu para segmentação otimal
- **Limpeza automática**: Remoção de ~10% das amostras ruidosas
- **Data augmentation**: 10x aumento com transformações geométricas

### 2. Estrutura dos Dados
```
data/
├── train/           # Dados originais de treinamento
├── eval/            # Dados originais de teste
├── resized/         # Imagens redimensionadas
├── thresholded/     # Binarização Otsu
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

### Hiperparâmetros Otimizados
- **Taxa de aprendizado**: 5e-4 (melhor entre 1e-3, 5e-4, 1e-4)
- **Arquitetura**: Capacidade média [32, 64] filtros, 128 neurônios dense
- **Regularização**: Dropout 0.4, BatchNormalization
- **Otimizador**: Adam com early stopping

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
├── data-exploration.ipynb  # Análise exploratória
├── preprocessing/          # Utilitários de pré-processamento
├── challenge_report.md     # Relatório técnico detalhado
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
- **Tempo de treinamento**: ~2 horas (incluindo otimização de hiperparâmetros)

## 📄 Licença

Este projeto está licenciado sob [LICENSE](LICENSE).

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request para discutir mudanças significativas.

---

**Desenvolvido com foco em eficiência e precisão para aplicações reais de ALPR** 🚗