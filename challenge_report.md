# CNN de Alto Desempenho para Reconhecimento de Caracteres de Placas
Data do Projeto: 31 de julho de 2025

## 1. Visão Geral
Este projeto detalha o desenvolvimento de uma Rede Neural Convolucional (CNN) altamente eficiente para Reconhecimento Óptico de Caracteres (OCR) de caracteres individuais de placas de veículos. O objetivo principal foi alcançar precisão de ponta e principalmente minimizar o custo computacional do tempo de inferência, um fator crítico para aplicações do mundo real, as quais são majoritariamente baseadas em edge computing.

A solução final é uma CNN com arquitetura customizada, ajustada e quantizada que alcança 99,78% de precisão no conjunto de validação com um tempo médio de inferência de apenas 0,49 ms por caractere em uma GPU T4.

## 2. Metodologia e Raciocínio
A estratégia central foi construir um modelo pequeno, rápido e preciso do zero, ao invés de adaptar arquiteturas grandes e pré-treinadas como VGG ou AlexNet. Esta decisão foi baseada no raciocínio de que tais modelos são computacionalmente excessivos para uma tarefa restrita como reconhecimento de caracteres únicos e teriam desempenho ruim sob requisitos rigorosos de eficiência.

O desenvolvimento seguiu um processo sistemático e multi-estágio:

**Pré-processamento de Dados**: Um pipeline foi criado para limpar, normalizar e aumentar o conjunto de dados para garantir entrada de alta qualidade para o modelo.

**Ajuste Sistemático de Hiperparâmetros**: Uma busca estruturada foi conduzida para encontrar a taxa de aprendizado ótima, capacidade do modelo e parâmetros de regularização.

**Ajuste Fino Avançado**: O modelo de melhor desempenho foi submetido a uma fase de polimento final usando decaimento da taxa de aprendizado e treinamento de baixa taxa para maximizar a precisão.

**Quantização Pós-Treinamento**: O modelo final foi convertido para formato INT8 para reduzir drasticamente seu tamanho e tempo de inferência com uma compensação mínima na precisão.

Todos os experimentos foram conduzidos em um ambiente Google Colab usando uma GPU T4 para acelerar o treinamento.

## 3. Pipeline de Pré-processamento de Dados
Um pipeline de pré-processamento robusto foi crucial para alcançar alta precisão. Todos os passos foram implementados usando OpenCV e Albumentations para alto desempenho, com processamento paralelo para lidar eficientemente com o grande conjunto de dados.

**Redimensionamento de Imagem**: Todas as imagens de origem foram redimensionadas para 40x30 pixels uniformes. Este passo reduziu significativamente a carga computacional para a CNN enquanto retinha detalhes suficientes para reconhecimento de caracteres. cv2.INTER_AREA foi usado para redimensionamento ótimo.

**Limiarização**: A Binarização de Otsu (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) foi aplicada para segmentar o caractere do fundo. Isso simplificou a tarefa de aprendizado criando imagens de alto contraste em preto e branco.

**Limpeza de Dados**: Um script de análise de contornos foi implementado para descartar automaticamente imagens ruidosas ou mal segmentadas do conjunto de dados limiarizado. Este passo filtrou imagens baseado no número de contornos e na área do maior contorno, garantindo que apenas amostras de alta qualidade fossem usadas para treinamento.

**Aumento de Dados**: O conjunto de dados limpo foi aumentado para incrementar seu tamanho e variância, melhorando as capacidades de generalização do modelo. Os aumentos incluíram rotações leves, deslocamentos, escalonamento e desfoque de movimento.

O conjunto de dados final de treinamento consistiu de 251.471 imagens, com 10.650 imagens reservadas para o conjunto de validação.

## 4. Arquitetura do Modelo e Treinamento
Uma arquitetura CNN customizada inspirada na LeNet foi projetada para esta tarefa. O modelo é leve mas poderoso o suficiente para aprender as características distintivas das 35 classes de caracteres.

A arquitetura final consiste de:

- Dois blocos convolucionais, cada um com Conv2D -> BatchNormalization -> MaxPooling2D.
- Uma cabeça classificadora com Flatten -> Dense -> BatchNormalization -> Dropout -> Dense (Saída).

## 5. Ajuste de Hiperparâmetros e Otimização
Uma abordagem sistemática e multi-estágio foi usada para encontrar a configuração ótima do modelo. A precisão de validação (val_accuracy) foi a única métrica usada para seleção do modelo em cada estágio.

**Taxa de Aprendizado**: Experimentos com 1e-3, 5e-4 e 1e-4 identificaram 5e-4 como a taxa ótima, fornecendo o melhor equilíbrio de estabilidade de treinamento e desempenho de pico.

**Capacidade do Modelo**: Versões pequena, média e grande do modelo foram testadas. A capacidade média (filters=[32, 64], dense_units=128) foi selecionada como vencedora.

**Taxa de Dropout**: Taxas de 0,3, 0,4 e 0,5 foram testadas. Uma taxa de 0,4 forneceu a melhor regularização.

**Otimizador**: O otimizador AdamW foi testado contra o Adam baseline mas não produziu melhoria significativa, então Adam foi mantido.

**Polimento Final**: O modelo vencedor foi retreinado com um callback ReduceLROnPlateau e paciência aumentada (10 épocas). Isso foi seguido por uma execução final de treinamento de 5 épocas com uma taxa de aprendizado muito baixa e fixa de 1e-5 para alcançar precisão máxima.

## 6. Otimização Final: Quantização Pós-Treinamento
Para atender aos requisitos de eficiência, o modelo Keras final totalmente treinado (FP32) foi convertido para um modelo TensorFlow Lite (.tflite) com quantização completa de inteiros de 8 bits (INT8). Este passo é crítico para implantação em dispositivos edge e para minimizar custos de inferência.

## 7. Resultados Finais
A abordagem sistemática produziu um modelo com desempenho de ponta tanto em métricas de precisão quanto de eficiência.

| Métrica | Modelo Original (FP32) | Modelo Quantizado (INT8) | Melhoria |
|---------|------------------------|--------------------------|----------|
| Precisão de Validação | 99,8028% | 99,7840% | -0,02% (negligível) |
| Tempo de Inferência | 70,41 ms | 0,49 ms | ~143x Mais Rápido |
| Tamanho do Modelo | ~2,4 MB | ~0,6 MB | 4x Menor |

O modelo quantizado final fornece um aumento massivo na eficiência computacional ao custo de uma queda estatisticamente insignificante na precisão, tornando-o uma solução ideal para a competição.