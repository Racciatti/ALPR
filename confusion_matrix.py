import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

# --- Configuration ---
BATCH_SIZE = 32
IMG_HEIGHT = 30
IMG_WIDTH = 40
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)

print("🔄 Carregando dados de validação...")

# --- Load Validation Data ---
caminho_eval = 'data/processed_eval'

val_ds = tf.keras.utils.image_dataset_from_directory(
    caminho_eval,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    image_size=IMG_SHAPE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = val_ds.class_names
print(f"✅ Encontradas {len(class_names)} classes: {class_names}")

# --- Normalize data ---
normalization_layer = tf.keras.layers.Rescaling(1./255)
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)

print("🤖 Carregando modelo quantizado TensorFlow Lite...")

# --- Load the quantized TFLite model ---
try:
    interpreter = tf.lite.Interpreter(model_path='models/final_model_quant.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    print("✅ Modelo TFLite carregado com sucesso!")
    print(f"   Input shape: {input_details['shape']}")
    print(f"   Input dtype: {input_details['dtype']}")
    print(f"   Output shape: {output_details['shape']}")
    print(f"   Output dtype: {output_details['dtype']}")
except Exception as e:
    print(f"❌ Erro ao carregar o modelo TFLite: {e}")
    print("   Verifique se o arquivo 'models/final_model_quant.tflite' existe.")
    exit()

print("📊 Gerando predições...")

# Helper function to run inference with TFLite model
def predict_with_tflite(interpreter, input_details, output_details, images):
    """Run inference on a batch of images using TFLite interpreter"""
    predictions = []
    
    for i in range(images.shape[0]):
        # Get single image
        image = images[i:i+1]
        
        # Check if input needs to be quantized (INT8)
        if input_details['dtype'] == np.int8:
            input_scale, input_zero_point = input_details["quantization"]
            image = tf.cast((image / input_scale + input_zero_point), dtype=tf.int8)
        
        # Set input tensor
        interpreter.set_tensor(input_details['index'], image)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_tensor(output_details['index'])
        
        # If output is quantized, dequantize it
        if output_details['dtype'] == np.int8:
            output_scale, output_zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(output[0])
    
    return np.array(predictions)

# 1. Obter todas as previsões e rótulos verdadeiros do conjunto de validação
y_pred = []
y_true = []

# Iterar sobre o dataset de validação para coletar os dados
for images, labels in val_ds:
    # Converter para numpy se necessário
    if hasattr(images, 'numpy'):
        images_np = images.numpy()
    else:
        images_np = images
    
    # Fazer previsões para o lote de imagens usando TFLite
    predictions = predict_with_tflite(interpreter, input_details, output_details, images_np)
    
    # Converter as probabilidades de previsão para a classe com maior probabilidade (o rótulo previsto)
    predicted_labels = np.argmax(predictions, axis=1)
    
    y_pred.extend(predicted_labels)
    y_true.extend(labels.numpy())

# Converter as listas para arrays numpy
y_pred = np.array(y_pred)
y_true = np.array(y_true)

print(f"✅ Processadas {len(y_pred)} amostras")

# --- Calculate accuracy ---
accuracy = np.mean(y_pred == y_true)
print(f"🎯 Acurácia do modelo: {accuracy*100:.4f}%")

print("📈 Calculando matriz de confusão...")

# 2. Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Calculate per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

print("🎨 Gerando visualizações...")

# Create output directory for plots
os.makedirs('confusion_matrix_plots', exist_ok=True)

# 3. Visualizar a matriz de confusão com um Heatmap (Gráfico de Calor)
plt.figure(figsize=(22, 18)) # Aumenta o tamanho da figura para caberem as 35 classes
sns.heatmap(
    cm, 
    annot=True, # Exibe os números dentro de cada célula
    fmt='d',    # Formata os números como inteiros
    cmap='Blues', 
    xticklabels=class_names, 
    yticklabels=class_names,
    cbar_kws={'label': 'Número de Amostras'}
)
plt.title(f'Matriz de Confusão (TFLite INT8) - Acurácia: {accuracy*100:.2f}%', fontsize=20, pad=20)
plt.ylabel('Rótulo Verdadeiro (True Label)', fontsize=16)
plt.xlabel('Rótulo Previsto (Predicted Label)', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_plots/confusion_matrix_absolute.png', dpi=300, bbox_inches='tight')
print("✅ Matriz de confusão absoluta salva em: confusion_matrix_plots/confusion_matrix_absolute.png")
plt.close()

# 4. Create a percentage-based confusion matrix
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

plt.figure(figsize=(22, 18))
sns.heatmap(
    cm_percent, 
    annot=True, 
    fmt='.1f',
    cmap='RdYlGn', 
    xticklabels=class_names, 
    yticklabels=class_names,
    cbar_kws={'label': 'Porcentagem (%)'},
    vmin=0, vmax=100
)
plt.title(f'Matriz de Confusão (%) - TFLite INT8 - Acurácia por Classe', fontsize=20, pad=20)
plt.ylabel('Rótulo Verdadeiro (True Label)', fontsize=16)
plt.xlabel('Rótulo Previsto (Predicted Label)', fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_plots/confusion_matrix_percentage.png', dpi=300, bbox_inches='tight')
print("✅ Matriz de confusão percentual salva em: confusion_matrix_plots/confusion_matrix_percentage.png")
plt.close()

# 5. Display per-class accuracy statistics
print("\n📊 Estatísticas por Classe:")
print("="*50)
accuracy_df = pd.DataFrame({
    'Classe': class_names,
    'Acurácia (%)': per_class_accuracy,
    'Total de Amostras': cm.sum(axis=1),
    'Predições Corretas': cm.diagonal()
}).sort_values('Acurácia (%)', ascending=False)

print(accuracy_df.to_string(index=False, float_format='%.2f'))

# Save accuracy statistics to CSV
accuracy_df.to_csv('confusion_matrix_plots/per_class_accuracy.csv', index=False, float_format='%.2f')
print("✅ Estatísticas por classe salvas em: confusion_matrix_plots/per_class_accuracy.csv")

# 6. Show worst performing classes
print(f"\n⚠️  Classes com menor acurácia:")
worst_classes = accuracy_df.head(5)
for _, row in worst_classes.iterrows():
    print(f"   {row['Classe']}: {row['Acurácia (%)']:.2f}% ({row['Predições Corretas']:.0f}/{row['Total de Amostras']:.0f})")

# 7. Show best performing classes
print(f"\n🏆 Classes com maior acurácia:")
best_classes = accuracy_df.tail(5)
for _, row in best_classes.iterrows():
    print(f"   {row['Classe']}: {row['Acurácia (%)']:.2f}% ({row['Predições Corretas']:.0f}/{row['Total de Amostras']:.0f})")

# 8. (Opcional) Exibir a matriz como a tabela que você sugeriu, usando Pandas
print("\n📋 Matriz de Confusão em formato de Tabela:")
print("="*50)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
print(df_cm)

# Save confusion matrix to CSV
df_cm.to_csv('confusion_matrix_plots/confusion_matrix_table.csv')
print("✅ Matriz de confusão salva em: confusion_matrix_plots/confusion_matrix_table.csv")

# 9. Generate detailed classification report
print("\n📄 Relatório Detalhado de Classificação:")
print("="*70)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Save classification report
with open('confusion_matrix_plots/classification_report.txt', 'w') as f:
    f.write(f"Modelo: TensorFlow Lite Quantizado (INT8)\n")
    f.write(f"Arquivo: models/final_model_quant.tflite\n")
    f.write(f"Acurácia Geral do Modelo: {accuracy*100:.4f}%\n")
    f.write(f"Total de amostras processadas: {len(y_pred)}\n")
    f.write("="*70 + "\n")
    f.write(report)
print("✅ Relatório de classificação salvo em: confusion_matrix_plots/classification_report.txt")

print(f"\n✅ Análise completa da matriz de confusão finalizada!")
print(f"📁 Todos os arquivos foram salvos na pasta: confusion_matrix_plots/")
print(f"📊 Arquivos gerados:")
print(f"   - confusion_matrix_absolute.png (matriz absoluta)")
print(f"   - confusion_matrix_percentage.png (matriz percentual)")
print(f"   - per_class_accuracy.csv (estatísticas por classe)")
print(f"   - confusion_matrix_table.csv (matriz em formato tabela)")
print(f"   - classification_report.txt (relatório completo)")