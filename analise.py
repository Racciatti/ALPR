import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import os

# Create output directory for analysis plots
os.makedirs('analise_plots', exist_ok=True)

# === 1. LER E NORMALIZAR A MATRIZ DE CONFUSÃO ===
cm_df = pd.read_csv('./confusion_matrix_plots/confusion_matrix_table.csv', index_col=0)
cm = cm_df.values
classes = list(cm_df.columns)

with np.errstate(divide='ignore', invalid='ignore'):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = np.nan_to_num(cm_normalized)

# === 2. MATRIZ DE CONFUSÃO COM DESTAQUE DE ERROS ===
print("\n" + "="*50)
print("🔎 Matriz de Confusão Final com Destaque de Erros")
print("="*50)

# Criar um array de textos para anotações personalizadas
annot_labels = (np.asarray([f"{val:.1%}" if val > 0 else "" for val in cm_normalized.flatten()])).reshape(cm_normalized.shape)

plt.figure(figsize=(25, 22))
ax = sns.heatmap(
    cm_normalized,
    annot=annot_labels,
    fmt='',
    cmap='Blues',
    xticklabels=classes,
    yticklabels=classes,
    annot_kws={"size": 10},
    cbar=False
)

# --- NOVO: MUDAR A COR DOS TEXTOS DE ERRO PARA VERMELHO ---
for i in range(len(classes)):
    for j in range(len(classes)):
        # O texto correspondente está na lista ax.texts
        text_index = i * len(classes) + j
        if i != j and cm_normalized[i, j] > 0: # Se for um erro e o valor for maior que zero
            ax.texts[text_index].set_color('#C40000') # Define a cor como um vermelho forte
        elif i == j: # Se estiver na diagonal (acertos)
            ax.texts[text_index].set_weight('bold') # Coloca o texto em negrito
# --- FIM DA MUDANÇA ---

plt.title("Matriz de Confusão Normalizada (% de Previsões por Classe Verdadeira)", fontsize=22)
plt.xlabel("Classe Prevista", fontsize=18)
plt.ylabel("Classe Verdadeira", fontsize=18)
plt.xticks(fontsize=14, rotation=0) # Mantém os rótulos retos
plt.yticks(fontsize=14)
plt.tight_layout(pad=3.0)
plt.savefig('analise_plots/matriz_confusao_normalizada.png', dpi=300, bbox_inches='tight')
print("✅ Matriz de confusão normalizada salva em: analise_plots/matriz_confusao_normalizada.png")
plt.close()


# === 3. RELATÓRIO DE CLASSIFICAÇÃO (sem alteração) ===
print("\n" + "="*50)
print("Relatório de Classificação Detalhado")
print("="*50)
true_labels, predicted_labels = [], []
for i in range(len(classes)):
    for j in range(len(classes)):
        count = cm[i, j]
        true_labels.extend([i] * count)
        predicted_labels.extend([j] * count)
report = classification_report(true_labels, predicted_labels, target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Print the classification report
print(report_df.round(4))

# Save classification report to CSV
report_df.to_csv('analise_plots/classification_report_detailed.csv', float_format='%.4f')
print("✅ Relatório de classificação detalhado salvo em: analise_plots/classification_report_detailed.csv")


# === 4. GRÁFICO DE ACURÁCIA POR CLASSE (RECALL) COM ZOOM ===
acuracia_por_classe = cm_normalized.diagonal()
plt.figure(figsize=(18, 7))
sns.barplot(x=classes, y=acuracia_por_classe, palette='crest')

# --- NOVO: APLICANDO ZOOM NO EIXO Y ---
plt.ylim(0.95, 1.005) # Define o limite inferior como 95% e superior como 100.5%
# --- FIM DA MUDANÇA ---

plt.title("Acurácia por Classe (Recall) - Zoom em 95%-100%", fontsize=18)
plt.ylabel("Acurácia (%)", fontsize=14)
plt.xlabel("Classe", fontsize=14)
# Formata os rótulos do eixo Y para exibir como porcentagens
plt.gca().set_yticklabels(['{:.1f}%'.format(y*100) for y in plt.gca().get_yticks()])
plt.savefig('analise_plots/acuracia_por_classe_zoom.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico de acurácia por classe salvo em: analise_plots/acuracia_por_classe_zoom.png")
plt.close()


# === 5. & 6. ANÁLISES DE PIORES DESEMPENHOS E ERROS (sem alteração) ===
piores_classes_df = pd.DataFrame({'Classe': classes, 'Acurácia': acuracia_por_classe})
piores_classes_df = piores_classes_df.sort_values('Acurácia', ascending=True)
print("\n" + "="*50)
print("Classes com Pior Desempenho (Menor Acurácia)")
print("="*50)
print(piores_classes_df.head(5).to_string(index=False, float_format='%.4f'))

# Save worst performing classes to CSV
piores_classes_df.to_csv('analise_plots/piores_classes.csv', index=False, float_format='%.4f')
print("✅ Classes com pior desempenho salvas em: analise_plots/piores_classes.csv")

print("\n" + "="*50)
print("Análise de Erros: Confusões Mais Relevantes")
print("="*50)
confusoes = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm_normalized[i, j] > 0:
            confusoes.append((classes[i], classes[j], cm_normalized[i, j], cm[i, j]))
confusoes_ordenadas = sorted(confusoes, key=lambda x: x[2], reverse=True)
print("As 10 confusões mais impactantes (Classe Verdadeira → Classe Prevista):\n")
for real, pred, percent, count in confusoes_ordenadas[:10]:
    print(f"- {percent:.2%} das vezes que era um '{real}', o modelo previu '{pred}' (total de {count} erros).")

# Save confusion analysis to file
with open('analise_plots/analise_confusoes.txt', 'w') as f:
    f.write("Análise de Erros: Confusões Mais Relevantes\n")
    f.write("="*50 + "\n")
    f.write("As 10 confusões mais impactantes (Classe Verdadeira → Classe Prevista):\n\n")
    for real, pred, percent, count in confusoes_ordenadas[:10]:
        f.write(f"- {percent:.2%} das vezes que era um '{real}', o modelo previu '{pred}' (total de {count} erros).\n")

print("✅ Análise de confusões salva em: analise_plots/analise_confusoes.txt")

print(f"\n✅ Análise completa finalizada!")
print(f"📁 Todos os arquivos foram salvos na pasta: analise_plots/")
print(f"📊 Arquivos gerados:")
print(f"   - matriz_confusao_normalizada.png (matriz com destaque de erros)")
print(f"   - acuracia_por_classe_zoom.png (gráfico de acurácia com zoom)")
print(f"   - classification_report_detailed.csv (relatório detalhado)")
print(f"   - piores_classes.csv (classes com pior desempenho)")
print(f"   - analise_confusoes.txt (análise das principais confusões)")