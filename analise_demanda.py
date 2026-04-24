import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregamento dos dados
# Certifique-se de que o arquivo CSV esteja na pasta 'dataset' conforme o roteiro
try:
    df = pd.read_csv('dataset/trabalho_demanda.csv')
except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Verifique se o caminho 'dataset/trabalho_demanda.csv' está correto.")
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)
fig.suptitle('Comportamento Histórico da Demanda (2024-2025)', fontsize=16, y=0.92)

# Nomes das linhas para os títulos
nomes_linhas = [
    'L1: Discos de Freio', 
    'L2: Cubos de Roda', 
    'L3: Tambores de Freio', 
    'L4: Flanges Industriais', 
    'L5: Suportes Metálicos'
]

colunas = ['L1', 'L2', 'L3', 'L4', 'L5']
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 3. Plotagem iterativa para cada linha de produção
for i, (col, ax) in enumerate(zip(colunas, axes)):
    ax.plot(df.index, df[col], marker='o', linestyle='-', color=cores[i], linewidth=2)
    ax.set_title(nomes_linhas[i], fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume (un)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Destacando pontos atípicos visualmente (ex: outliers e mudanças bruscas)
    if col == 'L2':
        ax.annotate('Outlier provável', xy=(15, 6928), xytext=(16, 10000),
                    arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=9)

# Configuração do eixo X (Meses)
axes[-1].set_xlabel('Períodos (Meses de 1 a 24)', fontsize=12)
axes[-1].set_xticks(df.index)
axes[-1].set_xticklabels([f'{i+1}' for i in df.index])

plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # Ajusta o layout para não cortar o título

# 4. Salvar o gráfico em alta qualidade para o artigo
plt.savefig('grafico_historico_demanda.png', dpi=300, bbox_inches='tight')
plt.show()