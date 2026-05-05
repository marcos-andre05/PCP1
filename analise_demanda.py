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

# ==========================================================
# 5. Gerar gráfico da demanda TRATADA
# ==========================================================
from funções.TRATAMENTO import tratar_anomalias_demanda

fig_trat, axes_trat = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)
fig_trat.suptitle('Comportamento Histórico da Demanda Tratada (2024-2025)', fontsize=16, y=0.92)

for i, (col, ax) in enumerate(zip(colunas, axes_trat)):
    demandas_brutas = df[col].tolist()
    demandas_tratadas = tratar_anomalias_demanda(demandas_brutas)
    
    ax.plot(df.index, demandas_brutas, marker='', linestyle=':', color='#AAAAAA', linewidth=1.5, label='Bruta', zorder=1)
    ax.plot(df.index, demandas_tratadas, marker='o', linestyle='-', color=cores[i], linewidth=2, label='Tratada', zorder=2)
    
    ax.set_title(nomes_linhas[i] + ' (Tratada)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume (un)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

axes_trat[-1].set_xlabel('Períodos (Meses de 1 a 24)', fontsize=12)
axes_trat[-1].set_xticks(df.index)
axes_trat[-1].set_xticklabels([f'{i+1}' for i in df.index])

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig('grafico_historico_demanda_tratada.png', dpi=300, bbox_inches='tight')

# ==========================================================
# 6. Gerar gráfico de DIAGNÓSTICO (Limites IQR e Level Shift)
# ==========================================================
from funções.TRATAMENTO import analisar_anomalias

fig_diag, axes_diag = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)
fig_diag.suptitle('Diagnóstico de Anomalias (Limites IQR e Level Shift)', fontsize=16, y=0.92)

for i, (col, ax) in enumerate(zip(colunas, axes_diag)):
    demandas_brutas = df[col].tolist()
    rel = analisar_anomalias(demandas_brutas)
    
    ax.plot(df.index, demandas_brutas, marker='o', linestyle='-', color=cores[i], linewidth=2, label='Bruta', zorder=2)
    
    # Plot IQR limits
    ax.axhline(y=rel['limite_sup'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Limite Sup. IQR', zorder=1)
    ax.axhline(y=rel['limite_inf'], color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Limite Inf. IQR', zorder=1)
    
    # Highlight outliers
    if rel['n_outliers'] > 0:
        for idx in rel['indices_outliers']:
            ax.plot(df.index[idx], demandas_brutas[idx], marker='x', color='red', markersize=10, markeredgewidth=2, zorder=3)
            
    # Plot Level Shift
    if rel['level_shift']:
        ponto = rel['ponto_shift']
        ax.axvline(x=df.index[ponto], color='purple', linestyle=':', linewidth=2, alpha=0.8, label='Ponto Level Shift', zorder=1)
        # Plot media_ant
        ax.plot(df.index[:ponto], [rel['media_ant']]*ponto, color='green', linestyle='--', linewidth=2, label='Média Antes', zorder=1)
        # Plot media_post
        ax.plot(df.index[ponto:], [rel['media_post']]*(len(demandas_brutas)-ponto), color='orange', linestyle='--', linewidth=2, label='Média Depois', zorder=1)

    ax.set_title(nomes_linhas[i] + ' (Diagnóstico)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume (un)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Avoid duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if len(by_label) > 1: # Only show legend if there's more than just 'Bruta'
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')

axes_diag[-1].set_xlabel('Períodos (Meses de 1 a 24)', fontsize=12)
axes_diag[-1].set_xticks(df.index)
axes_diag[-1].set_xticklabels([f'{i+1}' for i in df.index])

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig('grafico_historico_demanda_diagnostico.png', dpi=300, bbox_inches='tight')

print("Gráficos de análise de demanda gerados e salvos (incluindo diagnóstico).")
