import pandas as pd
import matplotlib.pyplot as plt

# 1. Carregamento dos dados
# Certifique-se de que o arquivo CSV esteja na pasta 'dataset' conforme o roteiro
try:
    df = pd.read_csv('new_dataset/trabalho_demanda.csv')
    df_param = pd.read_csv('new_dataset/trabalho_parametros.csv', index_col=0)
except FileNotFoundError:
    print("Erro: Arquivo não encontrado. Verifique se o caminho 'dataset/trabalho_demanda.csv' está correto.")

# Leitura dinâmica das colunas e mapeamento dos produtos
colunas = df.columns[1:].tolist()
produtos_map = df_param.loc['produtos'].to_dict()
nomes_linhas = [f"{col}: {produtos_map.get(col, 'Produto ' + col)}" for col in colunas]

from funções.UTILS import obter_cores_dinamicas
cores = obter_cores_dinamicas(len(colunas))

fig, axes = plt.subplots(nrows=len(colunas), ncols=1, figsize=(12, max(4, 3.5 * len(colunas))), sharex=True)
if len(colunas) == 1: axes = [axes] # Garante que axes seja iterável
fig.suptitle('Comportamento Histórico da Demanda (2024-2025)', fontsize=16, y=0.92)

# 3. Plotagem iterativa para cada linha de produção
for i, (col, ax) in enumerate(zip(colunas, axes)):
    ax.plot(df.index, df[col], marker='o', linestyle='-', color=cores[i], linewidth=2)
    ax.set_title(nomes_linhas[i], fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume (un)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

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

fig_trat, axes_trat = plt.subplots(nrows=len(colunas), ncols=1, figsize=(12, max(4, 3.5 * len(colunas))), sharex=True)
if len(colunas) == 1: axes_trat = [axes_trat]
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

fig_diag, axes_diag = plt.subplots(nrows=len(colunas), ncols=1, figsize=(12, max(4, 3.5 * len(colunas))), sharex=True)
if len(colunas) == 1: axes_diag = [axes_diag]
fig_diag.suptitle('Diagnóstico de Anomalias (Limites IQR e Level Shift)', fontsize=16, y=0.92)

for i, (col, ax) in enumerate(zip(colunas, axes_diag)):
    demandas_brutas = df[col].tolist()
    rel = analisar_anomalias(demandas_brutas)

    # Série bruta — sem qualquer modificação
    ax.plot(df.index, demandas_brutas, marker='o', linestyle='-', color=cores[i], linewidth=2, label='Demanda Bruta', zorder=2)

    # Limites IQR (apenas linhas horizontais de referência)
    ax.axhline(y=rel['limite_sup'], color='red',  linestyle='--', linewidth=1.5, alpha=0.7, label=f"Limite Sup. IQR ({rel['limite_sup']:,.0f})")
    ax.axhline(y=rel['limite_inf'], color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label=f"Limite Inf. IQR ({rel['limite_inf']:,.0f})")

    # Marcação de outliers com 'X' — sem substituição de valor
    if rel['n_outliers'] > 0:
        for idx in rel['indices_outliers']:
            ax.plot(df.index[idx], demandas_brutas[idx],
                    marker='x', color='red', markersize=12,
                    markeredgewidth=2.5, zorder=4, label='Outlier (IQR)')

    # Linha vertical no ponto de level shift detectado — sem ajuste de médias
    if rel['level_shift']:
        ponto = rel['ponto_shift']
        ax.axvline(x=df.index[ponto], color='purple', linestyle=':',
                   linewidth=2, alpha=0.8, label=f'Level Shift (período {ponto + 1})')

    ax.set_title(nomes_linhas[i] + ' (Diagnóstico)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Volume (un)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Legenda sem rótulos duplicados
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

axes_diag[-1].set_xlabel('Períodos (Meses de 1 a 24)', fontsize=12)
axes_diag[-1].set_xticks(df.index)
axes_diag[-1].set_xticklabels([f'{i+1}' for i in df.index])

plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig('grafico_historico_demanda_diagnostico.png', dpi=300, bbox_inches='tight')

print("Gráficos de análise de demanda gerados e salvos (incluindo diagnóstico).")
plt.show()

