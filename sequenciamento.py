import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import os

from funções.UTILS import obter_cores_dinamicas

# ============================================================
#  SEQUENCIAMENTO DE ORDENS DE PRODUÇÃO
#  Regras testadas: FIFO, SPT, EDD, CR
# ============================================================

# ── Carregar dados ────────────────────────────────────────────
df_param = pd.read_csv('dataset/trabalho_parametros.csv', index_col=0)
linhas   = [c for c in df_param.columns if c.startswith('L')]

cores_lista = obter_cores_dinamicas(len(linhas))
cores = {linha: mcolors.to_hex(cores_lista[i]) for i, linha in enumerate(linhas)}

print(f"{'═'*80}")
print(f"  ⏱️  SEQUENCIAMENTO NO CHÃO DE FÁBRICA")
print(f"{'═'*80}")


def carregar_e_preparar(arquivo_seq, linha):
    """
    Carrega o CSV de sequenciamento e converte datas em prazos numéricos (dias).
    O dataset tem: ordem, chegada, tempo_processamento_dias, data_entrega.
    Converte para: ordem, tempo_processamento, prazo_dias (dias entre chegada e entrega).
    """
    df = pd.read_csv(arquivo_seq)

    # Converter colunas de data
    df['chegada'] = pd.to_datetime(df['chegada'])
    df['data_entrega'] = pd.to_datetime(df['data_entrega'])

    # Prazo = dias disponíveis entre a chegada e a data de entrega
    df['prazo_dias'] = (df['data_entrega'] - df['chegada']).dt.days

    # Renomear coluna de tempo de processamento para padronizar
    df = df.rename(columns={'tempo_processamento_dias': 'tempo_processamento'})

    return df[['ordem', 'tempo_processamento', 'prazo_dias', 'chegada', 'data_entrega']].copy()


def calcular_metricas(df_seq, nome_regra):
    """
    Calcula o Tempo de Fluxo e o Atraso para uma sequência dada.

    - Tempo de Fluxo: soma cumulativa dos tempos de processamento
    - Atraso: max(0, tempo_fluxo - prazo_dias)
    """
    df = df_seq.copy().reset_index(drop=True)

    # O tempo de fluxo é a soma cumulativa dos tempos de processamento
    df['tempo_fluxo'] = df['tempo_processamento'].cumsum()

    # O atraso é a diferença entre o tempo de fluxo e o prazo (se negativo, é 0)
    df['atraso'] = np.maximum(0, df['tempo_fluxo'] - df['prazo_dias'])

    fluxo_medio = df['tempo_fluxo'].mean()
    atraso_medio = df['atraso'].mean()
    n_atrasos = int((df['atraso'] > 0).sum())
    makespan = df['tempo_fluxo'].iloc[-1]

    return {
        'Regra': nome_regra,
        'Fluxo_Medio': round(fluxo_medio, 1),
        'Atraso_Medio': round(atraso_medio, 1),
        'Num_Atrasos': n_atrasos,
        'Makespan': int(makespan),
        'Sequencia': df['ordem'].tolist(),
        'detalhes': df[['ordem', 'tempo_processamento', 'prazo_dias',
                         'tempo_fluxo', 'atraso']].copy(),
    }


resultados_globais = {}

for linha in linhas:
    produto = df_param[linha]['produtos']
    arquivo_seq = f'dataset/trabalho_seq_{linha.lower()}.csv'

    # Carregar e preparar dados
    if os.path.exists(arquivo_seq):
        df_base = carregar_e_preparar(arquivo_seq, linha)
    else:
        # Gerador de dados simulados (8 ordens por linha)
        np.random.seed(int(linha[-1]) * 42)
        tempos = np.random.randint(2, 15, size=8)
        prazos = tempos + np.random.randint(5, 30, size=8)
        df_base = pd.DataFrame({
            'ordem': [f"OP-{linha[-1]}{i+1:02d}" for i in range(8)],
            'tempo_processamento': tempos,
            'prazo_dias': prazos,
        })
        print(f"  ⚠️  Dataset {arquivo_seq} não encontrado → dados simulados gerados")

    n_ordens = len(df_base)
    print(f"\n{'─'*80}")
    print(f"  🏭 Linha {linha} — {produto} ({n_ordens} Ordens em Espera)")
    print(f"{'─'*80}")

    # Exibir ordens de entrada
    df_entrada = df_base[['ordem', 'tempo_processamento', 'prazo_dias']].copy()
    df_entrada.columns = ['Ordem', 'Tempo Proc. (dias)', 'Prazo (dias)']
    print(f"\n  📋 Ordens de produção:")
    print(df_entrada.to_string(index=False))

    resultados_linha = []

    # 1. Regra FIFO (First In, First Out) - Mantém a ordem original do dataset
    df_fifo = df_base.copy()
    resultados_linha.append(calcular_metricas(df_fifo, 'FIFO'))

    # 2. Regra SPT (Shortest Processing Time) - Ordena pelo menor tempo de processamento
    df_spt = df_base.sort_values(by='tempo_processamento', ascending=True)
    resultados_linha.append(calcular_metricas(df_spt, 'SPT'))

    # 3. Regra EDD (Earliest Due Date) - Ordena pela data de entrega mais próxima
    df_edd = df_base.sort_values(by='prazo_dias', ascending=True)
    resultados_linha.append(calcular_metricas(df_edd, 'EDD'))

    # 4. Regra CR (Critical Ratio) - Razão Crítica: (Prazo / Tempo de Processamento)
    # Menor CR significa menos folga, logo tem maior prioridade
    df_cr = df_base.copy()
    df_cr['razao_critica'] = df_cr['prazo_dias'] / df_cr['tempo_processamento']
    df_cr = df_cr.sort_values(by='razao_critica', ascending=True)
    resultados_linha.append(calcular_metricas(df_cr, 'CR'))

    # Converter resultados para DataFrame (sem a coluna detalhes)
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != 'detalhes'}
                           for r in resultados_linha])
    resultados_globais[linha] = {
        'df_res': df_res,
        'detalhes': {r['Regra']: r['detalhes'] for r in resultados_linha},
        'resultados': resultados_linha,
    }

    # Encontrar as regras vencedoras
    melhor_fluxo = df_res.loc[df_res['Fluxo_Medio'].idxmin()]['Regra']
    melhor_atraso = df_res.loc[df_res['Atraso_Medio'].idxmin()]['Regra']

    # Imprimir tabela comparativa
    print(f"\n  📊 Comparativo de regras:")
    df_print = df_res[['Regra', 'Fluxo_Medio', 'Atraso_Medio', 'Num_Atrasos', 'Makespan']].copy()
    df_print.columns = ['Regra', 'Fluxo Médio (dias)', 'Atraso Médio (dias)',
                         'Nº Atrasos', 'Makespan (dias)']
    print(df_print.to_string(index=False))

    print(f"\n  🏆 Melhor Fluxo Médio: {melhor_fluxo}")
    print(f"  🏆 Melhor Atraso Médio: {melhor_atraso}")

    # Imprimir sequência detalhada da melhor regra por atraso
    det = resultados_globais[linha]['detalhes'][melhor_atraso]
    print(f"\n  📋 Sequência detalhada ({melhor_atraso}):")
    det_print = det.copy()
    det_print.columns = ['Ordem', 'Tempo Proc.', 'Prazo', 'Fluxo', 'Atraso']
    print(det_print.to_string(index=False))


# ============================================================
#  TABELA RESUMO CONSOLIDADA
# ============================================================

print(f"\n\n{'═'*80}")
print(f"  📊 RESUMO CONSOLIDADO — SEQUENCIAMENTO")
print(f"{'═'*80}")

resumo_rows = []
for linha in linhas:
    df_res = resultados_globais[linha]['df_res']
    produto = df_param[linha]['produtos']
    melhor_fluxo = df_res.loc[df_res['Fluxo_Medio'].idxmin()]['Regra']
    melhor_atraso = df_res.loc[df_res['Atraso_Medio'].idxmin()]['Regra']

    resumo_rows.append({
        'Linha': linha,
        'Produto': produto,
        'Melhor (Fluxo)': melhor_fluxo,
        'Fluxo Médio': df_res.loc[df_res['Fluxo_Medio'].idxmin()]['Fluxo_Medio'],
        'Melhor (Atraso)': melhor_atraso,
        'Atraso Médio': df_res.loc[df_res['Atraso_Medio'].idxmin()]['Atraso_Medio'],
        'Nº Atrasos': int(df_res.loc[df_res['Atraso_Medio'].idxmin()]['Num_Atrasos']),
    })

df_resumo = pd.DataFrame(resumo_rows)
print(df_resumo.to_string(index=False))


# ============================================================
#  GRÁFICOS
# ============================================================

regras = ['FIFO', 'SPT', 'EDD', 'CR']
n_linhas = len(linhas)
x = np.arange(len(regras))
width = 0.15

# ── Gráfico 1: Atraso Médio por regra e por linha ─────────────

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

for i, linha in enumerate(linhas):
    df_res = resultados_globais[linha]['df_res']
    # Garantir a ordem correta das regras
    atrasos = [float(df_res[df_res['Regra'] == r]['Atraso_Medio'].values[0]) for r in regras]
    pos = x - (width * (n_linhas - 1) / 2) + (i * width)
    bars = ax.bar(pos, atrasos, width, label=f'{linha} — {df_param[linha]["produtos"].split()[0]}',
                  color=cores[linha], alpha=0.85, zorder=2)
    # Rótulos de valor
    for bar, v in zip(bars, atrasos):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_title('Comparativo de Regras de Sequenciamento — Atraso Médio por Linha',
             fontsize=13, fontweight='bold', color='#1A1A2E', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(regras, fontsize=11, fontweight='bold')
ax.set_ylabel('Atraso Médio (dias)', fontsize=11)
ax.legend(title='Linhas de Produção', fontsize=9, title_fontsize=10,
          loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.25, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

plt.tight_layout()
plt.savefig('sequenciamento_atrasos.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"\n  📊 Gráfico de Atrasos salvo: sequenciamento_atrasos.png")


# ── Gráfico 2: Fluxo Médio por regra e por linha ─────────────

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

for i, linha in enumerate(linhas):
    df_res = resultados_globais[linha]['df_res']
    fluxos = [float(df_res[df_res['Regra'] == r]['Fluxo_Medio'].values[0]) for r in regras]
    pos = x - (width * (n_linhas - 1) / 2) + (i * width)
    bars = ax.bar(pos, fluxos, width, label=f'{linha} — {df_param[linha]["produtos"].split()[0]}',
                  color=cores[linha], alpha=0.85, zorder=2)
    for bar, v in zip(bars, fluxos):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_title('Comparativo de Regras de Sequenciamento — Fluxo Médio por Linha',
             fontsize=13, fontweight='bold', color='#1A1A2E', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(regras, fontsize=11, fontweight='bold')
ax.set_ylabel('Tempo de Fluxo Médio (dias)', fontsize=11)
ax.legend(title='Linhas de Produção', fontsize=9, title_fontsize=10,
          loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.25, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

plt.tight_layout()
plt.savefig('sequenciamento_fluxo.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"  📊 Gráfico de Fluxo salvo: sequenciamento_fluxo.png")


# ── Gráfico 3: Gantt simplificado — melhor regra por linha ───

fig, axes = plt.subplots(len(linhas), 1, figsize=(15, 3 * len(linhas)), sharex=False)
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Diagrama de Gantt — Melhor Regra (menor atraso) por Linha',
             fontsize=14, fontweight='bold', color='#1A1A2E', y=1.01)

for idx, (linha, ax) in enumerate(zip(linhas, axes)):
    ax.set_facecolor('#FFFFFF')
    df_res = resultados_globais[linha]['df_res']
    melhor = df_res.loc[df_res['Atraso_Medio'].idxmin()]['Regra']
    det = resultados_globais[linha]['detalhes'][melhor].reset_index(drop=True)
    produto = df_param[linha]['produtos']

    n_ordens = len(det)
    inicio = 0
    for i in range(n_ordens):
        tp = det.loc[i, 'tempo_processamento']
        prazo = det.loc[i, 'prazo_dias']
        atraso = det.loc[i, 'atraso']

        cor_barra = '#E53935' if atraso > 0 else cores[linha]
        ax.barh(i, tp, left=inicio, color=cor_barra, alpha=0.85,
                edgecolor='white', linewidth=0.8, zorder=2)
        ax.text(inicio + tp / 2, i, det.loc[i, 'ordem'],
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # Marca do prazo
        ax.plot(prazo, i, marker='|', color='#1A1A2E', markersize=15, markeredgewidth=2, zorder=3)

        inicio += tp

    ax.set_yticks(range(n_ordens))
    ax.set_yticklabels([f"{det.loc[i, 'ordem']}" for i in range(n_ordens)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Dias', fontsize=9)
    ax.set_title(f'{linha} — {produto}  |  Regra: {melhor}', fontsize=10,
                 fontweight='bold', color=cores[linha])
    ax.grid(True, alpha=0.15, linestyle='--', axis='x')
    for sp in ax.spines.values():
        sp.set_edgecolor('#CCCCCC')

    # Legenda inline
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cores[linha], alpha=0.85, label='No prazo'),
        Patch(facecolor='#E53935', alpha=0.85, label='Atrasado'),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5, loc='lower right', framealpha=0.85)

plt.tight_layout()
plt.savefig('sequenciamento_gantt.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"  📊 Gráfico Gantt salvo: sequenciamento_gantt.png")


# ============================================================
#  EXPORTAÇÃO CSV
# ============================================================

consolidados = []
for linha in linhas:
    df_res = resultados_globais[linha]['df_res'].copy()
    df_res.insert(0, 'Linha', linha)
    consolidados.append(df_res)

df_final = pd.concat(consolidados, ignore_index=True)
df_final.to_csv('sequenciamento_resumo.csv', index=False, encoding='utf-8-sig')
print(f"  📁 Tabela exportada: sequenciamento_resumo.csv")


# ============================================================
#  CONCLUSÃO
# ============================================================

print(f"\n{'═'*80}")
print(f"  ✅  Sequenciamento de produção concluído.")
print(f"     Linhas processadas: {len(linhas)}")
print(f"     Regras avaliadas: FIFO, SPT, EDD, CR")
print(f"     Resultados por linha:")
for linha in linhas:
    df_res = resultados_globais[linha]['df_res']
    melhor = df_res.loc[df_res['Atraso_Medio'].idxmin()]['Regra']
    atraso = df_res.loc[df_res['Atraso_Medio'].idxmin()]['Atraso_Medio']
    print(f"       {linha}: melhor regra = {melhor} (atraso médio = {atraso:.1f} dias)")
print(f"{'═'*80}")
