import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torneio import executar_torneio
from funções.PREVISAO import gerar_previsao
from funções.TRATAMENTO import tratar_anomalias_demanda

# ============================================================
#  PREVISÃO DE DEMANDA — JANEIRO A JULHO DE 2026
#  Usando a técnica vencedora NÃO enviesada de cada linha
# ============================================================

df = pd.read_csv('dataset/trabalho_demanda.csv')
meses_historico = df['mes'].tolist()
meses_previsao = ['jan/26', 'fev/26', 'mar/26', 'abr/26', 'mai/26', 'jun/26', 'jul/26']
todos_meses = meses_historico + meses_previsao

print("=" * 70)
print("  📊 PREVISÃO DE DEMANDA — JANEIRO A JULHO DE 2026")
print("=" * 70)

# Armazena resultados de todas as linhas
todas_previsoes = {'Mês': [m.title() for m in meses_previsao]}
resumo_tecnicas = []
dados_grafico = {}

for linha in ['L1', 'L2', 'L3', 'L4', 'L5']:
    demandas_orig = df[linha].tolist()
    demandas = tratar_anomalias_demanda(demandas_orig)
    
    # Torneio + seleção da técnica sem viés
    resultado = executar_torneio(demandas, n_mms=3, alpha=0.3)
    nome_vencedora = resultado['vencedora']
    
    # Previsão de 7 meses
    previsoes_futuras = gerar_previsao(nome_vencedora, demandas, horizonte=7, n_mms=3, alpha=0.3)
    
    todas_previsoes[linha] = previsoes_futuras
    resumo_tecnicas.append({
        'Linha': linha,
        'Técnica Vencedora': nome_vencedora,
        'MAPE (%)': f"{resultado['mape']:.2f}",
        'TS Final': f"{resultado['ts_final']:.2f}"
    })
    dados_grafico[linha] = {
        'real': demandas_orig,
        'previsao_historica': resultado['previsoes'][nome_vencedora],
        'previsao_futura': previsoes_futuras,
        'tecnica': nome_vencedora
    }

# ============================================================
#  EXIBIÇÃO DOS RESULTADOS
# ============================================================

print(f"\n{'─' * 70}")
print("  🏆 TÉCNICAS VENCEDORAS POR LINHA (sem viés)")
print(f"{'─' * 70}")
print(pd.DataFrame(resumo_tecnicas).to_string(index=False))

df_prev = pd.DataFrame(todas_previsoes)
df_prev['Total'] = df_prev[['L1', 'L2', 'L3', 'L4', 'L5']].sum(axis=1)

print(f"\n{'─' * 70}")
print("  📅 PREVISÃO CONSOLIDADA — JAN/2026 A JUL/2026")
print(f"{'─' * 70}")
print(df_prev.to_string(index=False))

print(f"\n{'=' * 70}")
print(f"  Total geral previsto (Jan-Jul 2026): {df_prev['Total'].sum():,.0f} unidades")
print(f"{'=' * 70}")

# ============================================================
#  GRÁFICOS INDIVIDUAIS: um por linha de produção
# ============================================================

cores = {'L1': '#2196F3', 'L2': '#4CAF50', 'L3': '#FF9800', 'L4': '#9C27B0', 'L5': '#F44336'}
labels_x = [m.title() for m in todos_meses]

print()
for linha in ['L1', 'L2', 'L3', 'L4', 'L5']:
    dados   = dados_grafico[linha]
    real     = dados['real']
    prev_hist = dados['previsao_historica']
    prev_fut  = dados['previsao_futura']
    cor       = cores[linha]

    n_hist = len(real)
    n_fut  = len(prev_fut)

    x_real = list(range(n_hist))
    x_prev = list(range(n_hist + n_fut))
    y_prev_completa = prev_hist + prev_fut

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')

    # --- dados reais ---
    ax.plot(x_real, real,
            color='#1A1A2E', marker='o', markersize=5, linewidth=2.2,
            label='Real (histórico)', zorder=3)

    # --- linha de previsão (histórica + futura) ---
    ax.plot(x_prev, y_prev_completa,
            color=cor, marker='s', markersize=5, linewidth=2.2,
            linestyle='--', alpha=0.9,
            label=f'Previsão ({dados["tecnica"]})', zorder=3)

    # --- área sombreada da zona de previsão ---
    ax.axvspan(n_hist - 0.5, n_hist + n_fut - 0.5, alpha=0.07, color=cor, zorder=1)

    # --- linha divisória histórico / previsão ---
    ax.axvline(x=n_hist - 0.5, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.text(n_hist - 0.4, ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
            '◀ Histórico  Previsão ▶', fontsize=7.5, color='gray', va='bottom')

    # --- eixo X ---
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=8.5)

    # --- formatação ---
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_title(f'Linha {linha} — Demanda Real vs Previsão  (Jan/2024 – Jul/2026)',
                 fontsize=13, fontweight='bold', pad=12, color='#1A1A2E')
    ax.set_xlabel('Mês', fontsize=10)
    ax.set_ylabel('Demanda (unidades)', fontsize=10)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.85)
    ax.grid(True, alpha=0.25, linestyle='--')

    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')

    plt.tight_layout()

    nome_arquivo = f'grafico_{linha}.png'
    plt.savefig(nome_arquivo, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  📊 Gráfico da {linha} salvo em: {nome_arquivo}")

# ============================================================
#  GRÁFICO CONSOLIDADO (todas as linhas juntas)
# ============================================================

fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Demanda Real vs Previsão — Jan/2024 a Jul/2026',
             fontsize=14, fontweight='bold', y=0.995, color='#1A1A2E')

for i, linha in enumerate(['L1', 'L2', 'L3', 'L4', 'L5']):
    ax    = axes[i]
    dados = dados_grafico[linha]
    real     = dados['real']
    prev_hist = dados['previsao_historica']
    prev_fut  = dados['previsao_futura']
    cor       = cores[linha]

    n_hist = len(real)
    n_fut  = len(prev_fut)

    x_real = list(range(n_hist))
    x_prev = list(range(n_hist + n_fut))
    y_prev_completa = prev_hist + prev_fut

    ax.set_facecolor('#FFFFFF')
    ax.plot(x_real, real,
            color='#1A1A2E', marker='o', markersize=4, linewidth=2, label='Real (histórico)', zorder=3)
    ax.plot(x_prev, y_prev_completa,
            color=cor, marker='s', markersize=4, linewidth=2,
            linestyle='--', alpha=0.85, label=f'Previsão ({dados["tecnica"]})', zorder=3)
    ax.axvspan(n_hist - 0.5, n_hist + n_fut - 0.5, alpha=0.07, color=cor, zorder=1)
    ax.axvline(x=n_hist - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel(f'Linha {linha[-1]}', fontsize=11, fontweight='bold', color=cor)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.8)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    for spine in ax.spines.values():
        spine.set_edgecolor('#CCCCCC')

axes[-1].set_xticks(range(len(labels_x)))
axes[-1].set_xticklabels(labels_x, rotation=45, ha='right', fontsize=8)
axes[-1].set_xlabel('Mês', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.993])
plt.savefig('previsao_2026_consolidado.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n📊 Gráfico consolidado salvo em: previsao_2026_consolidado.png")
