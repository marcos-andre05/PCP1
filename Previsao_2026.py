import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torneio import executar_torneio
from funções.PREVISAO import gerar_previsao
from funções.TRATAMENTO import tratar_anomalias_demanda, analisar_anomalias

# ============================================================
#  PREVISÃO DE DEMANDA — JANEIRO A JULHO DE 2026
#  Usando a técnica vencedora NÃO enviesada de cada linha
# ============================================================

df = pd.read_csv('dataset/trabalho_demanda.csv')
linhas = df.columns[1:].tolist()
meses_historico = df['mes'].tolist()

from funções.UTILS import gerar_meses_futuros, obter_cores_dinamicas
meses_previsao = gerar_meses_futuros(meses_historico[-1], 7)
todos_meses = meses_historico + meses_previsao

print("=" * 70)
print("  📊 PREVISÃO DE DEMANDA — JANEIRO A JULHO DE 2026")
print("=" * 70)

# Armazena resultados de todas as linhas
todas_previsoes = {'Mês': [m.title() for m in meses_previsao]}
resumo_tecnicas = []
dados_grafico = {}

for linha in linhas:
    demandas_orig = df[linha].tolist()

    # --- Análise de anomalias (justificativa técnica do tratamento) ---
    rel = analisar_anomalias(demandas_orig)
    if rel['n_outliers'] > 0:
        meses_out = [m + 1 for m in rel['indices_outliers']]
        print(f"  ⚠️  {linha}: {rel['n_outliers']} outlier(s) no(s) mês(es) {meses_out} "
              f"→ substituídos pela mediana (método IQR)")
    if rel['level_shift']:
        print(f"  ⚠️  {linha}: Level shift detectado no mês {rel['ponto_shift'] + 1} "
              f"→ série ajustada ao novo patamar")
    if rel['n_outliers'] == 0 and not rel['level_shift']:
        print(f"  ✅  {linha}: sem anomalias detectadas — série usada sem alteração")

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
        'MAD': f"{resultado['mad']:.2f}",
        'MAPE (%)': f"{resultado['mape']:.2f}",
        'TS Final': f"{resultado['ts_final']:.2f}"
    })
    dados_grafico[linha] = {
        'real':      demandas,        # série tratada (usada no modelo e no gráfico)
        'real_orig': demandas_orig,    # série bruta (exibida como referência quando houve tratamento)
        'tratado':   rel['n_outliers'] > 0 or rel['level_shift'],
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
df_prev['Total'] = df_prev[linhas].sum(axis=1)

print(f"\n{'─' * 70}")
print("  📅 PREVISÃO CONSOLIDADA — JAN/2026 A JUL/2026")
print(f"{'─' * 70}")
print(df_prev.to_string(index=False))

df_prev.to_csv('previsao_2026_resultados.csv', index=False, encoding='utf-8-sig')
print("\n  Tabela exportada: previsao_2026_resultados.csv")
print(f"\n{'=' * 70}")
print(f"  Total geral previsto (Jan-Jul 2026): {df_prev['Total'].sum():,.0f} unidades")
print(f"{'=' * 70}")

# ============================================================
#  GRÁFICOS INDIVIDUAIS: um por linha de produção
# ============================================================

import matplotlib.colors as mcolors
cores_lista = obter_cores_dinamicas(len(linhas))
cores = {linha: mcolors.to_hex(cores_lista[i]) for i, linha in enumerate(linhas)}
labels_x = [m.title() for m in todos_meses]

print()
for linha in linhas:
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

    # --- série bruta (fundo, apenas quando houve tratamento) ---
    if dados['tratado']:
        ax.plot(x_real, dados['real_orig'],
                color='#AAAAAA', linewidth=1.2, linestyle=':', alpha=0.6,
                label='Original (bruto)', zorder=2)

    # --- dados tratados (série usada no modelo) ---
    ax.plot(x_real, real,
            color='#1A1A2E', marker='o', markersize=5, linewidth=2.2,
            label='Real tratado', zorder=3)

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
    sufixo = '  [série tratada]' if dados['tratado'] else ''
    ax.set_title(f'Linha {linha} — Demanda Real vs Previsão  (Jan/2024 – Jul/2026){sufixo}',
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

fig, axes = plt.subplots(len(linhas), 1, figsize=(13, max(5, 3.2 * len(linhas))), sharex=True)
if len(linhas) == 1: axes = [axes]
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Demanda Real vs Previsão — Jan/2024 a Jul/2026',
             fontsize=14, fontweight='bold', y=0.995, color='#1A1A2E')

for i, linha in enumerate(linhas):
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
    if dados['tratado']:
        ax.plot(x_real, dados['real_orig'],
                color='#AAAAAA', linewidth=1.0, linestyle=':', alpha=0.5,
                label='Original (bruto)', zorder=2)
    ax.plot(x_real, real,
            color='#1A1A2E', marker='o', markersize=4, linewidth=2, label='Real tratado', zorder=3)
    ax.plot(x_prev, y_prev_completa,
            color=cor, marker='s', markersize=4, linewidth=2,
            linestyle='--', alpha=0.85, label=f'Previsão ({dados["tecnica"]})', zorder=3)
    ax.axvspan(n_hist - 0.5, n_hist + n_fut - 0.5, alpha=0.07, color=cor, zorder=1)
    ax.axvline(x=n_hist - 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel(f'Linha {linha}', fontsize=11, fontweight='bold', color=cor)
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
