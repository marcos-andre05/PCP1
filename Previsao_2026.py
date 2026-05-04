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
#  GRÁFICO: REAL vs PREVISTO
# ============================================================

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
fig.suptitle('Demanda Real vs Previsão — Jan/2024 a Jul/2026', fontsize=14, fontweight='bold', y=0.98)

cores = {'L1': '#2196F3', 'L2': '#4CAF50', 'L3': '#FF9800', 'L4': '#9C27B0', 'L5': '#F44336'}

for i, linha in enumerate(['L1', 'L2', 'L3', 'L4', 'L5']):
    ax = axes[i]
    dados = dados_grafico[linha]
    real = dados['real']
    prev_hist = dados['previsao_historica']
    prev_fut = dados['previsao_futura']
    cor = cores[linha]
    
    n_hist = len(real)
    n_fut = len(prev_fut)
    
    x_real = list(range(n_hist))
    x_prev = list(range(n_hist + n_fut))
    y_prev_completa = prev_hist + prev_fut
    
    # Definindo cores diferentes para Real e Previsão
    cor_real = 'black'  # Cor neutra/escura para a demanda real
    cor_prev = cor      # Cor vibrante específica da linha para a previsão
    
    # Linha real
    ax.plot(x_real, real, color=cor_real, marker='o', markersize=4, linewidth=2, label='Real (histórico)')
    
    # Linha de previsão
    ax.plot(x_prev, y_prev_completa, color=cor_prev, marker='s', markersize=4, linewidth=2,
            linestyle='--', alpha=0.8, label=f'Previsão ({dados["tecnica"]})')
    
    # Área sombreada na zona de previsão
    ax.axvspan(n_hist - 0.5, n_hist + 6.5, alpha=0.08, color=cor)
    
    # Linha divisória
    ax.axvline(x=n_hist - 0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_ylabel(f'Linha {linha[-1]}', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))

# Configurar eixo X com nomes dos meses
labels_x = [m.title() for m in todos_meses]
axes[-1].set_xticks(range(len(labels_x)))
axes[-1].set_xticklabels(labels_x, rotation=45, ha='right', fontsize=8)
axes[-1].set_xlabel('Mês', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('previsao_2026_real_vs_previsto.png', dpi=100, bbox_inches='tight')
plt.show()
print("\n📊 Gráfico salvo em: previsao_2026_real_vs_previsto.png")
