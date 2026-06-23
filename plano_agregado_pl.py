import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pulp import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from torneio import executar_torneio
from funções.PREVISAO import gerar_previsao
from funções.TRATAMENTO import tratar_anomalias_demanda
from funções.UTILS import gerar_meses_futuros, obter_cores_dinamicas

# ============================================================
#  ETAPA A — PLANO AGREGADO VIA PROGRAMAÇÃO LINEAR (PuLP)
# ============================================================

print(f"{'═'*90}")
print(f"  🧠 PLANO AGREGADO (N1) — OTIMIZAÇÃO VIA PROGRAMAÇÃO LINEAR (PL)")
print(f"{'═'*90}")

# ── 1. Carregar dados ─────────────────────────────────────────
df_param = pd.read_csv('dataset/trabalho_parametros.csv', index_col=0)
df_hist = pd.read_csv('dataset/trabalho_demanda.csv')
linhas = df_hist.columns[1:].tolist()

T_MESES = 7
MESES = [m.title() for m in gerar_meses_futuros(df_hist['mes'].iloc[-1], T_MESES)]

try:
    df_te = pd.read_csv('comparativo_custos_plano_agregado.csv', index_col=0)
except FileNotFoundError:
    df_te = None

# ── 2. Obter Previsões ────────────────────────────────────────
previsoes = {}
for linha in linhas:
    demandas_orig = df_hist[linha].tolist()
    demandas = tratar_anomalias_demanda(demandas_orig)
    resultado = executar_torneio(demandas, n_mms=3, alpha=0.3)
    nome_venc = resultado['vencedora']
    prev_fut = gerar_previsao(nome_venc, demandas, horizonte=T_MESES, n_mms=3, alpha=0.3)
    previsoes[linha] = [round(v) for v in prev_fut]

# ── 3. Modelagem Matemática (PL) ──────────────────────────────
resultados_pl = {}
analise_sensibilidade = []

for linha in linhas:
    p = df_param[linha]
    c_n = float(p['custo_normal'])
    c_e = float(p['custo_extra'])
    c_s = float(p['custo_subcontratacao'])
    c_i = float(p['custo_estoque'])
    
    cap_n = float(p['capacidade_normal'])
    cap_e = float(p['capacidade_extra'])
    cap_s = float(p['subcontratacao_max'])
    
    est_inicial = float(p['estoque_inicial'])
    est_minimo = float(p['estoque_minimo'])
    
    D = previsoes[linha]
    
    # Criar problema de minimização
    prob = LpProblem(f"Plano_Agregado_{linha}", LpMinimize)
    
    # Variáveis de Decisão (sem limites superiores explícitos para extrair Preço Sombra das restrições)
    X = LpVariable.dicts(f"X", range(T_MESES), 0, None, LpContinuous)
    O = LpVariable.dicts(f"O", range(T_MESES), 0, None, LpContinuous)
    S = LpVariable.dicts(f"S", range(T_MESES), 0, None, LpContinuous)
    I = LpVariable.dicts(f"I", range(T_MESES), est_minimo, None, LpContinuous)
    
    # Função Objetivo: Minimizar Custos
    prob += lpSum([c_n * X[t] + c_e * O[t] + c_s * S[t] + c_i * I[t] for t in range(T_MESES)]), "Custo_Total"
    
    # Restrições
    for t in range(T_MESES):
        # 1. Balanço de Estoque: It = I(t-1) + Xt + Ot + St - Dt
        if t == 0:
            prob += I[t] == est_inicial + X[t] + O[t] + S[t] - D[t], f"Balanco_Estoque_{t}"
        else:
            prob += I[t] == I[t-1] + X[t] + O[t] + S[t] - D[t], f"Balanco_Estoque_{t}"
            
        # 2. Capacidades (Adicionadas explicitamente para extração de Shadow Price)
        prob += X[t] <= cap_n, f"Cap_Normal_{t}"
        prob += O[t] <= cap_e, f"Cap_Extra_{t}"
        prob += S[t] <= cap_s, f"Cap_Sub_{t}"

    # Resolver
    prob.solve(PULP_CBC_CMD(msg=0))
    
    status = LpStatus[prob.status]
    custo_otimo = value(prob.objective)
    
    # Analisar Sensibilidade (Shadow Prices)
    for name, c in prob.constraints.items():
        if "Cap_" in name:
            mes_idx = int(name.split('_')[-1])
            tipo_cap = name.split('_')[1] # Normal, Extra, Sub
            shadow_price = c.pi
            slack = c.slack
            if shadow_price < 0 or shadow_price > 0: # Preço sombra não nulo
                analise_sensibilidade.append({
                    'Linha': linha,
                    'Mês': MESES[mes_idx],
                    'Restrição': name,
                    'Shadow Price (R$)': shadow_price,
                    'Folga (Slack)': slack
                })
    
    resultados_pl[linha] = {
        'Status': status,
        'Custo_PL': custo_otimo,
        'X': [X[t].varValue for t in range(T_MESES)],
        'O': [O[t].varValue for t in range(T_MESES)],
        'S': [S[t].varValue for t in range(T_MESES)],
        'I': [I[t].varValue for t in range(T_MESES)]
    }

    print(f"\n  🏭 {linha} — Status Solver: {status}")
    print(f"     Custo Ótimo (PL): R$ {custo_otimo:,.2f}")

# ── 4. Comparativo T&E vs PL ──────────────────────────────────
print(f"\n{'═'*90}")
print(f"  📊 COMPARATIVO — TENTATIVA E ERRO (T&E) VS PROGRAMAÇÃO LINEAR (PL)")
print(f"{'═'*90}")

dados_comp = []
total_te = 0
total_pl = 0

for linha in linhas:
    custo_pl = resultados_pl[linha]['Custo_PL']
    
    if df_te is not None and linha in df_te.index:
        melhor_te = df_te.loc[linha]
        estrategia_te = melhor_te.get('Melhor', 'N/A')
        # A coluna com o custo varia de nome. Se não acharmos, calculamos ou tentamos ler.
        # Mas df_te não tem o custo na coluna 'Melhor', tem nas colunas Chase/Level/Mista.
        try:
            str_val = melhor_te[estrategia_te.title()]
            custo_te = float(str_val.replace('R$ ', '').replace(',', ''))
        except:
            # Fallback se não conseguir parsear
            custo_te = custo_pl 
    else:
        estrategia_te = "N/A"
        custo_te = custo_pl

    economia = custo_te - custo_pl
    economia_pct = (economia / custo_te) * 100 if custo_te > 0 else 0
    
    total_te += custo_te
    total_pl += custo_pl

    dados_comp.append({
        'Linha': linha,
        'Melhor T&E': estrategia_te,
        'Custo T&E (R$)': custo_te,
        'Custo PL (R$)': custo_pl,
        'Economia (R$)': economia,
        'Redução (%)': economia_pct
    })
    
    print(f"  {linha: <4} | T&E: R$ {custo_te:>12,.2f} | PL: R$ {custo_pl:>12,.2f} | Economia: R$ {economia:>10,.2f} ({economia_pct:.1f}%)")

print(f"  {'-'*86}")
print(f"  TOTAL | T&E: R$ {total_te:>12,.2f} | PL: R$ {total_pl:>12,.2f} | Economia: R$ {(total_te - total_pl):>10,.2f}")

df_comparativo = pd.DataFrame(dados_comp)
df_comparativo.to_csv('comparativo_pl_vs_te.csv', index=False)

# ── 5. Análise de Sensibilidade (Exportação) ──────────────────
if analise_sensibilidade:
    df_sens = pd.DataFrame(analise_sensibilidade)
    df_sens.to_csv('analise_sensibilidade_pl.csv', index=False)
    print(f"\n  📁 Análise de Sensibilidade exportada: analise_sensibilidade_pl.csv")
    print(f"     (Mostra o impacto no custo total para cada unidade adicional de capacidade gargalo)")
else:
    print(f"\n  📁 Sem restrições de capacidade ativas (Shadow Price = 0 para todas as capacidades).")

# ── 6. Gráfico Comparativo ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

x = np.arange(len(linhas))
w = 0.35

custos_te = [d['Custo T&E (R$)'] for d in dados_comp]
custos_pl = [d['Custo PL (R$)'] for d in dados_comp]

rects1 = ax.bar(x - w/2, custos_te, w, label='T&E (Melhor Heurística)', color='#78909C')
rects2 = ax.bar(x + w/2, custos_pl, w, label='PL (Ótimo Matemático)', color='#2E7D32')

ax.set_ylabel('Custo Total (R$)', fontsize=11)
ax.set_title('Comparativo de Custos: Tentativa e Erro vs Programação Linear', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(linhas)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'R$ {v:,.0f}'))
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.2, axis='y')

# Rotulos
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h/1e3:.0f}k',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),  textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('comparativo_pl_grafico.png', dpi=120)
print(f"  📊 Gráfico salvo: comparativo_pl_grafico.png")

print(f"\n{'═'*90}")
print(f"  ✅  Otimização Concluída!")
print(f"{'═'*90}")
