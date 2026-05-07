import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from torneio import executar_torneio
from funções.PREVISAO import gerar_previsao
from funções.TRATAMENTO import tratar_anomalias_demanda, analisar_anomalias

# ============================================================
#  PLANO AGREGADO — MONTAGEM (T&E)  —  Jan/2026 a Jul/2026
#  Estratégias: Chase | Level | Mista
#  Custo: Σ(Cn·Xt + Ce·Ot + Cs·St + Ci·It)
# ============================================================

MESES = ['Jan/26', 'Fev/26', 'Mar/26', 'Abr/26', 'Mai/26', 'Jun/26', 'Jul/26']
T = len(MESES)

# ── Parâmetros e histórico ────────────────────────────────────
df_param = pd.read_csv('dataset/trabalho_parametros.csv', index_col=0)
df_hist  = pd.read_csv('dataset/trabalho_demanda.csv')
linhas   = ['L1', 'L2', 'L3', 'L4', 'L5']

# Override manual por linha — espelha a mesma decisão de Previsao_2026.py
# L4: Holt-Winters preferido ao Holt Duplo (que gera projeção linear)
FORCAS_TECNICA = {'L4': 'Holt-Winters'}

# ── Gerar previsões via torneio ───────────────────────────────
previsoes = {}
tecnicas  = {}
for linha in linhas:
    demandas_orig = df_hist[linha].tolist()

    # Análise de anomalias (justificativa técnica)
    rel = analisar_anomalias(demandas_orig)
    if rel['n_outliers'] > 0:
        meses_out = [m + 1 for m in rel['indices_outliers']]
        print(f"  ⚠️  {linha}: {rel['n_outliers']} outlier(s) no(s) mês(es) {meses_out} → tratados (IQR)")
    if rel['level_shift']:
        print(f"  ⚠️  {linha}: level shift no mês {rel['ponto_shift'] + 1} → série ajustada")

    demandas      = tratar_anomalias_demanda(demandas_orig)
    resultado     = executar_torneio(demandas, n_mms=3, alpha=0.3)

    # Override manual (se configurado para esta linha)
    if linha in FORCAS_TECNICA:
        nome_venc = FORCAS_TECNICA[linha]
    else:
        nome_venc = resultado['vencedora']

    prev_fut         = gerar_previsao(nome_venc, demandas, horizonte=T, n_mms=3, alpha=0.3)
    previsoes[linha] = [round(v) for v in prev_fut]
    tecnicas[linha]  = nome_venc

# ============================================================
#  FUNÇÕES DAS ESTRATÉGIAS
# ============================================================

def _params(linha):
    """Retorna dicionário de parâmetros da linha."""
    p = df_param[linha]
    return {
        'cap_n':   float(p['capacidade_normal']),
        'cap_e':   float(p['capacidade_extra']),
        'sub_max': float(p['subcontratacao_max']),
        'est_ini': float(p['estoque_inicial']),
        'est_min': float(p['estoque_minimo']),
        'cn': float(p['custo_normal']),
        'ce': float(p['custo_extra']),
        'cs': float(p['custo_subcontratacao']),
        'ch': float(p['custo_estoque']),   # ch = Ci (custo de manutenção de estoque/inventário)
        'produto': p['produtos'],
    }

def _simular(D, cap_n, cap_e, sub_max, est_ini, est_min,
             cn, ce, cs, ch, X_fixo=None, P_base=None):
    """
    Simula um período. Se X_fixo[t] é dado, usa como produção normal (Level/Mista).
    Se X_fixo é None (Chase), calcula X_t = min(need, cap_n).
    Horas extras e subcontratação completam quando X < necessidade.
    Déficits reais (capacidade insuficiente para atender D + est_min) são registrados
    separadamente em DEF_out e não são mascarados.
    """
    X_out, O_out, S_out, E_out, EI_out, DEF_out = [], [], [], [], [], []
    ei = est_ini

    for t in range(T):
        EI_out.append(ei)
        need = max(0.0, D[t] - ei + est_min)   # quantidade a produzir para atingir est_min

        # Produção normal
        if X_fixo is not None:
            xt = float(X_fixo[t])               # Level / Mista: valor pré-calculado
        else:
            xt = min(need, cap_n)               # Chase: rastreia demanda

        # Horas extras e subcontratação
        restante = max(0.0, need - xt)
        ot = min(restante, cap_e)
        restante -= ot
        st = min(restante, sub_max)

        # Equação de balanço: EF = EI + X + O + S − D
        ef_natural = ei + xt + ot + st - D[t]
        # Déficit: quanto falta para atingir est_min com os recursos disponíveis
        deficit_t  = max(0.0, est_min - ef_natural)
        # EF respeita est_min como restrição (déficit registrado separadamente)
        ef = max(ef_natural, est_min)

        X_out.append(xt); O_out.append(ot); S_out.append(st)
        E_out.append(ef); DEF_out.append(deficit_t)
        ei = ef

    # Custo total: Σ(Cn·X + Ce·O + Cs·S + Ci·I)  [ch = Ci = custo de inventário]
    custo = sum(X_out[t]*cn + O_out[t]*ce + S_out[t]*cs + E_out[t]*ch
                for t in range(T))
    return X_out, O_out, S_out, E_out, EI_out, custo, DEF_out


def estrategia_chase(linha):
    """
    Chase: a produção normal acompanha a demanda a cada período.
    Usa X_t = min(necessidade, Cap_n) e completa com hora extra / subcontratação.
    Estoque é mantido no mínimo possível (= Est_min).
    """
    p = _params(linha)
    D = previsoes[linha]
    X, O, S, E, EI, custo, DEF = _simular(D, **{k: p[k] for k in
        ['cap_n','cap_e','sub_max','est_ini','est_min','cn','ce','cs','ch']})
    return dict(X=X, O=O, S=S, E=E, EI=EI, D=D, DEF=DEF, custo_total=custo, **p)


def estrategia_level(linha):
    """
    Level: produção normal constante = Cap_n em todos os períodos.
    O estoque absorve variações. Horas extras e subcontratação são usadas
    somente quando o estoque não consegue evitar que EF < Est_min.
    """
    p = _params(linha)
    D = previsoes[linha]
    P = p['cap_n']              # produção normal constante = capacidade total normal
    X_fixo = [P] * T
    X, O, S, E, EI, custo, DEF = _simular(D, **{k: p[k] for k in
        ['cap_n','cap_e','sub_max','est_ini','est_min','cn','ce','cs','ch']},
        X_fixo=X_fixo)
    return dict(X=X, O=O, S=S, E=E, EI=EI, D=D, DEF=DEF, custo_total=custo,
                P_constante=P, **p)


def estrategia_mista(linha):
    """
    Mista: base constante de produção normal = média das demandas previstas
    (componente Level), acrescida de hora extra / subcontratação apenas quando
    a produção base + estoque existente não cobrem a demanda (componente Chase).
    Combina estabilidade operacional com reatividade aos picos.
    """
    p = _params(linha)
    D = previsoes[linha]
    P_base = min(p['cap_n'], round(float(np.mean(D))))
    X_fixo = [float(P_base)] * T
    X, O, S, E, EI, custo, DEF = _simular(D, **{k: p[k] for k in
        ['cap_n','cap_e','sub_max','est_ini','est_min','cn','ce','cs','ch']},
        X_fixo=X_fixo)
    return dict(X=X, O=O, S=S, E=E, EI=EI, D=D, DEF=DEF, custo_total=custo,
                P_base=P_base, **p)


# ============================================================
#  EXECUÇÃO
# ============================================================

estrategias = {
    'Chase': estrategia_chase,
    'Level': estrategia_level,
    'Mista': estrategia_mista,
}

# resultados[linha][estrategia] = dict
resultados = {l: {} for l in linhas}
for linha in linhas:
    for nome, func in estrategias.items():
        resultados[linha][nome] = func(linha)


# ============================================================
#  IMPRESSÃO — 3 PLANILHAS (uma por estratégia)
# ============================================================

def imprimir_planilha(nome_estrategia, linha_dados):
    print(f"\n{'═'*76}")
    print(f"  📋 ESTRATÉGIA: {nome_estrategia.upper()}")
    print(f"{'═'*76}")

    for linha in linhas:
        r = linha_dados[linha]
        print(f"\n  🏭 Linha {linha} — {r['produto']}  |  Técnica previsão: {tecnicas[linha]}")

        # Info extra de cada estratégia
        if 'P_constante' in r:
            print(f"  Produção normal constante: {int(r['P_constante']):,} un/mês")
        if 'P_base' in r:
            print(f"  Base constante (média demanda): {int(r['P_base']):,} un/mês")
        print(f"  Cap.Normal={int(r['cap_n']):,} | Cap.Extra={int(r['cap_e']):,} "
              f"| Sub.Máx={int(r['sub_max']):,} | Est.Mín={int(r['est_min']):,}")
        print(f"  {'─'*70}")

        rows = []
        for t in range(T):
            custo_t = (r['X'][t]*r['cn'] + r['O'][t]*r['ce'] +
                       r['S'][t]*r['cs'] + r['E'][t]*r['ch'])
            def_t   = int(round(r['DEF'][t]))
            rows.append({
                'Mês':          MESES[t],
                'Demanda':      int(round(r['D'][t])),
                'EI':           int(round(r['EI'][t])),
                'Prod.Normal':  int(round(r['X'][t])),
                'H.Extra':      int(round(r['O'][t])),
                'Subcontr.':    int(round(r['S'][t])),
                'EF':           int(round(r['E'][t])),
                'Déficit':      def_t if def_t > 0 else '-',
                'Custo(R$)':    f"R$ {custo_t:,.2f}"
            })

        df_pl = pd.DataFrame(rows)
        print(df_pl.to_string(index=False))
        print(f"  💰 Custo total {linha}: R$ {r['custo_total']:,.2f}")

for nome in estrategias:
    imprimir_planilha(nome, {l: resultados[l][nome] for l in linhas})


# ============================================================
#  TABELA COMPARATIVA DE CUSTOS + ESCOLHA DA MELHOR
# ============================================================

print(f"\n\n{'═'*76}")
print("  📊 TABELA COMPARATIVA — CUSTO TOTAL POR LINHA E ESTRATÉGIA")
print(f"{'═'*76}")

header = f"  {'Linha':<6} {'Produto':<26} {'Chase':>15} {'Level':>15} {'Mista':>15}  {'Melhor':<8}"
print(header)
print(f"  {'─'*70}")

totais = {'Chase': 0, 'Level': 0, 'Mista': 0}
vencedoras_linha = {}
linhas_csv_custos = []

for linha in linhas:
    custos_linha = {e: resultados[linha][e]['custo_total'] for e in estrategias}
    melhor = min(custos_linha, key=custos_linha.get)
    vencedoras_linha[linha] = melhor
    produto = df_param[linha]['produtos']

    c_chase = custos_linha['Chase']
    c_level = custos_linha['Level']
    c_mista = custos_linha['Mista']
    
    linhas_csv_custos.append({
        'Linha': linha,
        'Produto': produto,
        'Custo Chase (R$)': c_chase,
        'Custo Level (R$)': c_level,
        'Custo Mista (R$)': c_mista,
        'Melhor Estratégia': melhor
    })

    def fmt(v, melhor_v):
        s = f"R$ {v:,.0f}"
        return f"★ {s}" if v == melhor_v else f"  {s}"

    best_val = min(c_chase, c_level, c_mista)
    print(f"  {linha:<6} {produto:<26} {fmt(c_chase, best_val):>17} "
          f"{fmt(c_level, best_val):>17} {fmt(c_mista, best_val):>17}  {melhor}")

    for e in estrategias:
        totais[e] += custos_linha[e]

df_comparativo_custos = pd.DataFrame(linhas_csv_custos)
df_comparativo_custos.to_csv('comparativo_custos_plano_agregado.csv', index=False, encoding='utf-8-sig')
print("\n  Tabela exportada: comparativo_custos_plano_agregado.csv")

print(f"  {'─'*70}")
melhor_global = min(totais, key=totais.get)
t_chase = f"R$ {totais['Chase']:,.0f}"
t_level = f"R$ {totais['Level']:,.0f}"
t_mista = f"R$ {totais['Mista']:,.0f}"
print(f"  {'TOTAL GERAL':<32} {t_chase:>17} {t_level:>17} {t_mista:>17}")
print(f"{'═'*76}")

print(f"\n  🏆 ESTRATÉGIA GLOBAL VENCEDORA: {melhor_global.upper()}")
print(f"     Custo total: R$ {totais[melhor_global]:,.2f}")

economia = {e: totais[e] - totais[melhor_global] for e in estrategias if e != melhor_global}
for e, eco in economia.items():
    print(f"     Economia vs {e}: R$ {eco:,.2f} ({eco/totais[e]*100:.1f}%)")

print(f"\n  Justificativa por linha:")
for linha, venc in vencedoras_linha.items():
    custos_linha = {e: resultados[linha][e]['custo_total'] for e in estrategias}
    outros = [e for e in estrategias if e != venc]
    eco_txt = " | ".join(
        f"−R$ {custos_linha[e]-custos_linha[venc]:,.0f} vs {e}"
        for e in outros
    )
    print(f"     {linha}: {venc} ({eco_txt})")

print(f"{'═'*76}")


# ============================================================
#  GRÁFICOS
# ============================================================

cores_est  = {'Chase': '#2196F3', 'Level': '#4CAF50', 'Mista': '#FF9800'}
cores_lin  = {'L1': '#1565C0', 'L2': '#2E7D32', 'L3': '#E65100',
              'L4': '#6A1B9A', 'L5': '#B71C1C'}

x_idx = list(range(T))

# ── Gráfico 1: 5 linhas × 3 estratégias — composição de custo (stacked) ──
print()
for linha in linhas:
    produto = df_param[linha]['produtos']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle(f'Plano Agregado — Linha {linha} ({produto})  |  '
                 f'Comparação de Estratégias',
                 fontsize=13, fontweight='bold', color='#1A1A2E', y=1.02)

    for ax, nome in zip(axes, estrategias):
        r   = resultados[linha][nome]
        cor = cores_est[nome]
        ax.set_facecolor('#FFFFFF')

        # barras empilhadas
        ax.bar(x_idx, r['X'], label='Prod. Normal', color=cor, alpha=0.85, zorder=2)
        ax.bar(x_idx, r['O'], bottom=r['X'], label='Hora Extra',
               color='#FFC107', alpha=0.85, zorder=2)
        bot_s = [r['X'][t] + r['O'][t] for t in range(T)]
        ax.bar(x_idx, r['S'], bottom=bot_s, label='Subcontratação',
               color='#78909C', alpha=0.85, zorder=2)

        # linha de demanda
        ax.plot(x_idx, r['D'], color='#1A1A2E', marker='o', markersize=5,
                linewidth=2, linestyle='--', label='Demanda', zorder=3)

        ax.set_title(f'{nome}\nR$ {r["custo_total"]:,.0f}',
                     fontsize=11, fontweight='bold', color=cor)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(MESES, rotation=40, ha='right', fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
        ax.grid(True, alpha=0.2, linestyle='--', axis='y')
        for sp in ax.spines.values():
            sp.set_edgecolor('#CCCCCC')

    plt.tight_layout()
    nome_arq = f'plano_agregado_{linha}.png'
    plt.savefig(nome_arq, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  📊 Gráfico {linha} salvo: {nome_arq}")


# ── Gráfico 2: Comparativo de custo total (barras agrupadas) ─
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

n_linhas = len(linhas)
n_est    = len(estrategias)
w        = 0.25          # largura de cada barra
offsets  = [-w, 0, w]    # Chase, Level, Mista

for i, (nome, off, cor) in enumerate(zip(estrategias, offsets, cores_est.values())):
    vals = [resultados[l][nome]['custo_total'] for l in linhas]
    pos  = [j + off for j in range(n_linhas)]
    bars = ax.bar(pos, vals, width=w*0.9, color=cor, alpha=0.85, label=nome, zorder=2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.005,
                f'{v/1e3:.0f}k', ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color=cor)

# marca a barra vencedora com ★
for j, linha in enumerate(linhas):
    custos_linha = {e: resultados[linha][e]['custo_total'] for e in estrategias}
    melhor = min(custos_linha, key=custos_linha.get)
    idx_m  = list(estrategias.keys()).index(melhor)
    vmax   = custos_linha[melhor]
    xpos   = j + offsets[idx_m]
    ax.text(xpos, vmax + max(custos_linha.values())*0.04, '★',
            ha='center', va='bottom', fontsize=13, color='gold')

ax.set_xticks(range(n_linhas))
ax.set_xticklabels(
    [f"L{i+1}\n{df_param[f'L{i+1}']['produtos'].split()[0]}" for i in range(n_linhas)],
    fontsize=9
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'R$ {v:,.0f}'))
ax.set_title('Comparativo de Custo Total — Chase × Level × Mista\n'
             '★ = estratégia de menor custo por linha  (Jan/2026 – Jul/2026)',
             fontsize=13, fontweight='bold', color='#1A1A2E', pad=12)
ax.set_xlabel('Linha de Produção', fontsize=11)
ax.set_ylabel('Custo Total (R$)', fontsize=11)
ax.legend(fontsize=10, loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.25, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

plt.tight_layout()
plt.savefig('plano_agregado_comparativo.png', dpi=120, bbox_inches='tight')
plt.show()
print("\n📊 Gráfico comparativo salvo: plano_agregado_comparativo.png")


# ── Gráfico 3: Custo total por estratégia (pizza share) ──────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Distribuição de Custo por Linha — Cada Estratégia\n(Jan/2026 – Jul/2026)',
             fontsize=13, fontweight='bold', color='#1A1A2E')

pie_colors = [cores_lin[l] for l in linhas]
for ax, (nome, cor) in zip(axes, cores_est.items()):
    vals   = [resultados[l][nome]['custo_total'] for l in linhas]
    wedges, texts, autotexts = ax.pie(
        vals, labels=linhas, colors=pie_colors,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax.set_title(f'{nome}\nTotal: R$ {sum(vals):,.0f}',
                 fontsize=11, fontweight='bold', color=cor)

plt.tight_layout()
plt.savefig('plano_agregado_distribuicao.png', dpi=120, bbox_inches='tight')
plt.show()
print("📊 Gráfico distribuição salvo: plano_agregado_distribuicao.png")

print(f"\n{'═'*76}")
print(f"  ✅  Plano Agregado concluído.")
print(f"     Estratégia vencedora global: {melhor_global.upper()}")
print(f"     Custo total: R$ {totais[melhor_global]:,.2f}")
print(f"{'═'*76}")
