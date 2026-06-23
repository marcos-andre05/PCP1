import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy import stats

from funções.UTILS import obter_cores_dinamicas

# ============================================================
#  GESTÃO DE ESTOQUE — Classificação ABC, LEC, PP, ES
#  Aplicado às 5 linhas de produção (L1 a L5)
# ============================================================

# ── Carregar dados ────────────────────────────────────────────
df_param = pd.read_csv('dataset/trabalho_parametros.csv', index_col=0)
linhas   = [c for c in df_param.columns if c.startswith('L')]

# Cores dinâmicas para as linhas
cores_lista = obter_cores_dinamicas(len(linhas))
cores = {linha: mcolors.to_hex(cores_lista[i]) for i, linha in enumerate(linhas)}

print(f"{'═'*80}")
print(f"  📦 GESTÃO DE ESTOQUE — Classificação ABC, LEC, PP e Estoque de Segurança")
print(f"{'═'*80}")


# ============================================================
#  1. CLASSIFICAÇÃO ABC — POR LINHA
# ============================================================

def classificar_abc(df_abc):
    """
    Classifica itens pelo método ABC (Curva de Pareto).
    A: ~80% do valor acumulado
    B: ~80%–95%
    C: ~95%–100%
    """
    df = df_abc.copy()
    df = df.sort_values('valor_anual', ascending=False).reset_index(drop=True)
    df['valor_acumulado'] = df['valor_anual'].cumsum()
    df['perc_valor'] = df['valor_anual'] / df['valor_anual'].sum() * 100
    df['perc_acumulado'] = df['valor_acumulado'] / df['valor_anual'].sum() * 100

    # Classificação
    classes = []
    for perc in df['perc_acumulado']:
        if perc <= 80:
            classes.append('A')
        elif perc <= 95:
            classes.append('B')
        else:
            classes.append('C')
    df['classe'] = classes

    return df


resultados_abc = {}

print(f"\n{'─'*80}")
print(f"  📊 CLASSIFICAÇÃO ABC POR LINHA DE PRODUÇÃO")
print(f"{'─'*80}")

for linha in linhas:
    produto = df_param[linha]['produtos']
    arquivo_abc = f'dataset/trabalho_abc_{linha.lower()}.csv'

    try:
        df_abc = pd.read_csv(arquivo_abc)
    except FileNotFoundError:
        print(f"  ⚠️  Arquivo {arquivo_abc} não encontrado — pulando {linha}")
        continue

    df_class = classificar_abc(df_abc)
    resultados_abc[linha] = df_class

    print(f"\n  🏭 Linha {linha} — {produto}")
    print(f"  {'─'*74}")

    # Exibir tabela formatada
    df_exib = df_class[['item', 'preco_unitario', 'demanda_anual',
                         'valor_anual', 'perc_valor', 'perc_acumulado', 'classe']].copy()
    df_exib.columns = ['Item', 'Preço Unit.', 'Dem. Anual', 'Valor Anual (R$)',
                        '% Valor', '% Acumulado', 'Classe']
    print(df_exib.to_string(index=False, float_format='{:,.1f}'.format))

    # Resumo por classe
    for cls in ['A', 'B', 'C']:
        itens_cls = df_class[df_class['classe'] == cls]
        n = len(itens_cls)
        val = itens_cls['valor_anual'].sum()
        perc_itens = n / len(df_class) * 100
        perc_valor = val / df_class['valor_anual'].sum() * 100
        print(f"     Classe {cls}: {n} itens ({perc_itens:.0f}%) → "
              f"R$ {val:,.0f} ({perc_valor:.1f}% do valor)")


# ============================================================
#  2. ESTOQUE DE SEGURANÇA (ES)
# ============================================================

print(f"\n\n{'═'*80}")
print(f"  🛡️  ESTOQUE DE SEGURANÇA — POR LINHA")
print(f"{'═'*80}")
print(f"  Fórmula: ES = Z × σ_d × √(LT)")
print(f"  Z = valor da distribuição normal para o nível de serviço desejado")
print(f"  σ_d = desvio-padrão da demanda diária | LT = lead time (dias)")
print(f"{'─'*80}")

resultados_es = {}

for linha in linhas:
    p = df_param[linha]
    produto      = p['produtos']
    sigma_d      = float(p['sigma_demanda_diaria'])
    lead_time    = int(p['lead_time_dias'])
    nivel_serv   = float(p['nivel_servico'])
    estoque_min  = int(p['estoque_minimo'])

    # Z para o nível de serviço
    z = stats.norm.ppf(nivel_serv)

    # ES = Z × σ_d × √LT
    es = z * sigma_d * np.sqrt(lead_time)
    es_arredondado = int(np.ceil(es))

    viabilidade = "Viável" if es_arredondado <= estoque_min else "Inviável (Requer maior espaço ou mudança de política)"

    resultados_es[linha] = {
        'produto': produto,
        'sigma_d': sigma_d,
        'lead_time': lead_time,
        'nivel_servico': nivel_serv,
        'estoque_minimo': estoque_min,
        'z': round(z, 4),
        'es': round(es, 1),
        'es_arredondado': es_arredondado,
        'viabilidade': viabilidade
    }

    print(f"\n  🏭 {linha} — {produto}")
    print(f"     σ_d = {sigma_d:,.0f} un/dia  |  LT = {lead_time} dias  |  NS = {nivel_serv*100:.0f}%  →  Z = {z:.4f}")
    print(f"     ES = {z:.4f} × {sigma_d:,.0f} × √{lead_time} = {es:,.1f} ≈ {es_arredondado:,} unidades")
    print(f"     Interpretação: O ES calculado ({es_arredondado:,}) vs Estoque Mínimo da empresa ({estoque_min:,}) → {viabilidade}")


# ============================================================
#  3. LOTE ECONÔMICO DE COMPRA (LEC / EOQ)
# ============================================================

print(f"\n\n{'═'*80}")
print(f"  📐 LOTE ECONÔMICO DE COMPRA (LEC / EOQ) — POR LINHA")
print(f"{'═'*80}")
print(f"  Fórmula: LEC = √(2 × D × S / H)")
print(f"  D = demanda anual | S = custo de setup/pedido | H = custo de manter estoque")
print(f"{'─'*80}")

resultados_lec = {}

for linha in linhas:
    p = df_param[linha]
    produto     = p['produtos']
    custo_setup = float(p['custo_setup'])
    custo_est   = float(p['custo_estoque'])

    # O insumo principal é o primeiro item da classe A (maior valor anual)
    df_abc = resultados_abc[linha]
    insumo_principal = df_abc.iloc[0]['item']
    demanda_anual = df_abc.iloc[0]['demanda_anual']

    # Se o custo_estoque nos parâmetros é mensal, multiplicamos por 12 para ter o H anual (R$/un/ano)
    H_anual = custo_est * 12

    # LEC = √(2DS / H)
    lec = np.sqrt(2 * demanda_anual * custo_setup / H_anual)
    lec_arredondado = int(np.ceil(lec / 100) * 100)  if lec >= 100 else int(np.ceil(lec))

    # Número de pedidos por ano
    n_pedidos = demanda_anual / lec
    # Intervalo entre pedidos (dias úteis, ~252 dias/ano)
    dias_uteis_ano = 252
    intervalo_pedidos = dias_uteis_ano / n_pedidos

    # Custo total anual de estoque (pedido + manutenção)
    custo_total_pedido = (demanda_anual / lec) * custo_setup
    custo_total_manut  = (lec / 2) * H_anual
    custo_total_estoque = custo_total_pedido + custo_total_manut

    resultados_lec[linha] = {
        'produto': produto,
        'insumo_principal': insumo_principal,
        'demanda_anual': round(demanda_anual),
        'custo_setup': custo_setup,
        'custo_estoque_anual': H_anual,
        'lec': round(lec, 1),
        'lec_arredondado': lec_arredondado,
        'n_pedidos': round(n_pedidos, 1),
        'intervalo_dias': round(intervalo_pedidos, 1),
        'custo_total_pedido': round(custo_total_pedido, 2),
        'custo_total_manut': round(custo_total_manut, 2),
        'custo_total_estoque': round(custo_total_estoque, 2),
    }

    print(f"\n  🏭 {linha} — Insumo Principal: {insumo_principal}")
    print(f"     D = {demanda_anual:,.0f} un/ano  |  S = R$ {custo_setup:,.2f}/pedido  |  H = R$ {H_anual:,.2f}/un/ano")
    print(f"     LEC = √(2 × {demanda_anual:,.0f} × {custo_setup} / {H_anual}) = {lec:,.1f} ≈ {lec_arredondado:,} un")
    print(f"     Nº pedidos/ano: {n_pedidos:.1f}  |  Intervalo: {intervalo_pedidos:.1f} dias úteis")
    print(f"     Custo anual — Pedidos: R$ {custo_total_pedido:,.2f}  |  Manutenção: R$ {custo_total_manut:,.2f}  |  Total: R$ {custo_total_estoque:,.2f}")


# ============================================================
#  4. PONTO DE PEDIDO (PP)
# ============================================================

print(f"\n\n{'═'*80}")
print(f"  📍 PONTO DE PEDIDO (PP) — POR LINHA")
print(f"{'═'*80}")
print(f"  Fórmula: PP = d̄ × LT + ES")
print(f"  d̄ = demanda média diária | LT = lead time (dias) | ES = estoque de segurança")
print(f"{'─'*80}")

resultados_pp = {}

for linha in linhas:
    p = df_param[linha]
    produto    = p['produtos']
    lead_time  = int(p['lead_time_dias'])
    es         = resultados_es[linha]['es_arredondado']

    # Demanda média diária (baseada em 21 dias úteis/mês)
    df_hist = pd.read_csv('dataset/trabalho_demanda.csv')
    demanda_media_mensal = df_hist[linha].mean()
    dias_uteis_mes = 21
    demanda_diaria = demanda_media_mensal / dias_uteis_mes

    # PP = d̄ × LT + ES
    pp = demanda_diaria * lead_time + es
    pp_arredondado = int(np.ceil(pp))

    resultados_pp[linha] = {
        'produto': produto,
        'demanda_diaria': round(demanda_diaria, 1),
        'lead_time': lead_time,
        'es': es,
        'pp': round(pp, 1),
        'pp_arredondado': pp_arredondado,
    }

    print(f"\n  🏭 {linha} — {produto}")
    print(f"     d̄ = {demanda_diaria:,.1f} un/dia  |  LT = {lead_time} dias  |  ES = {es:,} un")
    print(f"     PP = {demanda_diaria:,.1f} × {lead_time} + {es:,} = {pp:,.1f} ≈ {pp_arredondado:,} unidades")


# ============================================================
#  TABELA RESUMO CONSOLIDADA
# ============================================================

print(f"\n\n{'═'*80}")
print(f"  📊 RESUMO CONSOLIDADO — GESTÃO DE ESTOQUE")
print(f"{'═'*80}")

resumo_rows = []
for linha in linhas:
    es_data  = resultados_es[linha]
    lec_data = resultados_lec[linha]
    pp_data  = resultados_pp[linha]

    resumo_rows.append({
        'Linha':          linha,
        'Produto':        es_data['produto'],
        'NS (%)':         f"{es_data['nivel_servico']*100:.0f}%",
        'ES (un)':        es_data['es_arredondado'],
        'Viabilidade':    es_data['viabilidade'],
        'LEC (un)':       lec_data['lec_arredondado'],
        'Pedidos/Ano':    lec_data['n_pedidos'],
        'Interv. (dias)': lec_data['intervalo_dias'],
        'PP (un)':        pp_data['pp_arredondado'],
        'Custo Est. (R$)': f"R$ {lec_data['custo_total_estoque']:,.2f}",
    })

df_resumo = pd.DataFrame(resumo_rows)
print(df_resumo.to_string(index=False))

# Exportar CSV
df_resumo_csv = pd.DataFrame([{
    'Linha': r['Linha'],
    'Produto': r['Produto'],
    'Nivel Servico': r['NS (%)'],
    'Estoque Seguranca': resumo_rows[i]['ES (un)'],
    'Viabilidade ES': resumo_rows[i]['Viabilidade'],
    'LEC': resumo_rows[i]['LEC (un)'],
    'Pedidos Ano': resumo_rows[i]['Pedidos/Ano'],
    'Intervalo Dias': resumo_rows[i]['Interv. (dias)'],
    'Ponto Pedido': resumo_rows[i]['PP (un)'],
    'Custo Total Estoque': resultados_lec[r['Linha']]['custo_total_estoque'],
} for i, r in enumerate(resumo_rows)])

df_resumo_csv.to_csv('gestao_estoque_resumo.csv', index=False, encoding='utf-8-sig')
print(f"\n  📁 Tabela exportada: gestao_estoque_resumo.csv")


# ============================================================
#  EXPORTAR CLASSIFICAÇÃO ABC DETALHADA (CSV)
# ============================================================

for linha in linhas:
    if linha in resultados_abc:
        df_exp = resultados_abc[linha][['item', 'preco_unitario', 'demanda_anual',
                                         'valor_anual', 'perc_valor', 'perc_acumulado', 'classe']]
        nome_csv = f'abc_{linha}.csv'
        df_exp.to_csv(nome_csv, index=False, encoding='utf-8-sig')
        print(f"  📁 ABC detalhado exportado: {nome_csv}")


# ============================================================
#  GRÁFICOS
# ============================================================

# ── Gráfico 1: Curva ABC por linha (Pareto) ───────────────────

for linha in linhas:
    if linha not in resultados_abc:
        continue

    df_abc = resultados_abc[linha]
    produto = df_param[linha]['produtos']
    cor = cores[linha]

    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#F8F9FA')
    ax1.set_facecolor('#FFFFFF')

    n_itens = len(df_abc)
    x_pos = np.arange(n_itens)

    # Cores por classe ABC
    cores_abc = {'A': '#E53935', 'B': '#FB8C00', 'C': '#43A047'}
    bar_colors = [cores_abc[c] for c in df_abc['classe']]

    # Barras de valor anual
    bars = ax1.bar(x_pos, df_abc['valor_anual'], color=bar_colors, alpha=0.85,
                   zorder=2, width=0.7, edgecolor='white', linewidth=0.5)

    ax1.set_ylabel('Valor Anual (R$)', fontsize=11, color='#333333')
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'R$ {v:,.0f}'))

    # Eixo secundário: % acumulado
    ax2 = ax1.twinx()
    ax2.plot(x_pos, df_abc['perc_acumulado'], color='#1A1A2E', marker='o',
             markersize=5, linewidth=2.2, zorder=4, label='% Acumulado')
    ax2.set_ylabel('% Acumulado', fontsize=11, color='#1A1A2E')
    ax2.set_ylim(0, 105)

    # Linhas de referência 80% e 95%
    ax2.axhline(y=80, color='#E53935', linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.axhline(y=95, color='#FB8C00', linestyle='--', linewidth=1.2, alpha=0.6)
    ax2.text(n_itens - 0.5, 81, '80%', color='#E53935', fontsize=9, ha='right')
    ax2.text(n_itens - 0.5, 96, '95%', color='#FB8C00', fontsize=9, ha='right')

    # Rótulos dos itens
    nomes_itens = [nome[:20] + '…' if len(nome) > 20 else nome
                   for nome in df_abc['item']]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(nomes_itens, rotation=40, ha='right', fontsize=8.5)

    # Legenda de classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E53935', alpha=0.85, label='Classe A (≤80%)'),
        Patch(facecolor='#FB8C00', alpha=0.85, label='Classe B (80–95%)'),
        Patch(facecolor='#43A047', alpha=0.85, label='Classe C (>95%)'),
    ]
    ax1.legend(handles=legend_elements, fontsize=9, loc='upper center',
               framealpha=0.85, ncol=3)

    ax1.set_title(f'Curva ABC — Linha {linha} ({produto})\n'
                  f'Classificação de Materiais por Valor Anual',
                  fontsize=13, fontweight='bold', color='#1A1A2E', pad=12)

    ax1.grid(True, alpha=0.15, linestyle='--', axis='y')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#CCCCCC')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#CCCCCC')

    plt.tight_layout()
    nome_arq = f'abc_{linha}.png'
    plt.savefig(nome_arq, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"  📊 Gráfico Curva ABC {linha} salvo: {nome_arq}")


# ── Gráfico 2: Comparativo ES, LEC e PP por linha ────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('Comparativo de Gestão de Estoque — Todas as Linhas\n'
             'Estoque de Segurança (ES) × LEC × Ponto de Pedido (PP)',
             fontsize=13, fontweight='bold', color='#1A1A2E', y=1.02)

x_pos = np.arange(len(linhas))
nomes_linhas = [f"{l}\n{df_param[l]['produtos'].split()[0]}" for l in linhas]
bar_colors = [cores[l] for l in linhas]

# ES
ax = axes[0]
ax.set_facecolor('#FFFFFF')
vals_es = [resultados_es[l]['es_arredondado'] for l in linhas]
bars = ax.bar(x_pos, vals_es, color=bar_colors, alpha=0.85, zorder=2, width=0.6)
for bar, v in zip(bars, vals_es):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_es)*0.02,
            f'{v:,}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(nomes_linhas, fontsize=8.5)
ax.set_title('Estoque de Segurança (ES)', fontsize=11, fontweight='bold', color='#E53935')
ax.set_ylabel('Unidades', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.grid(True, alpha=0.2, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

# LEC
ax = axes[1]
ax.set_facecolor('#FFFFFF')
vals_lec = [resultados_lec[l]['lec_arredondado'] for l in linhas]
bars = ax.bar(x_pos, vals_lec, color=bar_colors, alpha=0.85, zorder=2, width=0.6)
for bar, v in zip(bars, vals_lec):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_lec)*0.02,
            f'{v:,}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(nomes_linhas, fontsize=8.5)
ax.set_title('Lote Econômico de Compra (LEC)', fontsize=11, fontweight='bold', color='#2196F3')
ax.set_ylabel('Unidades', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.grid(True, alpha=0.2, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

# PP
ax = axes[2]
ax.set_facecolor('#FFFFFF')
vals_pp = [resultados_pp[l]['pp_arredondado'] for l in linhas]
bars = ax.bar(x_pos, vals_pp, color=bar_colors, alpha=0.85, zorder=2, width=0.6)
for bar, v in zip(bars, vals_pp):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_pp)*0.02,
            f'{v:,}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(nomes_linhas, fontsize=8.5)
ax.set_title('Ponto de Pedido (PP)', fontsize=11, fontweight='bold', color='#4CAF50')
ax.set_ylabel('Unidades', fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
ax.grid(True, alpha=0.2, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

plt.tight_layout()
plt.savefig('gestao_estoque_comparativo.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"\n  📊 Gráfico comparativo salvo: gestao_estoque_comparativo.png")


# ── Gráfico 3: Custo total de estoque por linha ──────────────

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('#F8F9FA')
ax.set_facecolor('#FFFFFF')

custos_pedido = [resultados_lec[l]['custo_total_pedido'] for l in linhas]
custos_manut  = [resultados_lec[l]['custo_total_manut'] for l in linhas]
custos_total  = [resultados_lec[l]['custo_total_estoque'] for l in linhas]

w = 0.35
bars1 = ax.bar(x_pos - w/2, custos_pedido, width=w, color='#2196F3', alpha=0.85,
               label='Custo de Pedido', zorder=2)
bars2 = ax.bar(x_pos + w/2, custos_manut, width=w, color='#FF9800', alpha=0.85,
               label='Custo de Manutenção', zorder=2)

# Rótulos de valor
for bar, v in zip(bars1, custos_pedido):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(custos_total)*0.01,
            f'R$ {v/1e3:.0f}k', ha='center', va='bottom', fontsize=7.5, color='#2196F3')
for bar, v in zip(bars2, custos_manut):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(custos_total)*0.01,
            f'R$ {v/1e3:.0f}k', ha='center', va='bottom', fontsize=7.5, color='#FF9800')

# Linha de custo total
ax.plot(x_pos, custos_total, color='#1A1A2E', marker='D', markersize=7,
        linewidth=2, linestyle='--', label='Custo Total', zorder=3)
for i, v in enumerate(custos_total):
    ax.text(i, v + max(custos_total)*0.03,
            f'R$ {v:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(nomes_linhas, fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'R$ {v:,.0f}'))
ax.set_title('Custo Total de Estoque (LEC) — Pedido vs Manutenção\nPor Linha de Produção',
             fontsize=13, fontweight='bold', color='#1A1A2E', pad=12)
ax.set_xlabel('Linha de Produção', fontsize=11)
ax.set_ylabel('Custo Anual (R$)', fontsize=11)
ax.legend(fontsize=10, loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.25, linestyle='--', axis='y')
for sp in ax.spines.values():
    sp.set_edgecolor('#CCCCCC')

plt.tight_layout()
plt.savefig('gestao_estoque_custos.png', dpi=120, bbox_inches='tight')
plt.show()
print(f"  📊 Gráfico custos salvo: gestao_estoque_custos.png")


# ============================================================
#  CONCLUSÃO
# ============================================================

print(f"\n{'═'*80}")
print(f"  ✅  Gestão de Estoque concluída.")
print(f"     Linhas processadas: {len(linhas)}")
print(f"     Análises realizadas:")
print(f"       • Classificação ABC (Curva de Pareto) por linha")
print(f"       • Estoque de Segurança (ES) com nível de serviço")
print(f"       • Lote Econômico de Compra (LEC / EOQ)")
print(f"       • Ponto de Pedido (PP)")
custo_total_geral = sum(resultados_lec[l]['custo_total_estoque'] for l in linhas)
print(f"     Custo total de estoque (LEC): R$ {custo_total_geral:,.2f}")
print(f"{'═'*80}")
