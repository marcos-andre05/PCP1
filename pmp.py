import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from funções.UTILS import gerar_meses_futuros, obter_cores_dinamicas

# ============================================================
#  PLANO MESTRE DE PRODUÇÃO (PMP)  —  Jan/2026 a Mar/2026
#  Desagregação mensal → semanal  (Jan=5, Fev=4, Mar=4 semanas)
# ============================================================

# ── Carregar dados ────────────────────────────────────────────
df_param = pd.read_csv('dataset/trabalho_parametros.csv', index_col=0)
df_hist  = pd.read_csv('dataset/trabalho_demanda.csv')
linhas   = df_hist.columns[1:].tolist()

ultimo_mes_historico = df_hist['mes'].iloc[-1]
MESES = [m.title() for m in gerar_meses_futuros(ultimo_mes_historico, 3)]
T_MESES = 3
SEMANAS_POR_MES_LISTA = [5, 4, 4]
T_SEMANAS = sum(SEMANAS_POR_MES_LISTA)   # 13 semanas

with open('plano_agregado_detalhado.json', 'r') as f:
    dados_agregado = json.load(f)

print(f"{'═'*80}")
print(f"  📋 PLANO MESTRE DE PRODUÇÃO (PMP) — Jan/2026 a Mar/2026")
print(f"{'═'*80}")
print(f"\n  🔍 Processando previsões e desagregando o Plano Agregado por linha...\n")


# ============================================================
#  FUNÇÕES DO PMP
# ============================================================

def desagregar_mensal_para_semanal_dinamico(mensal_list, semanas_lista):
    """
    Desagrega previsão mensal em semanal (distribuição uniforme).
    Garante que a soma exata do mês bata com o total fornecido.
    """
    semanal = []
    for m_idx, mensal in enumerate(mensal_list):
        n_semanas = semanas_lista[m_idx]
        base = round(mensal / n_semanas)
        semanas_mes = [base] * n_semanas
        # Ajusta a primeira semana para fechar a soma exata (compensar arredondamento)
        semanas_mes[0] += (int(mensal) - sum(semanas_mes))
        semanal.extend(semanas_mes)
    return semanal


def gerar_pedidos_firmes(previsao_semanal, horizonte_firme=8):
    """
    Simula pedidos em carteira para as primeiras semanas.
    """
    np.random.seed(42)
    pedidos = []
    for t in range(len(previsao_semanal)):
        if t < horizonte_firme:
            variacao = np.random.uniform(-0.10, 0.10)
            pedidos.append(max(0, round(previsao_semanal[t] * (1 + variacao))))
        else:
            pedidos.append(0)
    return pedidos


def calcular_pmp(estoque_inicial, previsao, pedidos, producao_semanal, subcontratacao_semanal, estoque_seguranca=0):
    pmp = []
    estoque_projetado = []
    estoque_disponivel = []
    demanda_considerada = []
    recebimentos_programados = []
    estoque_atual = estoque_inicial

    for t in range(len(previsao)):
        demanda_periodo = max(previsao[t], pedidos[t])
        demanda_considerada.append(demanda_periodo)
        
        producao = producao_semanal[t]
        subcontratacao = subcontratacao_semanal[t]
        
        estoque_atual = estoque_atual + producao + subcontratacao - demanda_periodo
        
        atp = max(0, estoque_atual - estoque_seguranca)
        
        pmp.append(producao)
        recebimentos_programados.append(subcontratacao)
        estoque_projetado.append(estoque_atual)
        estoque_disponivel.append(atp)

    return {
        'pmp': pmp,
        'recebimentos_programados': recebimentos_programados,
        'estoque_projetado': estoque_projetado,
        'demanda_considerada': demanda_considerada,
        'disponivel_promessa': estoque_disponivel,
    }


# ============================================================
#  EXECUÇÃO DO PMP PARA TODAS AS LINHAS
# ============================================================

resultados_pmp = {}
tecnicas = {}

for linha in linhas:
    p = df_param[linha]
    est_inicial = float(p['estoque_inicial'])
    # O estoque mínimo no agregado é mensal. O semanal será a mesma referência de segurança
    est_seguranca = int(float(p['estoque_minimo'])) 

    # Dados do Plano Agregado para os 3 primeiros meses
    dados = dados_agregado[linha]
    tecnicas[linha] = dados['estrategia']
    
    D_mensal = dados['D'][:3]
    X_mensal = dados['X'][:3]
    O_mensal = dados['O'][:3]
    S_mensal = dados['S'][:3]
    
    P_mensal = [x + o for x, o in zip(X_mensal, O_mensal)] # Total Produzido
    
    # Desagregação
    prev_semanal = desagregar_mensal_para_semanal_dinamico(D_mensal, SEMANAS_POR_MES_LISTA)
    prod_semanal = desagregar_mensal_para_semanal_dinamico(P_mensal, SEMANAS_POR_MES_LISTA)
    sub_semanal = desagregar_mensal_para_semanal_dinamico(S_mensal, SEMANAS_POR_MES_LISTA)

    pedidos_firmes = gerar_pedidos_firmes(prev_semanal, horizonte_firme=6)

    resultado = calcular_pmp(
        estoque_inicial=est_inicial,
        previsao=prev_semanal,
        pedidos=pedidos_firmes,
        producao_semanal=prod_semanal,
        subcontratacao_semanal=sub_semanal,
        estoque_seguranca=est_seguranca
    )

    resultados_pmp[linha] = {
        **resultado,
        'previsao_semanal': prev_semanal,
        'pedidos_firmes': pedidos_firmes,
        'produto': p['produtos']
    }


# ============================================================
#  RÓTULOS DAS SEMANAS
# ============================================================

semanas_labels = []
for m_idx, mes in enumerate(MESES):
    n_semanas = SEMANAS_POR_MES_LISTA[m_idx]
    for s in range(1, n_semanas + 1):
        semanas_labels.append(f"S{s}-{mes[:3]}")


# ============================================================
#  IMPRESSÃO — PLANILHA PMP POR LINHA
# ============================================================

def imprimir_pmp_linha(linha):
    r = resultados_pmp[linha]
    print(f"\n{'═'*90}")
    print(f"  🏭 PMP — Linha {linha} — {r['produto']}")
    print(f"     Estratégia Agregada Vencedora: {tecnicas[linha]}")
    print(f"{'═'*90}")

    idx_global = 0
    for m_idx, mes in enumerate(MESES):
        n_semanas = SEMANAS_POR_MES_LISTA[m_idx]
        inicio = idx_global
        fim = inicio + n_semanas
        idx_global += n_semanas

        print(f"\n  📅 {mes.upper()} ({n_semanas} semanas)")
        print(f"  {'─'*95}")

        rows = []
        for t in range(inicio, fim):
            rows.append({
                'Semana':       semanas_labels[t],
                'Previsão':     int(r['previsao_semanal'][t]),
                'Pedidos':      int(r['pedidos_firmes'][t]),
                'Demanda':      int(r['demanda_considerada'][t]),
                'PMP(Prod)':    int(r['pmp'][t]),
                'Subcontr.':    int(r['recebimentos_programados'][t]),
                'Est.Proj.':    int(r['estoque_projetado'][t]),
                'ATP':          int(r['disponivel_promessa'][t]),
            })

        df_mes = pd.DataFrame(rows)
        print(df_mes.to_string(index=False))

    total_pmp = sum(r['pmp'])
    total_sub = sum(r['recebimentos_programados'])
    total_demanda = sum(r['demanda_considerada'])
    est_final = r['estoque_projetado'][-1]
    
    print(f"\n  {'─'*95}")
    print(f"  📊 TOTAIS — {linha} (3 Meses)")
    print(f"     Produção total (PMP):   {total_pmp:>10,} un")
    print(f"     Subcontratação total:   {total_sub:>10,} un")
    print(f"     Demanda total:          {total_demanda:>10,} un")
    print(f"     Estoque final:          {est_final:>10,} un")


for linha in linhas:
    imprimir_pmp_linha(linha)


# ============================================================
#  EXPORTAR PMP DETALHADO POR LINHA (CSV)
# ============================================================

for linha in linhas:
    r = resultados_pmp[linha]
    df_det = pd.DataFrame({
        'Semana': semanas_labels,
        'Previsão': r['previsao_semanal'],
        'Pedidos Firmes': r['pedidos_firmes'],
        'Demanda Considerada': r['demanda_considerada'],
        'PMP (Produção)': r['pmp'],
        'Recebimento Subcontratação': r['recebimentos_programados'],
        'Estoque Projetado': r['estoque_projetado'],
        'ATP': r['disponivel_promessa'],
    })
    nome_csv = f'pmp_{linha}.csv'
    df_det.to_csv(nome_csv, index=False, encoding='utf-8-sig')
    print(f"\n  Detalhamento exportado: {nome_csv}")


# ============================================================
#  GRÁFICOS — PMP por Linha
# ============================================================

cores_lista = obter_cores_dinamicas(len(linhas))
cores = {linha: mcolors.to_hex(cores_lista[i]) for i, linha in enumerate(linhas)}

for linha in linhas:
    r = resultados_pmp[linha]
    cor = cores[linha]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.patch.set_facecolor('#F8F9FA')
    fig.suptitle(f'Plano Mestre de Produção — Linha {linha} ({r["produto"]})\n'
                 f'Agregado: {tecnicas[linha]}',
                 fontsize=13, fontweight='bold', color='#1A1A2E', y=0.98)

    x_idx = list(range(T_SEMANAS))

    # ── Gráfico superior: PMP vs Demanda ──────────────────────
    ax1.set_facecolor('#FFFFFF')

    # Barras do PMP + Subcontratação
    ax1.bar(x_idx, r['pmp'], color=cor, alpha=0.75, label='PMP (Produção)', zorder=2, width=0.6)
    bot_sub = r['pmp']
    ax1.bar(x_idx, r['recebimentos_programados'], bottom=bot_sub, color='#78909C', alpha=0.85, label='Subcontratação', zorder=2, width=0.6)

    # Linha de demanda considerada
    ax1.plot(x_idx, r['demanda_considerada'], color='#1A1A2E', marker='o',
             markersize=3.5, linewidth=1.8, linestyle='--',
             label='Demanda considerada', zorder=3)

    # Pedidos firmes (apenas onde > 0)
    pedidos_nz = [(i, v) for i, v in enumerate(r['pedidos_firmes']) if v > 0]
    if pedidos_nz:
        px, py = zip(*pedidos_nz)
        ax1.scatter(px, py, color='#E53935', marker='D', s=25,
                    label='Pedidos firmes', zorder=4)

    # Separadores de mês
    idx_acum = 0
    for m in range(T_MESES - 1):
        idx_acum += SEMANAS_POR_MES_LISTA[m]
        ax1.axvline(x=idx_acum - 0.5, color='#CCCCCC',
                    linestyle='-', linewidth=1.2, alpha=0.8)

    ax1.set_ylabel('Unidades', fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax1.legend(fontsize=8.5, loc='upper left', framealpha=0.85, ncol=2)
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#CCCCCC')

    # ── Gráfico inferior: Estoque Projetado ───────────────────
    ax2.set_facecolor('#FFFFFF')

    ax2.fill_between(x_idx, r['estoque_projetado'], alpha=0.3, color=cor, zorder=2)
    ax2.plot(x_idx, r['estoque_projetado'], color=cor, marker='o',
             markersize=3, linewidth=1.8, label='Estoque Projetado', zorder=3)

    # Separadores de mês
    idx_acum = 0
    for m in range(T_MESES - 1):
        idx_acum += SEMANAS_POR_MES_LISTA[m]
        ax2.axvline(x=idx_acum - 0.5, color='#CCCCCC',
                    linestyle='-', linewidth=1.2, alpha=0.8)

    ax2.set_ylabel('Estoque (un)', fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax2.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
    ax2.grid(True, alpha=0.2, linestyle='--', axis='y')
    for sp in ax2.spines.values():
        sp.set_edgecolor('#CCCCCC')

    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(semanas_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_xlabel('Semana', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    nome_arq = f'pmp_{linha}.png'
    plt.savefig(nome_arq, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  📊 Gráfico PMP {linha} salvo: {nome_arq}")


print(f"\n{'═'*90}")
print(f"  ✅  Plano Mestre de Produção (PMP) concluído com sucesso.")
print(f"     Horizonte: {T_MESES} meses ({T_SEMANAS} semanas: Jan=5, Fev=4, Mar=4)")
print(f"{'═'*90}")