"""
caracteristicas_demanda.py
==========================
Analisa as características estatísticas de cada linha de produção
e imprime uma tabela-resumo com:
  - Tendência       : regressão linear (slope + p-valor)
  - Sazonalidade    : ACF no lag 12 comparado ao limite de significância
  - Estacionariedade: Teste ADF (Augmented Dickey-Fuller)
  - CV (%)          : coeficiente de variação
  - Level Shift     : detecção por janela deslizante
  - Outliers (IQR)  : contagem de pontos fora dos limites IQR
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
from funções.TRATAMENTO import analisar_anomalias

# ── 1. Dados ────────────────────────────────────────────────────────────────
df = pd.read_csv('new_dataset/trabalho_demanda.csv')
df_param = pd.read_csv('new_dataset/trabalho_parametros.csv', index_col=0)

colunas = df.columns[1:].tolist()
produtos_map = df_param.loc['produtos'].to_dict()
nomes = {col: produtos_map.get(col, f'Produto {col}') for col in colunas}

# ── 2. Análise por linha ─────────────────────────────────────────────────────
resultados = []

for col in colunas:
    serie = df[col].values.astype(float)
    n     = len(serie)

    # ── Estatísticas básicas ──────────────────────────────────────────────
    media = np.mean(serie)
    desvio = np.std(serie, ddof=1)
    cv    = (desvio / media) * 100

    # ── Tendência (regressão linear simples) ─────────────────────────────
    x = np.arange(n)
    slope, _, r_value, p_trend, _ = stats.linregress(x, serie)

    if p_trend < 0.05:
        tendencia = 'Crescente' if slope > 0 else 'Decrescente'
    else:
        tendencia = 'Estável'

    # ── Estacionariedade (ADF) ───────────────────────────────────────────
    # H0: série possui raiz unitária (não-estacionária)
    # Se p < 0.05 → rejeita H0 → série é estacionária
    adf_stat, adf_p, *_ = adfuller(serie, autolag='AIC')
    estacionaria = 'Sim' if adf_p < 0.05 else 'Não'

    # ── Sazonalidade (ACF no lag 12) ─────────────────────────────────────
    # Limite de significância Bartlett para n observações: ±1.96/√n
    limite_acf = 1.96 / np.sqrt(n)
    acf_vals   = acf(serie, nlags=12, fft=False)
    acf_lag12  = acf_vals[12]
    sazonalidade = 'Sim' if abs(acf_lag12) > limite_acf else 'Não'

    # ── Anomalias (módulo TRATAMENTO) ────────────────────────────────────
    rel = analisar_anomalias(serie.tolist())

    resultados.append({
        'Linha'         : col,
        'Produto'       : nomes[col],
        'Média (un)'    : int(round(media)),
        'CV (%)'        : round(cv, 1),
        'Tendência'     : tendencia,
        'p-valor Tend.' : round(p_trend, 4),
        'R²'            : round(r_value**2, 3),
        'Sazonalidade'  : sazonalidade,
        'ACF Lag-12'    : round(acf_lag12, 3),
        'Lim. ACF'      : round(limite_acf, 3),
        'Estacionária'  : estacionaria,
        'ADF p-valor'   : round(adf_p, 4),
        'Level Shift'   : 'Sim' if rel['level_shift'] else 'Não',
        'Outliers (IQR)': rel['n_outliers'],
    })

# ── 3. Tabela ────────────────────────────────────────────────────────────────
tabela = pd.DataFrame(resultados).set_index('Linha')

# Impressão formatada no console
largura = 110
print()
print('=' * largura)
print('  CARACTERÍSTICAS ESTATÍSTICAS DAS LINHAS DE PRODUÇÃO')
print('=' * largura)

# Tabela resumida (principais colunas)
resumo = tabela[[
    'Produto', 'Média (un)', 'CV (%)',
    'Tendência', 'p-valor Tend.', 'R²',
    'Sazonalidade', 'ACF Lag-12',
    'Estacionária', 'ADF p-valor',
    'Level Shift', 'Outliers (IQR)'
]]
print(resumo.to_string())
print('=' * largura)
print()
print('Legenda:')
print('  CV (%)        : Coeficiente de Variacao - variabilidade relativa da serie')
print('  Tendência     : Significativa se p-valor Tend. < 0.05')
print('  R2            : Coeficiente de determinacao da regressao linear')
print('  Sazonalidade  : ACF no lag 12 significativo se |ACF Lag-12| > Lim. ACF (~{:.3f})'.format(1.96/np.sqrt(len(df))))
print('  Estacionaria  : Teste ADF - "Sim" se ADF p-valor < 0.05')
print('  Level Shift   : Mudanca de patamar detectada por janela deslizante (6 periodos, limiar 40%)')
print('  Outliers (IQR): Pontos fora de [Q1 - 1.5*IQR , Q3 + 1.5*IQR]')
print()

# ── 4. Exportar para CSV ─────────────────────────────────────────────────────
tabela.to_csv('caracteristicas_linhas.csv', encoding='utf-8-sig')
print("OK - Tabela completa salva em 'caracteristicas_linhas.csv'")
