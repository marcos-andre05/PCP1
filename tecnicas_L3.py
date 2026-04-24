import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ========================================================
# 1. CARREGAR OS DADOS (L3)
# ========================================================
df = pd.read_csv('dataset/trabalho_demanda.csv')
y = df['L3'].values
n = len(y)

# ========================================================
# 2. FUNÇÕES DO PROFESSOR (MMS e MEM)
# ========================================================
def media_movel_simples(demandas, n_janela):
    previsoes = [None] * n_janela
    for t in range(n_janela, len(demandas)):
        janela = demandas[t - n_janela:t]
        previsoes.append(sum(janela) / n_janela)
    return previsoes

def media_exponencial_movel(demandas, alpha, m0=None):
    if m0 is None:
        m0 = demandas[0]
    previsoes = [m0]
    for t in range(1, len(demandas)):
        m_t = alpha * demandas[t - 1] + (1 - alpha) * previsoes[-1]
        previsoes.append(m_t)
    return previsoes

# ========================================================
# 3. FUNÇÃO DE CÁLCULO DE ERROS
# ========================================================
def calc_errors(y_true, y_pred):
    y_t = []
    y_p = []
    # Ignora os None (casos iniciais da MMS)
    for true_val, pred_val in zip(y_true, y_pred):
        if pred_val is not None:
            y_t.append(true_val)
            y_p.append(pred_val)
            
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    
    e = y_t - y_p
    mad = np.mean(np.abs(e))
    mse = np.mean(e**2)
    mape = np.mean(np.abs(e / y_t)) * 100
    return mad, mse, mape

results = []

# ========================================================
# TÉCNICA 1: MMS (Média Móvel Simples) usando código da aula
# ========================================================
janela = 3
y_pred_mms = media_movel_simples(y, janela)

# Projeção futura
historico_mms = list(y)
y_fut_mms = []
for i in range(7):
    prox_valor = sum(historico_mms[-janela:]) / janela
    y_fut_mms.append(prox_valor)
    historico_mms.append(prox_valor)

mad_mms, mse_mms, mape_mms = calc_errors(y, y_pred_mms)

results.append({
    'Metodo': f'MMS (n={janela})', 
    'MAD': round(mad_mms, 2), 'MSE': round(mse_mms, 2), 'MAPE (%)': round(mape_mms, 2), 
    'Previsao_Jan_Jul_2026': [int(round(x)) for x in y_fut_mms]
})

# ========================================================
# TÉCNICA 2: MEM (Média Exponencial Móvel) usando código da aula
# ========================================================
alpha = 0.3
y_pred_mem = media_exponencial_movel(y, alpha)

# Na MEM simples, a previsão futura é apenas o último valor calculado, que se repete
ultimo_valor_mem = y_pred_mem[-1]
y_fut_mem = [ultimo_valor_mem] * 7

mad_mem, mse_mem, mape_mem = calc_errors(y, y_pred_mem)

results.append({
    'Metodo': f'MEM (alpha={alpha})', 
    'MAD': round(mad_mem, 2), 'MSE': round(mse_mem, 2), 'MAPE (%)': round(mape_mem, 2), 
    'Previsao_Jan_Jul_2026': [int(round(x)) for x in y_fut_mem]
})

# ========================================================
# TÉCNICA 3: Holt-Winters (Sazonal sem Tendência)
# ========================================================
# Esta é a técnica estatisticamente recomendada para a L3 (lida bem com a sazonalidade)
model_hw = ExponentialSmoothing(y, trend=None, seasonal='mul', seasonal_periods=12, initialization_method="estimated")
fit_hw = model_hw.fit()
y_pred_hw = fit_hw.fittedvalues
y_fut_hw = fit_hw.forecast(7)

mad_hw, mse_hw, mape_hw = calc_errors(y, y_pred_hw)

results.append({
    'Metodo': 'Holt-Winters Sazonal (seas=mul)', 
    'MAD': round(mad_hw, 2), 'MSE': round(mse_hw, 2), 'MAPE (%)': round(mape_hw, 2), 
    'Previsao_Jan_Jul_2026': [int(round(x)) for x in y_fut_hw]
})

# ========================================================
# EXIBINDO OS RESULTADOS
# ========================================================
res_df = pd.DataFrame(results)
print("=== PREVISAO DE DEMANDA: L3 ===")
print("Caracteristicas diagnosticadas: Estacionaria com Forte Sazonalidade\n")
print(res_df.to_string())
