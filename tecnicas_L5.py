import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

# ========================================================
# 1. CARREGAR OS DADOS (L5)
# ========================================================
df = pd.read_csv('dataset/trabalho_demanda.csv')
y = df['L5'].values
n = len(y)
t = np.arange(1, n + 1)
t_fut = np.arange(n + 1, n + 8) 

# ========================================================
# 2. FUNÇÃO DO PROFESSOR (Regressão Linear)
# ========================================================
def regressao_linear_simples(x, y):
    n_pontos = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)
    
    b = (n_pontos * sum_xy - sum_x * sum_y) / (n_pontos * sum_x2 - sum_x ** 2)
    a = (sum_y - b * sum_x) / n_pontos
    return a, b

# ========================================================
# 3. FUNÇÃO DE CÁLCULO DE ERROS
# ========================================================
def calc_errors(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    e = y_true - y_pred
    mad = np.mean(np.abs(e))
    mse = np.mean(e**2)
    mape = np.mean(np.abs(e / y_true)) * 100
    return mad, mse, mape

results = []

# ========================================================
# TÉCNICA 1: Regressão Linear Simples
# ========================================================
a, b = regressao_linear_simples(t, y)

y_pred_lr = [a + b * ti for ti in t]
mad_lr, mse_lr, mape_lr = calc_errors(y, y_pred_lr)
y_fut_lr = [a + b * ti for ti in t_fut]

results.append({
    'Metodo': 'Regressao Linear', 
    'MAD': round(mad_lr, 2), 'MSE': round(mse_lr, 2), 'MAPE (%)': round(mape_lr, 2), 
    'Previsao_Jan_Jul_2026': [int(round(val)) for val in y_fut_lr]
})

# ========================================================
# TÉCNICA 2: Holt-Winters (Triplo Aditivo)
# ========================================================
# Lida com a tendência crescente e a sazonalidade moderada de forma combinada
model_hw = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated")
fit_hw = model_hw.fit()
y_pred_hw = fit_hw.fittedvalues
mad_hw, mse_hw, mape_hw = calc_errors(y, y_pred_hw)
y_fut_hw = fit_hw.forecast(7)

results.append({
    'Metodo': 'Holt-Winters Triplo (add/add)', 
    'MAD': round(mad_hw, 2), 'MSE': round(mse_hw, 2), 'MAPE (%)': round(mape_hw, 2), 
    'Previsao_Jan_Jul_2026': [int(round(val)) for val in y_fut_hw]
})

# ========================================================
# TÉCNICA 3: Decomposição Sazonal + Regressão Linear
# ========================================================
# Captura a sazonalidade, aplica regressão na tendência purificada e recompõe
decomp = seasonal_decompose(y, model='multiplicative', period=12)
seasonal_indices = decomp.seasonal[:12]

y_deseasonalized = y / decomp.seasonal

a_seas, b_seas = regressao_linear_simples(t, y_deseasonalized)
y_pred_trend_seas = [a_seas + b_seas * ti for ti in t]

y_pred_decomp = np.array(y_pred_trend_seas) * decomp.seasonal
mad_decomp, mse_decomp, mape_decomp = calc_errors(y, y_pred_decomp)

y_fut_trend_seas = [a_seas + b_seas * ti for ti in t_fut]
fut_seasonal = np.array([seasonal_indices[i % 12] for i in range(n, n + 7)])
y_fut_decomp = np.array(y_fut_trend_seas) * fut_seasonal

results.append({
    'Metodo': 'Decomposicao Sazonal + Regressao', 
    'MAD': round(mad_decomp, 2), 'MSE': round(mse_decomp, 2), 'MAPE (%)': round(mape_decomp, 2), 
    'Previsao_Jan_Jul_2026': [int(round(val)) for val in y_fut_decomp]
})

# ========================================================
# EXIBINDO OS RESULTADOS
# ========================================================
res_df = pd.DataFrame(results)
print("=== PREVISAO DE DEMANDA: L5 ===")
print("Caracteristicas diagnosticadas: Tendencia Forte Crescente, Sazonalidade Moderada\n")
print(res_df.to_string())
