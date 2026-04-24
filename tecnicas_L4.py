import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ========================================================
# 1. CARREGAR OS DADOS (L4)
# ========================================================
df = pd.read_csv('dataset/trabalho_demanda.csv')
y = df['L4'].values
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
# TÉCNICA 2: Holt (Duplo) - Aditivo
# ========================================================
# Ideal para tendências lineares constantes e sem sazonalidade.
model_holt_add = ExponentialSmoothing(y, trend='add', seasonal=None, initialization_method="estimated")
fit_holt_add = model_holt_add.fit()
y_pred_holt_add = fit_holt_add.fittedvalues
mad_holt_add, mse_holt_add, mape_holt_add = calc_errors(y, y_pred_holt_add)
y_fut_holt_add = fit_holt_add.forecast(7)

results.append({
    'Metodo': 'Holt Duplo (Aditivo)', 
    'MAD': round(mad_holt_add, 2), 'MSE': round(mse_holt_add, 2), 'MAPE (%)': round(mape_holt_add, 2), 
    'Previsao_Jan_Jul_2026': [int(round(val)) for val in y_fut_holt_add]
})

# ========================================================
# TÉCNICA 3: Holt (Duplo) - Multiplicativo
# ========================================================
# Ideal para tendências exponenciais (crescimento agressivo).
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_holt_mul = ExponentialSmoothing(y, trend='mul', seasonal=None, initialization_method="estimated")
    fit_holt_mul = model_holt_mul.fit()
    y_pred_holt_mul = fit_holt_mul.fittedvalues
mad_holt_mul, mse_holt_mul, mape_holt_mul = calc_errors(y, y_pred_holt_mul)
y_fut_holt_mul = fit_holt_mul.forecast(7)

results.append({
    'Metodo': 'Holt Duplo (Multiplicativo)', 
    'MAD': round(mad_holt_mul, 2), 'MSE': round(mse_holt_mul, 2), 'MAPE (%)': round(mape_holt_mul, 2), 
    'Previsao_Jan_Jul_2026': [int(round(val)) for val in y_fut_holt_mul]
})

# ========================================================
# EXIBINDO OS RESULTADOS
# ========================================================
res_df = pd.DataFrame(results)
print("=== PREVISAO DE DEMANDA: L4 ===")
print("Caracteristicas diagnosticadas: Tendencia Forte Crescente, Sazonalidade Inexistente\n")
print(res_df.to_string())
