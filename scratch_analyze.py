import pandas as pd
import numpy as np

df = pd.read_csv('c:/Users/Usuário/Trabalho PCP/dataset(1)/dataset/trabalho_demanda.csv')
linhas = ['L1', 'L2', 'L3', 'L4', 'L5']

for linha in linhas:
    y = df[linha].values
    
    # Linear trend
    x = np.arange(len(y))
    coef = np.polyfit(x, y, 1)
    slope = coef[0]
    
    y_pred = np.polyval(coef, x)
    r_squared = 1 - (sum((y - y_pred)**2) / sum((y - np.mean(y))**2))
    
    # Autocorrelation at lag 12 for Seasonality
    s = pd.Series(y)
    autocorr_12 = s.autocorr(lag=12)
    autocorr_1 = s.autocorr(lag=1)
    
    cv = np.std(y) / np.mean(y) # coefficient of variation
    
    print(f"--- Linha {linha} ---")
    print(f"Mean: {np.mean(y):.0f}, CV: {cv:.2f}")
    print(f"Slope: {slope:.2f}, R2: {r_squared:.2f}")
    print(f"Lag 12 Autocorrelation: {autocorr_12:.2f}")
    
    if slope > 200 and r_squared > 0.4:
        trend = "Forte (Crescente)"
    elif slope < -200 and r_squared > 0.4:
        trend = "Forte (Decrescente)"
    elif abs(slope) > 50 and r_squared > 0.2:
        trend = "Leve"
    else:
        trend = "Estacionária (Sem tendência clara)"
        
    if autocorr_12 > 0.6:
        season = "Forte"
    elif autocorr_12 > 0.3:
        season = "Moderada"
    else:
        season = "Fraca/Inexistente"
        
    print(f"Diagnóstico -> Tendência: {trend} | Sazonalidade: {season}")
    print()
