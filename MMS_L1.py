import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregamento dos dados
try:
    df = pd.read_csv('dataset/trabalho_demanda.csv')
except FileNotFoundError:
    # Dados de exemplo caso o CSV não seja encontrado
    df = pd.DataFrame({
        'Mes': [f'M{i}' for i in range(1, 25)],
        'L1': [22246, 21106, 22292, 25550, 25980, 28902, 23429, 23757, 25267, 28330, 27944, 26525, 29049, 26633, 26505, 28350, 32595, 31023, 28063, 29141, 31989, 31004, 32554, 30638]
    })

# Vamos testar diferentes tamanhos de janela (N) para a Média Móvel Simples
periodos_N = [2, 3, 4, 5, 6]
resultados = []

# 2. Calcular previsões e métricas de erro para cada N
for N in periodos_N:
    col_name = f'MMS_{N}'
    # Calculando a Média Móvel Simples
    df[col_name] = df['L1'].rolling(window=N).mean().shift(1)
    
    # Filtrando os dados onde temos a previsão (removendo os N primeiros NaN)
    df_valid = df.dropna(subset=[col_name])
    
    # Variáveis de cálculo
    y_true = df_valid['L1']
    y_pred = df_valid[col_name]
    
    # 1. Calcular métricas de erro (MAD, MSE, MAPE)
    # MAD (Mean Absolute Deviation) ou MAE
    mad = np.mean(np.abs(y_true - y_pred))
    
    # MSE (Mean Squared Error)
    mse = np.mean((y_true - y_pred)**2)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    resultados.append({
        'Técnica': f'MMS (N={N})',
        'N': N,
        'MAD': mad,
        'MSE': mse,
        'MAPE (%)': mape
    })

# 2. Comparar e selecionar - Apresentar tabela comparativa
df_resultados = pd.DataFrame(resultados)
print("--- Tabela Comparativa de Erros (Passo 3) ---")
print(df_resultados.to_string(index=False))
print("\n")

# Escolhendo a técnica com menor erro (baseado no menor MAPE, que é um bom padrão)
melhor_modelo = df_resultados.loc[df_resultados['MAPE (%)'].idxmin()]
melhor_N = melhor_modelo['N']
print(f"-> Melhor parâmetro selecionado: N={melhor_N} com MAPE de {melhor_modelo['MAPE (%)']:.2f}%\n")

# Previsão para o futuro (Mês 25) com o modelo selecionado
previsao_futura = df['L1'].tail(melhor_N).mean()
print(f"Previsão para o próximo período (Mês 25) usando MMS (N={melhor_N}): {previsao_futura:.2f}\n")

# ---------------------------------------------------------
# NOVA PARTE: Projeção para os 7 meses de 2026 (M25 a M31)
# ---------------------------------------------------------
ultimos_dados = list(df['L1'].tail(int(melhor_N))) # Pega os últimos 'N' dados reais
previsoes_7_meses = []

for mes_futuro in range(25, 32): # Loop para os meses 25 até 31 (7 meses)
    nova_previsao = np.mean(ultimos_dados)
    previsoes_7_meses.append(nova_previsao)
    
    # Atualiza a "janela": remove o dado mais antigo e insere a nova previsão
    ultimos_dados.pop(0)
    ultimos_dados.append(nova_previsao)

print(f"\nPrevisões para os 7 meses de 2026 usando MMS (N={melhor_N}):")
for i, prev in enumerate(previsoes_7_meses):
    print(f"Mês {25+i}: {prev:.2f} unidades")
print("\n")
# ---------------------------------------------------------