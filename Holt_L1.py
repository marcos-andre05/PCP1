import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. Entrada - Série histórica
# =====================================================================
# Lendo a base de dados
caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'trabalho_demanda.csv')
df = pd.read_csv(caminho_arquivo)

n = len(df)
X = np.arange(1, n + 1)
Y = df['L1'].values

# =====================================================================
# 2. Processo 1 - Regressão Linear (para o gráfico comparativo)
# =====================================================================
sum_X = X.sum()
sum_Y = Y.sum()
sum_XY = (X * Y).sum()
sum_X2 = (X**2).sum()

b_reg = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - sum_X**2)
a_reg = (sum_Y - b_reg * sum_X) / n
P_regressao = a_reg + b_reg * X

# =====================================================================
# 3. Processo 2 - Método de Holt (Suavização Exponencial Dupla)
# =====================================================================
# Parâmetros de Calibração (conforme slide do gráfico)
alpha = 0.3  # peso do nível
beta = 0.1   # peso da tendência

# Inicializando arrays
M = np.zeros(n)  # Nível (M_t)
T = np.zeros(n)  # Tendência (T_t)
P = np.zeros(n)  # Previsão (P_t)

# Inicialização (conforme regras do slide: M_1 = D_1 e T_1 = 0)
M[0] = Y[0]
T[0] = 0
P[0] = Y[0]  # No período 1 a previsão é igual à demanda real para inicializar o gráfico

# Loop do período 2 em diante (índice 1 até n-1)
for t in range(1, n):
    # Fórmulas exatas do slide:
    # 1. Previsão de t usando os dados de t-1: P_t = M_{t-1} + T_{t-1}
    P[t] = M[t-1] + T[t-1]
    
    # 2. Atualização do Nível: M_t = α * D_t + (1 - α) * (M_{t-1} + T_{t-1})
    M[t] = alpha * Y[t] + (1 - alpha) * (M[t-1] + T[t-1])
    
    # 3. Atualização da Tendência: T_t = β * (M_t - M_{t-1}) + (1 - β) * T_{t-1}
    T[t] = beta * (M[t] - M[t-1]) + (1 - beta) * T[t-1]

# =====================================================================
# 4. Saída - Exibição de Tabelas e Resultados
# =====================================================================
df_resultados = pd.DataFrame({
    'Mes': df['mes'],
    'Período (t)': X,
    'Demanda (D_t)': Y,
    'Nível (M_t)': M,
    'Tendência (T_t)': T,
    'Previsão Holt (P_t)': P,
    'Erro (e_t)': Y - P
})

print("=== MÉTODO DE HOLT (DUPLO AJUSTAMENTO) ===")
print(f"Parâmetros adotados: Alfa (α) = {alpha} | Beta (β) = {beta}")
print("\nResultados - Primeiros 5 meses:")
print(df_resultados.head(5).round(2).to_string(index=False))

print("\nResultados - Últimos 5 meses:")
print(df_resultados.tail(5).round(2).to_string(index=False))

# Previsão para o período n+1 (futuro)
Previsao_futura = M[-1] + T[-1]
print(f"\n>>> PREVISÃO HOLT PARA O PRÓXIMO MÊS (Período {n+1}): {Previsao_futura:.2f} <<<")

# =====================================================================
# 5. Visualização Gráfica (Idêntica ao slide de exemplo)
# =====================================================================
plt.figure(figsize=(11, 6))

# Demanda Real: Linha preta com marcadores circulares
plt.plot(X, Y, color='#2c3e50', marker='o', markersize=4, linewidth=1.5, label='Demanda real', zorder=3)

# Método de Holt: Linha verde tracejada com marcadores quadrados
plt.plot(X, P, color='#9cb57c', marker='s', markersize=4, linestyle='--', linewidth=1.5, label=f'Holt (α={alpha}, β={beta})', zorder=4)

# Regressão Linear: Linha laranja pontilhada fina
plt.plot(X, P_regressao, color='#f39c12', linestyle=':', linewidth=1.2, alpha=0.8, label='Regressão linear', zorder=2)

# Estilização
plt.title('Método de Holt — Suavização de Nível e Tendência', fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
plt.xlabel('Período', fontsize=11, color='#2c3e50')
plt.ylabel('Demanda', fontsize=11, color='#2c3e50')

# Rótulos do eixo X (Apenas para mostrar os números dos períodos como no slide)
plt.xticks(np.arange(0, 30, 5)) 

# Ajustando as bordas e grids (grid apenas nas horizontais com transparência)
plt.grid(axis='y', linestyle='-', alpha=0.3, color='gray')
plt.grid(axis='x', linestyle='-', alpha=0.1, color='gray')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#dddddd')
ax.spines['bottom'].set_color('#dddddd')

# Legenda estilizada no topo à esquerda
plt.legend(loc='upper left', frameon=True, edgecolor='lightgray', fontsize=9)

plt.tight_layout()
plt.show()
