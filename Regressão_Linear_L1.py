import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================================
# 1. Entrada - Série histórica ordenada no tempo
# =====================================================================
# Lendo os dados reais do dataset (trabalho_demanda.csv)
#caminho_arquivo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'trabalho_demanda.csv')
df_completo = pd.read_csv('dataset/trabalho_demanda.csv')

# Vamos usar os meses como período X (1, 2, 3, ..., n) e L1 como demanda Y
n = len(df_completo)
X = pd.Series(range(1, n + 1))
Y = df_completo['L1']

df = pd.DataFrame({
    'Mes': df_completo['mes'],
    'X_Periodo': X,
    'Y_Demanda': Y
})

# =====================================================================
# 2. Processo - Agregar somatórios e aplicar as fórmulas
# =====================================================================
# Cálculos dos somatórios necessários
sum_X = X.sum()
sum_Y = Y.sum()
sum_XY = (X * Y).sum()
sum_X2 = (X**2).sum()
sum_Y2 = (Y**2).sum()

# Cálculo dos coeficientes 'b' (inclinação) e 'a' (intercepto)
b = (n * sum_XY - sum_X * sum_Y) / (n * sum_X2 - (sum_X**2))
a = (sum_Y - b * sum_X) / n

print("--- CÁLCULO DOS COEFICIENTES (Linha L1) ---")
print(f"a (Intercepto) = {a:.4f}")
print(f"b (Inclinação) = {b:.4f}")
print(f"Reta de Tendência: Y = {a:.4f} + {b:.4f}X\n")

# Cálculo do Coeficiente de Determinação (R²)
# Fórmula baseada no slide: R² = (a*ΣY + b*ΣXY - (ΣY)²/n) / (ΣY² - (ΣY)²/n)
numerador_R2 = (a * sum_Y) + (b * sum_XY) - ((sum_Y**2) / n)
denominador_R2 = sum_Y2 - ((sum_Y**2) / n)
R2 = numerador_R2 / denominador_R2

print("--- COEFICIENTE DE DETERMINAÇÃO (R²) ---")
print(f"R² = {R2:.4f}")

# Leitura prática do R² baseada no material
if R2 >= 0.9:
    print("Leitura: R² próximo de 1 indica ajuste muito forte.")
elif 0.7 <= R2 < 0.9:
    print("Leitura: R² entre 0,7 e 0,9 costuma ser aceitável.")
else:
    print("Leitura: R² abaixo de 0,7 sugere que a reta linear explica pouco da série.")
    
print("\nNota: Um R² elevado não garante perfeição, mas indica que a tendência linear é um bom resumo dos dados.\n")
print("-" * 50 + "\n")


# =====================================================================
# 3. Saída - Projeções e Mínimos Quadrados
# =====================================================================
# Previsão (P_t) e Erro (e_t)
df['P_t (Previsao)'] = a + b * X
df['e_t (Erro)'] = Y - df['P_t (Previsao)']
df['e_t^2 (Erro Quadrado)'] = df['e_t (Erro)']**2

print("--- TABELA DE RESULTADOS (Mínimos Quadrados) ---")
print(df[['Mes', 'X_Periodo', 'Y_Demanda', 'P_t (Previsao)', 'e_t (Erro)', 'e_t^2 (Erro Quadrado)']].round(2))

soma_erros_quadrados = df['e_t^2 (Erro Quadrado)'].sum()
print(f"\nSoma dos Erros ao Quadrado (Σe_t²): {soma_erros_quadrados:.2f}")
print("O método dos Mínimos Quadrados garantiu que este valor seja o menor possível para uma reta.\n")


# =====================================================================
# Visualização Gráfica do que a Regressão Linear Faz
# =====================================================================
plt.figure(figsize=(12, 6))

# Curva de demanda real
plt.plot(X, Y, color='#f39c12', marker='o', linewidth=2, label='Demanda Real (L1)', zorder=3)

# Reta de Regressão (A melhor reta que resume o comportamento da série)
plt.plot(X, df['P_t (Previsao)'], color='#2c3e50', linewidth=2, label=f'Reta: Y = {a:.2f} + {b:.2f}X', zorder=2)

# Destacar os erros (resíduos / e_t) - distâncias da reta
for i in range(n):
    plt.plot([X[i], X[i]], [Y[i], df['P_t (Previsao)'][i]], color='#e74c3c', linestyle='-', linewidth=1.5, zorder=1)

# Adicionar uma anotação "e_t" em um dos pontos de erro para ilustrar a explicação teórica
idx_exemplo = n // 2
plt.text(X[idx_exemplo] + 0.15, (Y[idx_exemplo] + df['P_t (Previsao)'][idx_exemplo]) / 2, 
         '$e_t$', color='#e74c3c', fontsize=12, fontweight='bold')

plt.title('Regressão Linear: Resumo do Sinal da Demanda L1 (Separando o Ruído)', fontsize=14)
plt.xlabel('Tempo (Meses de Jan/24 a Dez/25)', fontsize=12)
plt.ylabel('Demanda L1', fontsize=12)

# Adicionando os rótulos dos meses no eixo X
plt.xticks(X, df['Mes'], rotation=45)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Exibir gráfico
plt.show()
