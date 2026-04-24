import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregamento dos dados (24 meses históricos)
try:
    df = pd.read_csv('dataset/trabalho_demanda.csv')
except FileNotFoundError:
    df = pd.DataFrame({
        'mes': [f'M{i}' for i in range(1, 25)],
        'L1': [22246, 21106, 22292, 25550, 25980, 28902, 23429, 23757, 25267, 28330, 27944, 26525, 29049, 26633, 26505, 28350, 32595, 31023, 28063, 29141, 31989, 31004, 32554, 30638]
    })

# Alfas baseados nas imagens do professor
alfa_valores = [0.1, 0.5, 0.8]
resultados = []

# Tabela "Efeito do Valor de alfa" (igual ao slide 2)
df_tabela = pd.DataFrame({
    'Período': df['mes'] if 'mes' in df.columns else df.index + 1,
    'Demanda': df['L1']
})

# Dicionário para guardar as linhas para o gráfico (igual ao slide 3)
linhas_grafico = {}

# 2. CALIBRAÇÃO: Calcular as previsões históricas para cada alfa
# Isso é feito para descobrir qual alfa erra menos testando-os no passado.
for alfa in alfa_valores:
    col_name = f'alfa = {alfa}'
    
    # M_1 = D_1 (primeiro mês copia a demanda real)
    M = [df['L1'].iloc[0]] 
    
    for i in range(1, len(df)):
        # Fórmula do slide: M_t = M_{t-1} + alfa * (D_{t-1} - M_{t-1})
        nova_prev = M[-1] + alfa * (df['L1'].iloc[i-1] - M[-1])
        M.append(nova_prev)
    
    # Salvar na tabela de exibição e no dicionário do gráfico
    df_tabela[col_name] = M
    linhas_grafico[col_name] = M
    
    # Calcular métricas de erro (ignorando o primeiro mês)
    y_true = df['L1'].iloc[1:]
    y_pred = pd.Series(M)[1:]
    
    mad = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    resultados.append({
        'Técnica': f'MEM (alfa={alfa})',
        'Alfa': alfa,
        'MAD': mad,
        'MSE': mse,
        'MAPE (%)': mape
    })

# Formatando a tabela com 1 casa decimal (exatamente igual ao slide 2)
tabela_print = df_tabela.copy()
for alfa in alfa_valores:
    tabela_print[f'alfa = {alfa}'] = tabela_print[f'alfa = {alfa}'].apply(lambda x: f"{x:.1f}".replace('.', ','))

print("--- Tabela: Efeito do Valor de alfa (CALIBRAÇÃO NO HISTÓRICO) ---")
print(tabela_print.to_string(index=False))
print("\n")

# 3. Tabela Comparativa de Erros
df_resultados = pd.DataFrame(resultados)
print("--- Tabela Comparativa de Erros ---")
print(df_resultados.to_string(index=False))
print("\n")

# Selecionar o melhor alfa baseado no menor erro (MAPE)
melhor_modelo = df_resultados.loc[df_resultados['MAPE (%)'].idxmin()]
melhor_alfa = melhor_modelo['Alfa']
print(f"-> Melhor parâmetro selecionado após calibração: alfa = {melhor_alfa} com MAPE de {melhor_modelo['MAPE (%)']:.2f}%\n")

# 4. PREVISÃO FUTURA (Para 2026)
ultimo_valor_real = df['L1'].iloc[-1]
ultima_prev_historica = linhas_grafico[f'alfa = {melhor_alfa}'][-1]

# Previsão para o Mês 25 (jan/2026) usando o alfa vencedor
previsao_futura = ultima_prev_historica + melhor_alfa * (ultimo_valor_real - ultima_prev_historica)
print(f"A previsão para Jan/2026 (e meses seguintes) usando o modelo vencedor é: {previsao_futura:.1f} unidades\n")

# 5. Visualização IGUAL AO SLIDE 3 (Mostrando as 3 linhas de alfa no histórico)
plt.figure(figsize=(12, 6))

# Plot do histórico real (linha preta)
plt.plot(df.index, df['L1'], marker='o', label='Demanda real', color='#2c3e50', linewidth=2)

# Cores parecidas com as do slide (verde, laranja, vermelho)
cores = ['#95a5a6', '#f39c12', '#e74c3c'] 

for idx, alfa in enumerate(alfa_valores):
    col_name = f'alfa = {alfa}'
    plt.plot(df.index, linhas_grafico[col_name], marker='s', linestyle='--', label=f'MEM alfa={alfa}', color=cores[idx], linewidth=1.5)

plt.title('Média Exponencial Móvel — Efeito de alfa', fontsize=14, pad=15)
plt.xlabel('Período')
plt.ylabel('Valor')

# Eixo X personalizado
plt.xticks(range(len(df)), df['mes'] if 'mes' in df.columns else df.index + 1, rotation=45, ha='right')

plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.legend()
plt.tight_layout()
plt.show()
