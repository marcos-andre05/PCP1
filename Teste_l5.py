import pandas as pd
from funções.MMS import media_movel_simples
from funções.MEM import media_exponencial_movel
from funções.REGRESSÃO_LINEAR import regressao_linear
from funções.MÉTRICAS_ERRO import calcular_erros
from funções.TRATAMENTO import tratar_anomalias_demanda 

# 1. Carregamento dos Dados (Exemplo com os 24 meses da Linha 3)
df = pd.read_csv('dataset/trabalho_demanda.csv')
demandas_L5 = df['L5'].tolist()

print("--- INICIANDO O TORNEIO DE TÉCNICAS: LINHA 5 ---\n")

demandas_L5 = tratar_anomalias_demanda(demandas_L5)
# 2. Aplicação das 3 Técnicas Escolhidas (Apenas nos 24 meses históricos)
# Técnica 1: Média Móvel Simples
previsoes_mms = media_movel_simples(demandas_L5, n=2)

# Técnica 2: Média Exponencial Móvel
previsoes_mem = media_exponencial_movel(demandas_L5, alpha=0.1)

# Técnica 3: Regressão Linear
resultado_rl = regressao_linear(demandas_L5)
previsoes_rl = resultado_rl["previsoes"]

# 3. Cálculo das Métricas de Erro usando o seu arquivo independente
erros_mms = calcular_erros(demandas_L5, previsoes_mms)
erros_mem = calcular_erros(demandas_L5, previsoes_mem)
erros_rl = calcular_erros(demandas_L5, previsoes_rl)

# 4. Organização e Exibição dos Resultados em Tabela
resultados_torneio = [
    {"Técnica": "Média Móvel Simples", 
     "MAD": erros_mms["MAD"], 
     "MSE": erros_mms["MSE"], 
     "MAPE (%)": erros_mms["MAPE"]},
     
    {"Técnica": "Média Exponencial Móvel", 
     "MAD": erros_mem["MAD"], 
     "MSE": erros_mem["MSE"], 
     "MAPE (%)": erros_mem["MAPE"]},
     
    {"Técnica": "Regressão Linear", 
     "MAD": erros_rl["MAD"], 
     "MSE": erros_rl["MSE"], 
     "MAPE (%)": erros_rl["MAPE"]}
]

df_resultados = pd.DataFrame(resultados_torneio)

# Encontrando a técnica campeã (menor MAPE)
melhor_tecnica = df_resultados.loc[df_resultados['MAPE (%)'].idxmin()]

print(df_resultados.to_string(index=False, float_format="{:.2f}".format))
print("-" * 50)
print(f"🏆 TÉCNICA VENCEDORA: {melhor_tecnica['Técnica']} (MAPE: {melhor_tecnica['MAPE (%)']:.2f}%)\n")