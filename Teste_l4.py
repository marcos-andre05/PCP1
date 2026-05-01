import pandas as pd
from funções.REGRESSÃO_LINEAR import regressao_linear
from funções.HOLT_DUPLO_EXPONENCIAL import modelo_holt
from funções.MEM import media_exponencial_movel
from funções.MÉTRICAS_ERRO import calcular_erros
from funções.TRATAMENTO import tratar_anomalias_demanda 

# 1. Carregamento dos Dados (Exemplo com os 24 meses da Linha 4)
df = pd.read_csv('dataset/trabalho_demanda.csv')
demandas_L4 = df['L4'].tolist()

print("--- INICIANDO O TORNEIO DE TÉCNICAS: LINHA 4 ---\n")

demandas_L4 = tratar_anomalias_demanda(demandas_L4)
# 2. Aplicação das 3 Técnicas Escolhidas (Apenas nos 24 meses históricos)
# Técnica 1: Regressão Linear
resultado_regressao = regressao_linear(demandas_L4)
previsoes_regressao = resultado_regressao["previsoes"]

# Técnica 2: Holt Duplo
resultado_holt = modelo_holt(demandas_L4)
previsoes_holt = resultado_holt["previsoes"]

# Técnica 3: Média Exponencial Móvel
previsoes_mem = media_exponencial_movel(demandas_L4, alpha=0.3)

# 3. Cálculo das Métricas de Erro usando o seu arquivo independente
erros_regressao = calcular_erros(demandas_L4, previsoes_regressao)
erros_holt = calcular_erros(demandas_L4, previsoes_holt)
erros_mem = calcular_erros(demandas_L4, previsoes_mem)

# 4. Organização e Exibição dos Resultados em Tabela
resultados_torneio = [
    {"Técnica": "Regressão Linear", 
     "MAD": erros_regressao["MAD"], 
     "MSE": erros_regressao["MSE"], 
     "MAPE (%)": erros_regressao["MAPE"]},
     
    {"Técnica": "Holt Duplo", 
     "MAD": erros_holt["MAD"], 
     "MSE": erros_holt["MSE"], 
     "MAPE (%)": erros_holt["MAPE"]},
     
    {"Técnica": "Média Exponencial Móvel", 
     "MAD": erros_mem["MAD"], 
     "MSE": erros_mem["MSE"], 
     "MAPE (%)": erros_mem["MAPE"]}
]

df_resultados = pd.DataFrame(resultados_torneio)

# Encontrando a técnica campeã (menor MAPE)
melhor_tecnica = df_resultados.loc[df_resultados['MAPE (%)'].idxmin()]

print(df_resultados.to_string(index=False, float_format="{:.2f}".format))
print("-" * 50)
print(f"🏆 TÉCNICA VENCEDORA: {melhor_tecnica['Técnica']} (MAPE: {melhor_tecnica['MAPE (%)']:.2f}%)\n")