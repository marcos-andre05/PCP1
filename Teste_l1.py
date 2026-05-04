import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from torneio import executar_torneio, exibir_resultado
from funções.TRATAMENTO import tratar_anomalias_demanda

# Carregamento e tratamento dos dados
df = pd.read_csv('dataset/trabalho_demanda.csv')
demandas_L1 = tratar_anomalias_demanda(df['L1'].tolist())

# Torneio de técnicas
print("--- INICIANDO O TORNEIO DE TÉCNICAS: LINHA 1 ---\n")
resultado = executar_torneio(demandas_L1, n_mms=3, alpha=0.3)
exibir_resultado(resultado)