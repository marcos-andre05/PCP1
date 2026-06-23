import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from torneio import executar_torneio, exibir_resultado
from funções.TRATAMENTO import tratar_anomalias_demanda

# Carregamento e identificação de todas as linhas de produção
df = pd.read_csv('dataset/trabalho_demanda.csv')
linhas = df.columns[1:].tolist()

print(f"============================================================")
print(f"  🏆 INICIANDO TORNEIO DE TÉCNICAS ({len(linhas)} linhas identificadas)  ")
print(f"============================================================\n")

for linha in linhas:
    demandas_originais = df[linha].tolist()
    demandas_tratadas = tratar_anomalias_demanda(demandas_originais)
    
    print(f"--- ANALISANDO A LINHA: {linha} ---")
    resultado = executar_torneio(demandas_tratadas, n_mms=3, alpha=0.3)
    exibir_resultado(resultado)
    print("\n\n")
