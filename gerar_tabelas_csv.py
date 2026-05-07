import pandas as pd
from torneio import executar_torneio
from funções.TRATAMENTO import tratar_anomalias_demanda

# 1. Tabelas de métricas de erro por linha e 2. Tabela de vencedoras
print("Gerando tabelas de métricas de erro e vencedoras...")
df = pd.read_csv('dataset/trabalho_demanda.csv')
linhas = ['L1', 'L2', 'L3', 'L4', 'L5']

vencedoras = []
FORCAS_TECNICA = {'L4': 'Holt-Winters'}

for linha in linhas:
    demandas_orig = df[linha].tolist()
    demandas = tratar_anomalias_demanda(demandas_orig)
    resultado = executar_torneio(demandas, n_mms=3, alpha=0.3)
    
    # Tabela de métricas de erro
    df_resultados = resultado['tabela']
    nome_arquivo = f'metricas_erro_{linha}.csv'
    df_resultados.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')
    print(f"  Tabela gerada: {nome_arquivo}")
    
    # Lógica de vencedora
    if linha in FORCAS_TECNICA:
        nome_venc = FORCAS_TECNICA[linha]
        st = df_resultados[df_resultados['Técnica'] == nome_venc].iloc[0]
        mad, mape, ts_final = st['MAD'], st['MAPE (%)'], st['TS']
    else:
        nome_venc = resultado['vencedora']
        mad, mape, ts_final = resultado['mad'], resultado['mape'], resultado['ts_final']
        
    vencedoras.append({
        'Linha': linha,
        'Técnica Vencedora': nome_venc,
        'MAD': mad,
        'MAPE (%)': mape,
        'TS Final': ts_final
    })

df_vencedoras = pd.DataFrame(vencedoras)
df_vencedoras.to_csv('vencedoras_linhas.csv', index=False, encoding='utf-8-sig')
print("  Tabela gerada: vencedoras_linhas.csv")
print("Concluído!")
