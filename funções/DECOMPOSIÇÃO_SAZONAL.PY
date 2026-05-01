from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

def decomposicao_sazonal(demandas, periodos_sazonais=12):
    # Decompõe a série
    decomposicao = seasonal_decompose(demandas, model='multiplicative', period=periodos_sazonais)
    
    tendencia = decomposicao.trend
    sazonalidade = decomposicao.seasonal
    
    # A previsão aqui é a recombinação da tendência com a sazonalidade
    previsoes = []
    for i in range(len(demandas)):
        if not np.isnan(tendencia[i]):
            previsoes.append(tendencia[i] * sazonalidade[i])
        else:
            previsoes.append(None)
            
    return {"tendencia": tendencia.tolist(), "sazonalidade": sazonalidade.tolist(), "previsoes": previsoes}