from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

def decomposicao_sazonal(demandas, periodos_sazonais=12):
    # 1. Decompõe a série histórica
    decomposicao = seasonal_decompose(demandas, model='multiplicative', period=periodos_sazonais)
    
    tendencia = decomposicao.trend
    sazonalidade = decomposicao.seasonal
    
    # 2. Extrair a última tendência válida (removendo NaNs gerados pela média móvel interna)
    tendencia_valida = [x for x in tendencia if not np.isnan(x)]
    ultima_tendencia = tendencia_valida[-1] if tendencia_valida else np.mean(demandas)
    
    previsoes = []
    for i in range(len(demandas)):
        # Simulamos a previsão: usamos a tendência para projetar o valor
        # mas aplicamos a sazonalidade que o modelo identificou
        if not np.isnan(tendencia[i]):
            # Aqui ainda é ajuste, mas usamos a lógica de decomposição
            previsoes.append(tendencia[i] * sazonalidade[i])
        else:
            # Para os meses iniciais/finais onde a tendência falha, 
            # usamos a última tendência conhecida (extrapolação simples)
            previsoes.append(ultima_tendencia * sazonalidade[i])
            
    return {
        "tendencia": tendencia.tolist(), 
        "sazonalidade": sazonalidade.tolist(), 
        "previsoes": previsoes
    }

def prever_decomposicao_sazonal_futuro(demandas, periodos_sazonais, horizonte):
    n = len(demandas)
    decomp = seasonal_decompose(demandas, model='multiplicative', period=periodos_sazonais)
    tendencia = decomp.trend
    sazonalidade = decomp.seasonal
    tend_validos = [(i, v) for i, v in enumerate(tendencia) if not np.isnan(v)]
    if len(tend_validos) >= 2:
        x1, y1 = tend_validos[-2]
        x2, y2 = tend_validos[-1]
        slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        ultima_tend = y2
    else:
        slope = 0
        ultima_tend = tend_validos[-1][1] if tend_validos else np.mean(demandas)
    previsoes = []
    for h in range(1, horizonte + 1):
        idx_sazonal = (n + h - 1) % periodos_sazonais
        tend_extrapolada = ultima_tend + slope * h
        previsoes.append(round(tend_extrapolada * sazonalidade[idx_sazonal]))
    return previsoes

