def media_exponencial_movel(demandas, alpha, m0=None):
    if m0 is None:
        m0 = demandas[0]
    previsoes = [m0]
    for t in range(1, len(demandas)):
        m_t = alpha * demandas[t - 1] + (1 - alpha) * previsoes[-1]
        previsoes.append(m_t)
    return previsoes

def prever_media_exponencial_movel(demandas, alpha, horizonte):
    n = len(demandas)
    m = demandas[0]
    for t in range(1, n):
        m = alpha * demandas[t - 1] + (1 - alpha) * m
    m_final = alpha * demandas[-1] + (1 - alpha) * m
    return [round(m_final)] * horizonte

