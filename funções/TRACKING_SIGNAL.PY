import numpy as np

def calcular_tracking_signal(y_true, y_pred, limite=3):
    """
    Calcula o Sinal de Rastreamento (Tracking Signal) para validar viés da previsão.
    
    Fórmula:
        TS(t) = RSFE(t) / MAD(t)
        RSFE = Soma Acumulada dos Erros de Previsão (com sinal)
        MAD  = Desvio Absoluto Médio Acumulado
    
    Interpretação:
        Se |TS| > limite (geralmente ±3), a previsão está enviesada.
        TS positivo = previsão subestima (demanda real > previsão)
        TS negativo = previsão superestima (demanda real < previsão)
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Filtra valores NaN/None
    mascara = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true_f = y_true[mascara]
    y_pred_f = y_pred[mascara]
    
    if len(y_true_f) == 0:
        return None
    
    # Erros de previsão com sinal: e(t) = D(t) - P(t)
    erros = y_true_f - y_pred_f
    
    # Cálculos acumulados período a período
    rsfe_lista = []    # RSFE acumulado
    mad_lista = []     # MAD acumulado
    ts_lista = []      # Tracking Signal
    
    soma_erros = 0
    soma_abs_erros = 0
    
    for t in range(len(erros)):
        soma_erros += erros[t]           # RSFE = Σe(t)
        soma_abs_erros += abs(erros[t])  # Σ|e(t)|
        
        rsfe_lista.append(soma_erros)
        mad_t = soma_abs_erros / (t + 1)  # MAD acumulado
        mad_lista.append(mad_t)
        
        if mad_t != 0:
            ts_lista.append(soma_erros / mad_t)
        else:
            ts_lista.append(0)
    
    # Verificação de viés
    ts_final = ts_lista[-1]
    enviesado = abs(ts_final) > limite
    
    # Verificar se houve violação em algum período
    violacoes = sum(1 for ts in ts_lista if abs(ts) > limite)
    
    return {
        "TS_final": ts_final,
        "RSFE_final": rsfe_lista[-1],
        "MAD_final": mad_lista[-1],
        "TS_por_periodo": ts_lista,
        "enviesado": enviesado,
        "violacoes": violacoes,
        "total_periodos": len(ts_lista),
        "limite": limite
    }
