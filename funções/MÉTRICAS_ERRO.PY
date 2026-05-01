import numpy as np

def calcular_erros(y_true, y_pred):
    """
    Calcula as métricas de erro exigidas para o trabalho de PCP: MAD, MSE e MAPE.
    Filtra automaticamente valores None ou NaN (comuns no início das previsões).
    """
    # Convertendo as listas para arrays do numpy para facilitar a matemática
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Criando uma "máscara" para remover os meses onde a previsão é vazia (None/NaN)
    mascara_validos = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_true_filtrado = y_true[mascara_validos]
    y_pred_filtrado = y_pred[mascara_validos]
    
    # Se a filtragem remover tudo, retorna vazio para não quebrar o código
    if len(y_true_filtrado) == 0:
        return {"MAD": None, "MSE": None, "MAPE": None}
        
    # 1. MAD (Desvio Absoluto Médio) - Mede o erro absoluto médio de forma direta
    mad = np.mean(np.abs(y_true_filtrado - y_pred_filtrado))
    
    # 2. MSE (Erro Quadrático Médio) - Penaliza mais fortemente os erros grandes
    mse = np.mean((y_true_filtrado - y_pred_filtrado) ** 2)
    
    # 3. MAPE (Erro Percentual Absoluto Médio) - Facilita a comparação percentual
    mape = np.mean(np.abs((y_true_filtrado - y_pred_filtrado) / y_true_filtrado)) * 100
    
    return {
        "MAD": mad,
        "MSE": mse,
        "MAPE": mape
    }