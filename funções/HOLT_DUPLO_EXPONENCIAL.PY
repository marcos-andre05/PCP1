from statsmodels.tsa.holtwinters import ExponentialSmoothing
def modelo_holt(demandas):
    # 'add' significa tendência aditiva
    modelo = ExponentialSmoothing(demandas, trend='add', seasonal=None, initialization_method="estimated")
    modelo_ajustado = modelo.fit()
    previsoes = modelo_ajustado.fittedvalues.tolist()
    
    return {"modelo": modelo_ajustado, "previsoes": previsoes}