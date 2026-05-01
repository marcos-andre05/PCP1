from statsmodels.tsa.holtwinters import ExponentialSmoothing

def modelo_holt_winters(demandas, periodos_sazonais=12):

    # Consideramos ciclos de 12 meses para a sazonalidade
    modelo = ExponentialSmoothing(demandas, trend='add', seasonal='add', seasonal_periods=periodos_sazonais, initialization_method="estimated")
    modelo_ajustado = modelo.fit()
    previsoes = modelo_ajustado.fittedvalues.tolist()
    
    return {"modelo": modelo_ajustado, "previsoes": previsoes}