from statsmodels.tsa.holtwinters import ExponentialSmoothing

def modelo_holt_winters(demandas, periodos_sazonais=12, tipo_tendencia='add', tipo_sazonal='add'):
    """
    Holt-Winters: Ideal para Séries com Tendência e Sazonalidade (ex: L1, L3 e L4)
    """
    # Ajustamos o modelo com base nos parâmetros passados
    # trend='add' para crescimento constante, 'mul' para crescimento proporcional
    modelo = ExponentialSmoothing(
        demandas, 
        trend=tipo_tendencia, 
        seasonal=tipo_sazonal, 
        seasonal_periods=periodos_sazonais, 
        initialization_method="estimated"
    )
    
    modelo_ajustado = modelo.fit()
    
    # fittedvalues são as previsões dentro da amostra histórica (24 meses)
    previsoes = modelo_ajustado.fittedvalues.tolist()
    
    return {"modelo": modelo_ajustado, "previsoes": previsoes}

def prever_holt_winters(demandas, periodos_sazonais, horizonte):
    modelo = ExponentialSmoothing(demandas, trend='add', seasonal='add', seasonal_periods=periodos_sazonais, initialization_method='estimated')
    ajustado = modelo.fit()
    return [round(v) for v in ajustado.forecast(horizonte)]

