def regressao_linear(demandas):
    n = len(demandas)
    X = list(range(1, n + 1))
    soma_xy = sum(x * y for x, y in zip(X, demandas))
    soma_x2 = sum(x ** 2 for x in X)
    
    b = (n * soma_xy - sum(X) * sum(demandas)) / (n * soma_x2 - sum(X) ** 2)
    a = (sum(demandas) - b * sum(X)) / n
    
    # Gerando os pontos da reta para calcular o erro depois
    previsoes = [(a + b * x) for x in X]
    
    return {"a": a, "b": b, "previsoes": previsoes}