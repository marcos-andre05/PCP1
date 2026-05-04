def calcular_mad(real, previsao):
    """Calcula o MAD (Desvio Absoluto Médio), ignorando valores nulos (None)."""
    erros = []
    for r, p in zip(real, previsao):
        if p is not None:  # Ignora os meses em branco da MMS ou Holt-Winters
            erros.append(abs(r - p))
    # Prevenção contra divisão por zero
    if len(erros) == 0: return 0 
    return sum(erros) / len(erros)