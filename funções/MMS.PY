def media_movel_simples(demandas, n):
    # A lista começa com None para os primeiros 'n' meses onde não há histórico suficiente [cite: 1652]
    previsoes = [None] * n
    for t in range(n, len(demandas)):
        janela = demandas[t - n:t]
        previsoes.append(sum(janela) / n)
    return previsoes 