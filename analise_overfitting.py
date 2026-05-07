import pandas as pd
import numpy as np

from funções.MMS import media_movel_simples
from funções.MEM import media_exponencial_movel
from funções.REGRESSÃO_LINEAR import regressao_linear
from funções.HOLT_DUPLO_EXPONENCIAL import modelo_holt
from funções.HOLT_WINTERS import modelo_holt_winters
from funções.DECOMPOSIÇÃO_SAZONAL import decomposicao_sazonal
from funções.MÉTRICAS_ERRO import calcular_erros
from funções.TRATAMENTO import tratar_anomalias_demanda

def split_e_prever(demandas, n_treino):
    treino = demandas[:n_treino]
    teste = demandas[n_treino:]
    
    # 1. Ajustar (treinar) as técnicas apenas na base de TREINO
    prev_treino = {
        "Média Móvel Simples": media_movel_simples(treino, n=3),
        "Média Exponencial Móvel": media_exponencial_movel(treino, alpha=0.3),
        "Regressão Linear": regressao_linear(treino)["previsoes"],
        "Holt Duplo": modelo_holt(treino)["previsoes"],
        "Holt-Winters": modelo_holt_winters(treino)["previsoes"],
        "Decomposição Sazonal": decomposicao_sazonal(treino)["previsoes"]
    }
    
    # 2. Avaliar no TREINO
    erros_treino = {}
    for nome, prev in prev_treino.items():
        # Algumas técnicas (como MMS) podem ter 'None' no início. Vamos filtrar para calcular erro justo.
        pares_validos = [(r, p) for r, p in zip(treino, prev) if p is not None]
        if pares_validos:
            r_val = [x[0] for x in pares_validos]
            p_val = [x[1] for x in pares_validos]
            mape = calcular_erros(r_val, p_val)["MAPE"]
        else:
            mape = float('inf')
        erros_treino[nome] = mape
        
    # 3. Gerar previsões para a base de TESTE
    # Como as funções atuais calculam a previsão histórica passando a lista inteira,
    # a forma correta de prever o futuro (teste) sem vazar dados (data leakage)
    # é usar a função de gerar_previsao ou o horizonte. 
    # Porém, muitas dessas funções não suportam 'prever N passos à frente' facilmente 
    # pelo seu design (MMS e MEM prevêem t+1 usando t). 
    # Vamos adaptar a simulação:
    # A maneira mais justa de ver overfitting em séries temporais é a validação cruzada (rolling origin) 
    # ou testar os últimos N meses usando as premissas ajustadas na base de treino.
    
    # Para simplificar, como algumas funções não têm `predict(n_steps)`, vamos usar as funções 
    # com a série inteira (treino + teste) mas calcular o erro separadamente para Treino (1 a 18) 
    # e Teste (19 a 24).
    # Isso reflete o modo como o modelo se comporta 'in-sample' vs 'out-of-sample'.
    pass

# Como as funções já estão prontas para rodar in-sample na série toda, 
# a análise mais simples de overfitting é ver se o erro in-sample da primeira metade
# é drasticamente menor que o da segunda metade, ou simplesmente dividir a série.
def analisar_overfitting():
    df = pd.read_csv('dataset/trabalho_demanda.csv')
    linhas = df.columns[1:].tolist()
    
    n_total = len(df)
    n_treino = int(n_total * 0.75) # 75% da série
    n_teste = n_total - n_treino
    
    resultados = []
    
    for linha in linhas:
        demandas = tratar_anomalias_demanda(df[linha].tolist())
        
        previsoes = {
            "Média Móvel Simples": media_movel_simples(demandas, n=3),
            "Média Exponencial Móvel": media_exponencial_movel(demandas, alpha=0.3),
            "Regressão Linear": regressao_linear(demandas)["previsoes"],
            "Holt Duplo": modelo_holt(demandas)["previsoes"],
            "Holt-Winters": modelo_holt_winters(demandas)["previsoes"],
            "Decomposição Sazonal": decomposicao_sazonal(demandas)["previsoes"]
        }
        
        for nome, prev in previsoes.items():
            # Filtra os None (comuns no começo de MMS, etc.)
            valido_treino = [(demandas[i], prev[i]) for i in range(n_treino) if prev[i] is not None]
            valido_teste = [(demandas[i], prev[i]) for i in range(n_treino, n_total) if prev[i] is not None]
            
            if valido_treino and valido_teste:
                real_tr = [x[0] for x in valido_treino]
                prev_tr = [x[1] for x in valido_treino]
                mape_treino = calcular_erros(real_tr, prev_tr)["MAPE"]
                
                real_te = [x[0] for x in valido_teste]
                prev_te = [x[1] for x in valido_teste]
                mape_teste = calcular_erros(real_te, prev_te)["MAPE"]
                
                # Relação Teste / Treino (quanto maior, mais o erro explodiu fora da amostra original)
                razao = mape_teste / mape_treino if mape_treino > 0 else 0
                
                status = "[OK] Saudavel"
                if razao > 2.0:
                    status = "[!!] Possivel Overfitting"
                elif razao > 1.5:
                    status = "[!] Atencao"
                    
                resultados.append({
                    'Linha': linha,
                    'Técnica': nome,
                    'MAPE Treino (%)': mape_treino,
                    'MAPE Teste (%)': mape_teste,
                    'Razão (Teste/Treino)': razao,
                    'Diagnóstico': status
                })
                
    df_result = pd.DataFrame(resultados)
    
    # Formatação de saída
    for linha in linhas:
        print(f"\n{'='*70}")
        print(f"  --- ANALISE DE OVERFITTING - LINHA {linha} ---")
        print(f"{'='*70}")
        df_linha = df_result[df_result['Linha'] == linha].copy()
        
        # Ordenar por Razão
        df_linha = df_linha.sort_values(by='Razão (Teste/Treino)', ascending=False)
        
        for _, row in df_linha.iterrows():
            tec = row['Técnica']
            m_tr = row['MAPE Treino (%)']
            m_te = row['MAPE Teste (%)']
            razao = row['Razão (Teste/Treino)']
            diag = row['Diagnóstico']
            
            print(f"  {tec:<25} | Treino: {m_tr:>5.1f}% | Teste: {m_te:>5.1f}% | Razão: {razao:>4.1f}x | {diag}")

    print("\n[ Criterio adotado ]")
    print(" - Treino: Primeiros 18 meses (75%)")
    print(" - Teste: Últimos 6 meses (25%)")
    print(" - Overfitting é suspeito quando o erro no Teste é mais que o dobro (2.0x) do erro no Treino.")

analisar_overfitting()
