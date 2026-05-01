
import pandas as pd
import numpy as np

def tratar_anomalias_demanda(demandas, janela_outlier=5, limite_z=2.0, janela_shift=6, limite_shift=0.40):
    """
    Trata outliers e level shifts em uma série de demanda.
    Retorna uma nova lista com os dados suavizados e ajustados.
    """
    # Converte a lista para uma Série do Pandas para facilitar os cálculos
    df = pd.Series(demandas)
    
    # ==========================================
    # 1. TRATAMENTO DE OUTLIERS
    # ==========================================
    # Calcula a média e desvio padrão de uma janela móvel (ex: 5 meses)
    media_movel = df.rolling(window=janela_outlier, center=True, min_periods=1).mean()
    desvio_padrao = df.rolling(window=janela_outlier, center=True, min_periods=1).std()
    
    # Preenche possíveis valores nulos no desvio padrão (comum nas pontas)
    desvio_padrao = desvio_padrao.bfill().ffill()
    
    # Se a demanda fugir muito do padrão local (Z-score > limite), é um outlier
    # Tratamos o aviso de divisão por zero caso o desvio seja 0
    z_scores = np.abs((df - media_movel) / desvio_padrao.replace(0, 1e-10))
    outliers = z_scores > limite_z
    
    # Substitui os outliers pela média da janela local
    df_sem_outliers = df.copy()
    df_sem_outliers[outliers] = media_movel[outliers]
    
    # ==========================================
    # 2. TRATAMENTO DE LEVEL SHIFT
    # ==========================================
    df_tratado = df_sem_outliers.copy()
    
    # Varre a série comparando janelas adjacentes para achar a quebra de patamar
    for i in range(janela_shift, len(df_tratado) - janela_shift):
        media_ant = df_tratado[i-janela_shift:i].mean()
        media_post = df_tratado[i:i+janela_shift].mean()
        
        # Se a média mudar mais que o limite (ex: 40% = 0.40) de repente
        if media_ant > 0 and abs(media_post - media_ant) / media_ant > limite_shift:
            # Calcula o fator de ajuste baseado no novo patamar
            fator_ajuste = media_post / media_ant
            
            # Ajusta todo o passado para se alinhar à nova realidade
            # Altere a linha 48 para arredondar o resultado
            df_tratado[:i] = (df_tratado[:i] * fator_ajuste).round()
            
            # Paramos no primeiro grande shift encontrado para evitar distorções sucessivas
            break 
            
    # Arredonda os valores finais para não termos "meio produto" e converte de volta para lista
    lista_final = df_tratado.round(0).astype(int).tolist()
    
    return lista_final