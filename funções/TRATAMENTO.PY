
import pandas as pd
import numpy as np

def tratar_anomalias_demanda(demandas, fator_iqr=1.5, janela_shift=6, limite_shift=0.40):
    """
    Trata outliers (método IQR) e level shifts em uma série de demanda.
    Retorna uma nova lista com os dados ajustados.

    Método IQR:
        Limites = [Q1 - fator*IQR , Q3 + fator*IQR]
        Valores fora dos limites são substituídos pela mediana global da série.
    """
    df = pd.Series(demandas, dtype=float)

    # ==========================================
    # 1. TRATAMENTO DE OUTLIERS (Método IQR)
    # ==========================================
    Q1  = df.quantile(0.25)
    Q3  = df.quantile(0.75)
    IQR = Q3 - Q1

    limite_inf = Q1 - fator_iqr * IQR
    limite_sup = Q3 + fator_iqr * IQR

    outliers = (df < limite_inf) | (df > limite_sup)

    # Substitui os outliers pela mediana da série
    mediana = df.median()
    df_sem_outliers = df.copy()
    df_sem_outliers[outliers] = mediana

    # ==========================================
    # 2. TRATAMENTO DE LEVEL SHIFT
    # ==========================================
    df_tratado = df_sem_outliers.copy()

    # Varre a série comparando janelas adjacentes para achar a quebra de patamar
    for i in range(janela_shift, len(df_tratado) - janela_shift):
        media_ant  = df_tratado[i - janela_shift:i].mean()
        media_post = df_tratado[i:i + janela_shift].mean()

        # Se a média mudar mais que o limite (ex: 40% = 0.40) de repente
        if media_ant > 0 and abs(media_post - media_ant) / media_ant > limite_shift:
            # Ajusta todo o passado ao novo patamar
            fator_ajuste = media_post / media_ant
            df_tratado[:i] = (df_tratado[:i] * fator_ajuste).round()
            break   # apenas o primeiro shift relevante

    # Arredonda e converte para lista de inteiros
    return df_tratado.round(0).astype(int).tolist()


def analisar_anomalias(demandas, fator_iqr=1.5, janela_shift=6, limite_shift=0.40):
    """
    Detecta outliers (método IQR) e level shifts SEM modificar a série.
    Retorna um relatório com o que foi encontrado.

    Parâmetros:
        fator_iqr    : multiplicador do IQR para definir os limites (padrão 1.5)
        janela_shift : tamanho da janela para detectar level shift
        limite_shift : variação percentual mínima para considerar level shift
    """
    df = pd.Series(demandas, dtype=float)

    # --- Detecção de outliers por IQR ---
    Q1  = df.quantile(0.25)
    Q3  = df.quantile(0.75)
    IQR = Q3 - Q1

    limite_inf = Q1 - fator_iqr * IQR
    limite_sup = Q3 + fator_iqr * IQR

    outliers         = (df < limite_inf) | (df > limite_sup)
    n_outliers       = int(outliers.sum())
    indices_outliers = [int(i) for i in df[outliers].index]

    # --- Detecção de level shift ---
    level_shift = False
    ponto_shift = None
    for i in range(janela_shift, len(df) - janela_shift):
        media_ant  = df[i - janela_shift:i].mean()
        media_post = df[i:i + janela_shift].mean()
        if media_ant > 0 and abs(media_post - media_ant) / media_ant > limite_shift:
            level_shift = True
            ponto_shift = int(i)
            break

    return {
        'n_outliers':       n_outliers,
        'indices_outliers': indices_outliers,   # posições 0-indexadas
        'limite_inf':       round(limite_inf, 1),
        'limite_sup':       round(limite_sup, 1),
        'mediana':          round(df.median(), 1),
        'level_shift':      level_shift,
        'ponto_shift':      ponto_shift,        # posição 0-indexada, ou None
        'media_ant':        round(media_ant, 1) if level_shift else None,
        'media_post':       round(media_post, 1) if level_shift else None
    }