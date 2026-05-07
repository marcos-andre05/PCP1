from datetime import datetime
from dateutil.relativedelta import relativedelta

def gerar_meses_futuros(ultimo_mes_str, horizonte):
    """
    Lê o último mês no formato "mmm/yy" (ex: "dez/25") e retorna uma lista
    com os próximos 'horizonte' meses gerados dinamicamente no mesmo formato.
    """
    meses_pt = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6,
                'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}
    meses_inv = {v: k for k, v in meses_pt.items()}
    
    mes_str, ano_str = ultimo_mes_str.split('/')
    mes_num = meses_pt[mes_str.lower()]
    ano_num = 2000 + int(ano_str)
    
    dt = datetime(ano_num, mes_num, 1)
    futuros = []
    for _ in range(horizonte):
        dt += relativedelta(months=1)
        futuros.append(f"{meses_inv[dt.month]}/{str(dt.year)[-2:]}")
    return futuros

def obter_cores_dinamicas(num_cores):
    """
    Gera um dicionário ou lista de cores dinâmicas a partir da paleta tab10.
    """
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab10')
    return [cmap(i % 10) for i in range(num_cores)]
