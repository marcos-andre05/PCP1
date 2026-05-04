from funções.MMS import prever_media_movel_simples
from funções.MEM import prever_media_exponencial_movel
from funções.REGRESSÃO_LINEAR import prever_regressao_linear_futuro
from funções.HOLT_DUPLO_EXPONENCIAL import prever_holt_duplo
from funções.HOLT_WINTERS import prever_holt_winters
from funções.DECOMPOSIÇÃO_SAZONAL import prever_decomposicao_sazonal_futuro

def gerar_previsao(tecnica, demandas, horizonte=7, **params):
    """
    Gera previsões futuras para o horizonte especificado usando a técnica indicada,
    chamando diretamente a função correspondente em cada módulo.
    """
    if tecnica == "Média Móvel Simples":
        n_mms = params.get('n_mms', 3)
        return prever_media_movel_simples(demandas, n_mms, horizonte)
    
    elif tecnica == "Média Exponencial Móvel":
        alpha = params.get('alpha', 0.3)
        return prever_media_exponencial_movel(demandas, alpha, horizonte)
    
    elif tecnica == "Regressão Linear":
        return prever_regressao_linear_futuro(demandas, horizonte)
    
    elif tecnica == "Holt Duplo":
        return prever_holt_duplo(demandas, horizonte)
    
    elif tecnica == "Holt-Winters":
        periodos = params.get('periodos_sazonais', 12)
        return prever_holt_winters(demandas, periodos, horizonte)
    
    elif tecnica == "Decomposição Sazonal":
        periodos = params.get('periodos_sazonais', 12)
        return prever_decomposicao_sazonal_futuro(demandas, periodos, horizonte)
    
    else:
        raise ValueError(f"Técnica desconhecida: {tecnica}")
