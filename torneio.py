import pandas as pd
from funções.MMS import media_movel_simples
from funções.MEM import media_exponencial_movel
from funções.REGRESSÃO_LINEAR import regressao_linear
from funções.HOLT_DUPLO_EXPONENCIAL import modelo_holt
from funções.HOLT_WINTERS import modelo_holt_winters
from funções.DECOMPOSIÇÃO_SAZONAL import decomposicao_sazonal
from funções.MÉTRICAS_ERRO import calcular_erros
from funções.TRACKING_SIGNAL import calcular_tracking_signal

def executar_torneio(demandas, n_mms=3, alpha=0.3):
    """
    Executa o torneio completo: roda as 6 técnicas, calcula erros e TS,
    e seleciona a melhor técnica não enviesada.
    """
    # 1. Rodar todas as técnicas
    previsoes = {
        "Média Móvel Simples": media_movel_simples(demandas, n=n_mms),
        "Média Exponencial Móvel": media_exponencial_movel(demandas, alpha=alpha),
        "Regressão Linear": regressao_linear(demandas)["previsoes"],
        "Holt Duplo": modelo_holt(demandas)["previsoes"],
        "Holt-Winters": modelo_holt_winters(demandas)["previsoes"],
        "Decomposição Sazonal": decomposicao_sazonal(demandas)["previsoes"]
    }
    
    # 2. Calcular erros e Tracking Signal
    resultados = []
    ts_resultados = {}
    for nome, prev in previsoes.items():
        erros = calcular_erros(demandas, prev)
        ts = calcular_tracking_signal(demandas, prev)
        ts_resultados[nome] = ts
        resultados.append({
            "Técnica": nome,
            "MAD": erros["MAD"],
            "MSE": erros["MSE"],
            "MAPE (%)": erros["MAPE"],
            "TS": ts["TS_final"],
            "Enviesado": "Sim" if ts["enviesado"] else "Não"
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # 3. Selecionar a melhor técnica NÃO enviesada
    df_ok = df_resultados[df_resultados["Enviesado"] == "Não"]
    todas_enviesadas = df_ok.empty
    
    if not todas_enviesadas:
        melhor = df_ok.loc[df_ok['MAPE (%)'].idxmin()]
    else:
        df_resultados["TS_abs"] = df_resultados["TS"].abs()
        melhor = df_resultados.loc[df_resultados['TS_abs'].idxmin()]
    
    nome_vencedora = melhor['Técnica']
    
    return {
        "tabela": df_resultados,
        "vencedora": nome_vencedora,
        "mape": melhor['MAPE (%)'],
        "ts_final": melhor['TS'],
        "ts_detalhes": ts_resultados[nome_vencedora],
        "previsoes": previsoes,
        "todas_enviesadas": todas_enviesadas
    }


def exibir_resultado(resultado):
    """
    Exibe o resultado do torneio formatado no console.
    """
    print(resultado["tabela"].to_string(index=False, float_format="{:.2f}".format))
    print("-" * 60)
    
    if resultado["todas_enviesadas"]:
        print("⚠️  Todas as técnicas estão enviesadas! Escolhendo a menos enviesada:")
    
    print(f"🏆 TÉCNICA VENCEDORA: {resultado['vencedora']} "
          f"(MAPE: {resultado['mape']:.2f}% | TS: {resultado['ts_final']:.2f})")
    
    ts = resultado["ts_detalhes"]
    print(f"\n--- SINAL DE RASTREAMENTO: {resultado['vencedora']} ---")
    print(f"TS Final:      {ts['TS_final']:.2f}  (Limite: ±{ts['limite']})")
    print(f"RSFE Final:    {ts['RSFE_final']:.2f}")
    print(f"MAD Final:     {ts['MAD_final']:.2f}")
    print(f"Violações:     {ts['violacoes']}/{ts['total_periodos']} períodos")
    print(f"TS por período: {[round(float(v), 2) for v in ts['TS_por_periodo']]}")
    if ts['enviesado']:
        print("⚠️  RESULTADO: Previsão ENVIESADA — o modelo precisa de ajuste!")
    else:
        print("✅ RESULTADO: Previsão SEM VIÉS — o modelo é confiável!")
