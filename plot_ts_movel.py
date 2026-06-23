import pandas as pd
import matplotlib.pyplot as plt

from funções.TRATAMENTO import tratar_anomalias_demanda
from torneio import executar_torneio
from funções.UTILS import obter_cores_dinamicas

def plotar_ts_movel():
    df = pd.read_csv('dataset/trabalho_demanda.csv')
    df_param = pd.read_csv('dataset/trabalho_parametros.csv', index_col=0)
    
    linhas = df.columns[1:].tolist()
    produtos_map = df_param.loc['produtos'].to_dict()
    nomes_linhas = [f"{col}: {produtos_map.get(col, 'Produto ' + col)}" for col in linhas]
    
    cores = obter_cores_dinamicas(len(linhas))
    
    fig, axes = plt.subplots(nrows=len(linhas), ncols=1, figsize=(12, 3.5 * len(linhas)), sharex=True)
    if len(linhas) == 1: axes = [axes]
    
    fig.suptitle('Monitoramento do Tracking Signal (Sinal de Rastreamento) - 24 Meses', fontsize=16, y=0.92)
    
    for i, (linha, ax) in enumerate(zip(linhas, axes)):
        demandas_brutas = df[linha].tolist()
        demandas_tratadas = tratar_anomalias_demanda(demandas_brutas)
        
        # Executa o torneio para obter a técnica vencedora
        resultado = executar_torneio(demandas_tratadas, n_mms=3, alpha=0.3)
        vencedora = resultado["vencedora"]
        ts_por_periodo = resultado["ts_detalhes"]["TS_por_periodo"]
        
        # O torneio filtra None no início de algumas séries (como MMS), então 
        # o tamanho do TS pode ser menor que 24.
        # Precisamos alinhar os índices.
        ts_full = [None] * len(demandas_tratadas)
        offset = len(demandas_tratadas) - len(ts_por_periodo)
        for j, ts_val in enumerate(ts_por_periodo):
            ts_full[offset + j] = ts_val
        
        ax.plot(range(1, len(ts_full) + 1), ts_full, marker='o', linestyle='-', color=cores[i], linewidth=2, label=vencedora)
        
        # Limites aceitáveis (±3)
        ax.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='+3 (Subestimação Máx)')
        ax.axhline(y=-3, color='red', linestyle='--', alpha=0.7, label='-3 (Superestimação Máx)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title(nomes_linhas[i] + f' - Técnica: {vencedora}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tracking Signal', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)
        
        # Opcional: preencher a zona fora do limite
        ax.fill_between(range(1, len(ts_full) + 1), 3, max(max([v for v in ts_full if v is not None] + [3]), 4), color='red', alpha=0.1)
        ax.fill_between(range(1, len(ts_full) + 1), -3, min(min([v for v in ts_full if v is not None] + [-3]), -4), color='red', alpha=0.1)

    axes[-1].set_xlabel('Períodos (Meses)', fontsize=12)
    axes[-1].set_xticks(range(1, len(ts_full) + 1))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    plt.savefig('grafico_ts_movel.png', dpi=300, bbox_inches='tight')
    print("Gráfico de Tracking Signal móvel gerado e salvo como 'grafico_ts_movel.png'.")

if __name__ == "__main__":
    plotar_ts_movel()
