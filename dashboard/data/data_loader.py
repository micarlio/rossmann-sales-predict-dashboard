import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import time
import functools

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard_data_loader.log')
    ]
)

# Diretórios e caminhos globais
# O arquivo está em *dashboard/data/* → precisamos subir **duas** pastas para
# chegar à raiz do repositório, onde se encontra a pasta ``dataset``.
DIRETORIO_BASE = Path(__file__).resolve().parents[2]
DIRETORIO_DADOS = DIRETORIO_BASE / "dataset"

CAMINHO_ARQUIVO_TREINO_BRUTO = DIRETORIO_DADOS / "brutos" / "train.parquet"
CAMINHO_ARQUIVO_LOJAS_BRUTO = DIRETORIO_DADOS / "brutos" / "store.parquet"
CAMINHO_ARQUIVO_PROCESSADO = DIRETORIO_DADOS / "processados" / "df_completo_processado.parquet"

N_AMOSTRAS_PADRAO = 50
_principal_cache = {}

# --------------------------------------------------------------------------------------
# Funções utilitárias de carga e processamento
# --------------------------------------------------------------------------------------

def verificar_diretorios():
    for d in [
        DIRETORIO_DADOS / "brutos",
        DIRETORIO_DADOS / "processados",
        DIRETORIO_DADOS / "reduzidos",
    ]:
        d.mkdir(parents=True, exist_ok=True)


def reduzir_uso_memoria(df: pd.DataFrame, nome_df: str = "DataFrame") -> pd.DataFrame:
    inicio = time.time()
    uso_antes = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        tipo = str(df[col].dtype)
        if tipo == "object":
            if len(df[col].unique()) / len(df[col]) < 0.5:
                df[col] = df[col].astype("category")
        elif "int" in tipo:
            min_, max_ = df[col].min(), df[col].max()
            if min_ >= 0:
                if max_ < 2 ** 8:
                    df[col] = df[col].astype(np.uint8)
                elif max_ < 2 ** 16:
                    df[col] = df[col].astype(np.uint16)
                elif max_ < 2 ** 32:
                    df[col] = df[col].astype(np.uint32)
            else:
                if min_ > -2 ** 7 and max_ < 2 ** 7:
                    df[col] = df[col].astype(np.int8)
                elif min_ > -2 ** 15 and max_ < 2 ** 15:
                    df[col] = df[col].astype(np.int16)
                elif min_ > -2 ** 31 and max_ < 2 ** 31:
                    df[col] = df[col].astype(np.int32)
        elif "float" in tipo:
            df[col] = df[col].astype(np.float32)

    uso_depois = df.memory_usage(deep=True).sum() / 1024 ** 2
    logging.info(
        f"Otimização de memória para {nome_df}: Antes: {uso_antes:.2f} MB | "
        f"Depois: {uso_depois:.2f} MB | Redução: {(1 - uso_depois / uso_antes) * 100:.2f}%"
    )
    logging.info(f"  - Tempo: {time.time() - inicio:.2f} segundos")
    return df


def carregar_dados_brutos():
    try:
        logging.info("Carregando dados brutos em Parquet...")
        if not CAMINHO_ARQUIVO_TREINO_BRUTO.exists() or not CAMINHO_ARQUIVO_LOJAS_BRUTO.exists():
            logging.error("Arquivos Parquet não encontrados.")
            return None, None
        df_vendas = pd.read_parquet(CAMINHO_ARQUIVO_TREINO_BRUTO)
        df_vendas["Date"] = pd.to_datetime(df_vendas["Date"])
        df_lojas = pd.read_parquet(CAMINHO_ARQUIVO_LOJAS_BRUTO)
        df_vendas = reduzir_uso_memoria(df_vendas, "df_vendas")
        df_lojas = reduzir_uso_memoria(df_lojas, "df_lojas")
        logging.info(
            f"Dados brutos carregados com sucesso. Vendas: {len(df_vendas)} registros, "
            f"Lojas: {len(df_lojas)} registros"
        )
        return df_vendas, df_lojas
    except Exception as e:
        logging.error(f"Erro ao carregar dados brutos: {e}")
        return None, None


@functools.lru_cache(maxsize=2)
def processar_dados_brutos(force_reprocess: bool = False):
    try:
        if CAMINHO_ARQUIVO_PROCESSADO.exists() and not force_reprocess:
            logging.info(f"Carregando arquivo processado: {CAMINHO_ARQUIVO_PROCESSADO}")
            return pd.read_parquet(CAMINHO_ARQUIVO_PROCESSADO)

        df_vendas, df_lojas = carregar_dados_brutos()
        if df_vendas is None:
            return None

        df_vendas = df_vendas[df_vendas["Open"] == 1].copy()
        if "PromoInterval" in df_lojas.columns and str(df_lojas["PromoInterval"].dtype).startswith("category"):
            df_lojas["PromoInterval"] = df_lojas["PromoInterval"].astype(str)
        df_lojas["PromoInterval"].fillna("Nenhum", inplace=True)

        for col in [
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2SinceWeek",
            "Promo2SinceYear",
        ]:
            df_lojas[col].fillna(0, inplace=True)

        df_lojas["CompetitionDistance"].fillna(df_lojas["CompetitionDistance"].mean(), inplace=True)

        df_completo = pd.merge(df_vendas, df_lojas, on="Store", how="left")
        df_completo.drop(columns=["Open"], inplace=True)

        df_completo["Year"] = df_completo["Date"].dt.year
        df_completo["Month"] = df_completo["Date"].dt.month
        df_completo["Day"] = df_completo["Date"].dt.day
        df_completo["DayOfWeek"] = df_completo["Date"].dt.dayofweek + 1
        df_completo["WeekOfYear"] = df_completo["Date"].dt.isocalendar().week.astype(int)
        df_completo["SalesPerCustomer"] = np.where(
            df_completo["Customers"] > 0,
            df_completo["Sales"] / df_completo["Customers"],
            0,
        )

        df_completo = reduzir_uso_memoria(df_completo, "df_completo")
        verificar_diretorios()
        df_completo.to_parquet(CAMINHO_ARQUIVO_PROCESSADO, index=False)
        logging.info("DataFrame processado salvo com sucesso")
        return df_completo
    except Exception as e:
        logging.error(f"Erro ao processar dados brutos: {e}")
        return None


def get_principal_dataset(use_samples: bool = False, n_amostras: int = N_AMOSTRAS_PADRAO, random_state: int = 42):
    key = (use_samples, n_amostras)
    if key not in _principal_cache:
        df = processar_dados_brutos()
        if df is None:
            return pd.DataFrame()
        if use_samples:
            df = df.groupby("Store").sample(n=n_amostras, random_state=random_state)
        _principal_cache[key] = df
    return _principal_cache[key]


def filtrar_por_data(df: pd.DataFrame, data_inicio: str | None, data_fim: str | None):
    if data_inicio:
        df = df[df["Date"] >= pd.to_datetime(data_inicio)]
    if data_fim:
        df = df[df["Date"] <= pd.to_datetime(data_fim)]
    return df


# --------------------------------------------------------------------------------------
# Funções de Alto Nível para Carregamento e Estados do Dataset
# --------------------------------------------------------------------------------------

def carregar_dados(
    modo: str = 'completo',
    n_amostras: int = N_AMOSTRAS_PADRAO,
    data_inicio: str | None = None,
    data_fim: str | None = None,
    force_reprocess: bool = False,
    random_state: int = 42,
):
    """Carrega e devolve um dicionário com todas as estruturas de dados
    necessárias pelos layouts/callbacks do dashboard.

    Parâmetros
    ----------
    modo : {'completo', 'amostra', 'data'}
        Estrategia de carregamento.
    n_amostras : int
        Nº de amostras por loja quando ``modo == 'amostra'``.
    data_inicio, data_fim : str | None
        Intervalo de datas (formato YYYY-MM-DD) para filtrar quando
        ``modo == 'data'``.
    force_reprocess : bool
        Se ``True`` ignora o cache em disco e reconstrói o parquet
        processado.
    random_state : int
        Semente para amostragem reprodutível.
    """

    # 1) Processa (ou lê) o dataset limpo
    df_completo = processar_dados_brutos(force_reprocess=force_reprocess)
    if df_completo is None or df_completo.empty:
        return {
            'df_principal': pd.DataFrame(),
            'media_vendas_antes': 0,
            'media_vendas_depois': 0,
            'contagem_vendas_antes': 0,
            'contagem_vendas_depois': 0,
            'df_lojas_tratado': pd.DataFrame(),
        }

    # 2) Aplica estratégias de sub-conjunto
    if modo == 'amostra':
        df_principal = df_completo.groupby('Store').sample(n=n_amostras, random_state=random_state)
    elif modo == 'data':
        df_principal = filtrar_por_data(df_completo, data_inicio, data_fim)
    else:
        df_principal = df_completo.copy()

    # 3) Métricas para a página de Limpeza
    df_vendas_raw, _ = carregar_dados_brutos()
    if df_vendas_raw is None:
        media_vendas_antes = media_vendas_depois = contagem_vendas_antes = contagem_vendas_depois = 0
    else:
        df_vendas_raw_abertas = df_vendas_raw[df_vendas_raw['Open'] == 1].copy()
        media_vendas_antes = df_vendas_raw_abertas['Sales'].mean()
        media_vendas_depois = df_principal['Sales'].mean()
        contagem_vendas_antes = len(df_vendas_raw_abertas)
        contagem_vendas_depois = len(df_principal)

    # 4) DataFrame de lojas tratado (apenas colunas da store)
    col_lojas = ['Store', 'CompetitionDistance']
    df_lojas_tratado = df_principal[col_lojas].drop_duplicates().reset_index(drop=True) if set(col_lojas).issubset(df_principal.columns) else pd.DataFrame()

    return {
        'df_principal': df_principal,
        'media_vendas_antes': media_vendas_antes,
        'media_vendas_depois': media_vendas_depois,
        'contagem_vendas_antes': contagem_vendas_antes,
        'contagem_vendas_depois': contagem_vendas_depois,
        'df_lojas_tratado': df_lojas_tratado,
    }


@functools.lru_cache(maxsize=2)
def get_data_states(use_samples: bool = False, n_amostras: int = N_AMOSTRAS_PADRAO):
    """Devolve um dicionário com dois DataFrames para análise de limpeza.

    * ``antes`` – dados *antes* da remoção de ``Open == 0``.
    * ``depois`` – dados *depois* da limpeza (equivalente ao ``df_principal`` do
      modo *completo*).
    """
    # Dados brutos (mantém Open original)
    df_vendas_raw, df_lojas_raw = carregar_dados_brutos()
    if df_vendas_raw is None:
        return {'antes': pd.DataFrame(), 'depois': pd.DataFrame()}

    df_raw = pd.merge(df_vendas_raw, df_lojas_raw, on='Store', how='left')

    # Após limpeza – utiliza o pipeline padronizado
    df_limpo = get_principal_dataset(use_samples=use_samples, n_amostras=n_amostras)

    return {'antes': df_raw, 'depois': df_limpo} 