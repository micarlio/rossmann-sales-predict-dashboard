import pandas as pd
import numpy as np
from pathlib import Path
import os
import logging
import time

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard_data_loader.log')
    ]
)

# Define constantes globais
# Diretório base do projeto
DIRETORIO_BASE = Path(__file__).resolve().parent.parent
DIRETORIO_DADOS = DIRETORIO_BASE / "dataset"

# Caminhos para dados brutos
CAMINHO_ARQUIVO_TREINO_BRUTO = DIRETORIO_DADOS / "brutos" / "train.parquet"
CAMINHO_ARQUIVO_LOJAS_BRUTO = DIRETORIO_DADOS / "brutos" / "store.parquet"

# Caminho para dados processados em Parquet
CAMINHO_ARQUIVO_PROCESSADO = DIRETORIO_DADOS / "processados" / "df_completo_processado.parquet"

# Número padrão de amostras por loja
N_AMOSTRAS_PADRAO = 50

_principal_cache = {}

def verificar_diretorios():
    """Verifica se todos os diretórios necessários existem e os cria se não existirem."""
    diretorios = [
        DIRETORIO_DADOS / "brutos",
        DIRETORIO_DADOS / "processados",
        DIRETORIO_DADOS / "reduzidos"
    ]
    
    for diretorio in diretorios:
        if not diretorio.exists():
            diretorio.mkdir(parents=True, exist_ok=True)
            logging.info(f"Diretório criado: {diretorio}")


def reduzir_uso_memoria(df, nome_df="DataFrame"):
    """
    Reduz o uso de memória de um DataFrame convertendo tipos de dados.
    
    Args:
        df (pd.DataFrame): DataFrame a ser otimizado
        nome_df (str): Nome do DataFrame para logging
        
    Returns:
        pd.DataFrame: DataFrame otimizado
    """
    inicio = time.time()
    uso_memoria_antes = df.memory_usage(deep=True).sum() / 1024**2
    
    # Conversão de tipos para reduzir memória
    for col in df.columns:
        tipo_col = str(df[col].dtype)
        
        # Colunas de texto/categorias
        if tipo_col == 'object':
            num_valores_unicos = len(df[col].unique())
            num_total = len(df[col])
            
            # Se contiver muitos valores únicos, manter como object
            # Se tiver poucos valores únicos, converter para category
            if num_valores_unicos / num_total < 0.5:
                df[col] = df[col].astype('category')
        
        # Colunas numéricas inteiras
        elif 'int' in tipo_col:
            valores_min = df[col].min()
            valores_max = df[col].max()
            
            # Conversão para o menor tipo inteiro possível
            if valores_min >= 0:
                if valores_max < 2**8:
                    df[col] = df[col].astype(np.uint8)
                elif valores_max < 2**16:
                    df[col] = df[col].astype(np.uint16)
                elif valores_max < 2**32:
                    df[col] = df[col].astype(np.uint32)
            else:
                if valores_min > -2**7 and valores_max < 2**7:
                    df[col] = df[col].astype(np.int8)
                elif valores_min > -2**15 and valores_max < 2**15:
                    df[col] = df[col].astype(np.int16)
                elif valores_min > -2**31 and valores_max < 2**31:
                    df[col] = df[col].astype(np.int32)
        
        # Colunas numéricas float
        elif 'float' in tipo_col:
            df[col] = df[col].astype(np.float32)
    
    # Calcula redução de memória
    uso_memoria_depois = df.memory_usage(deep=True).sum() / 1024**2
    fim = time.time()
    
    logging.info(f"Otimização de memória para {nome_df}:")
    logging.info(f"  - Antes: {uso_memoria_antes:.2f} MB")
    logging.info(f"  - Depois: {uso_memoria_depois:.2f} MB")
    logging.info(f"  - Redução: {(1 - uso_memoria_depois/uso_memoria_antes)*100:.2f}%")
    logging.info(f"  - Tempo: {fim - inicio:.2f} segundos")
    
    return df


def carregar_dados_brutos():
    """
    Carrega os dados brutos dos arquivos CSV.
    
    Returns:
        tuple: (df_vendas, df_lojas) ou (None, None) em caso de erro
    """
    try:
        logging.info("Carregando dados brutos em Parquet...")
        
        # Verifica se os arquivos Parquet existem
        if not CAMINHO_ARQUIVO_TREINO_BRUTO.exists():
            logging.error(f"Arquivo de vendas não encontrado: {CAMINHO_ARQUIVO_TREINO_BRUTO}")
            return None, None
            
        if not CAMINHO_ARQUIVO_LOJAS_BRUTO.exists():
            logging.error(f"Arquivo de lojas não encontrado: {CAMINHO_ARQUIVO_LOJAS_BRUTO}")
            return None, None
        
        # Carrega os dados em Parquet
        df_vendas = pd.read_parquet(CAMINHO_ARQUIVO_TREINO_BRUTO)
        df_vendas['Date'] = pd.to_datetime(df_vendas['Date'])
        
        df_lojas = pd.read_parquet(CAMINHO_ARQUIVO_LOJAS_BRUTO)
        
        # Otimiza memória
        df_vendas = reduzir_uso_memoria(df_vendas, "df_vendas")
        df_lojas = reduzir_uso_memoria(df_lojas, "df_lojas")
        
        logging.info(f"Dados brutos carregados com sucesso. Vendas: {len(df_vendas)} registros, Lojas: {len(df_lojas)} registros")
        
        return df_vendas, df_lojas
        
    except Exception as e:
        logging.error(f"Erro ao carregar dados brutos: {str(e)}")
        return None, None


def processar_dados_brutos(force_reprocess=False):
    """
    Processa os dados brutos, aplicando limpeza e transformações.
    
    Args:
        force_reprocess (bool): Se True, força o reprocessamento mesmo se já existir arquivo processado
    
    Returns:
        pd.DataFrame: DataFrame processado ou None em caso de erro
    """
    try:
        # Verifica se já existe um arquivo processado e não está forçando reprocessamento
        if CAMINHO_ARQUIVO_PROCESSADO.exists() and not force_reprocess:
            logging.info(f"Carregando arquivo processado: {CAMINHO_ARQUIVO_PROCESSADO}")
            df_completo = pd.read_parquet(CAMINHO_ARQUIVO_PROCESSADO)
            logging.info(f"Arquivo processado carregado com sucesso: {len(df_completo)} registros")
            return df_completo
            
        # Carrega dados brutos
        df_vendas, df_lojas = carregar_dados_brutos()
        if df_vendas is None or df_lojas is None:
            return None
        
        # Processa os dados
        logging.info("Processando dados...")
        
        # Filtra apenas lojas abertas
        df_vendas_filtrado = df_vendas[df_vendas['Open'] == 1].copy()
        logging.info(f"Filtradas apenas lojas abertas: {len(df_vendas_filtrado)} de {len(df_vendas)} registros")
        
        # Tratamento de valores ausentes em df_lojas
        # Primeiro, converter coluna categórica para string para evitar erro de categoria
        if 'PromoInterval' in df_lojas.columns:
            # Se a coluna for categórica, primeiro a convertemos para string
            if str(df_lojas['PromoInterval'].dtype).startswith('category'):
                df_lojas['PromoInterval'] = df_lojas['PromoInterval'].astype(str)
            
            df_lojas['PromoInterval'] = df_lojas['PromoInterval'].fillna("Nenhum") 

        colunas_preencher_zero = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']
        for col in colunas_preencher_zero:
            if col in df_lojas.columns:
                df_lojas[col] = df_lojas[col].fillna(0)

        # CompetitionDistance: Preencher com a MÉDIA
        if 'CompetitionDistance' in df_lojas.columns:
            df_lojas['CompetitionDistance'] = df_lojas['CompetitionDistance'].fillna(df_lojas['CompetitionDistance'].mean())
        
        # Merge dos dataframes
        logging.info("Realizando merge dos dataframes...")
        df_completo = pd.merge(df_vendas_filtrado, df_lojas, on='Store', how='left')
        
        # Remove coluna Open que não é mais necessária
        if 'Open' in df_completo.columns:
            df_completo.drop(['Open'], axis=1, inplace=True)
        
        # Feature engineering básico
        df_completo['Year'] = df_completo['Date'].dt.year
        df_completo['Month'] = df_completo['Date'].dt.month
        df_completo['Day'] = df_completo['Date'].dt.day
        df_completo['DayOfWeek'] = df_completo['Date'].dt.dayofweek + 1  # +1 para ser 1 (Seg) a 7 (Dom)
        df_completo['WeekOfYear'] = df_completo['Date'].dt.isocalendar().week.astype(int)
        df_completo['SalesPerCustomer'] = np.where(df_completo['Customers'] > 0, df_completo['Sales'] / df_completo['Customers'], 0)
        
        # Otimiza memória novamente
        df_completo = reduzir_uso_memoria(df_completo, "df_completo")
        
        # Salva o DataFrame processado em Parquet
        verificar_diretorios()
        df_completo.to_parquet(CAMINHO_ARQUIVO_PROCESSADO, index=False)
        
        logging.info(f"DataFrame processado salvo com sucesso: {CAMINHO_ARQUIVO_PROCESSADO}")
        logging.info(f"Total de registros: {len(df_completo)}")
        
        return df_completo
        
    except Exception as e:
        logging.error(f"Erro ao processar dados brutos: {str(e)}")
        return None


def amostrar_por_loja(df, n_amostras=N_AMOSTRAS_PADRAO, random_state=42):
    """
    Amostra um número fixo de registros por loja.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        n_amostras (int): Número de amostras por loja
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        pd.DataFrame: DataFrame com amostras selecionadas
    """
    if df is None:
        logging.error("DataFrame é None, não é possível amostrar")
        return pd.DataFrame()
        
    if 'Store' not in df.columns:
        logging.error("Coluna 'Store' não encontrada no DataFrame")
        return df
    
    logging.info(f"Amostrando {n_amostras} registros por loja...")
    inicio = time.time()
    
    # Contagem de registros por loja
    registros_por_loja = df['Store'].value_counts()
    logging.info(f"Total de lojas: {len(registros_por_loja)}")
    
    # Lojas com poucos registros
    lojas_com_poucos_registros = registros_por_loja[registros_por_loja < n_amostras]
    if not lojas_com_poucos_registros.empty:
        logging.warning(f"{len(lojas_com_poucos_registros)} lojas têm menos de {n_amostras} registros disponíveis")
        logging.warning(f"Média de registros nestas lojas: {lojas_com_poucos_registros.mean():.2f}")
    
    resultado = []
    for loja, grupo in df.groupby('Store'):
        if len(grupo) >= n_amostras:
            # Se tiver registros suficientes, amostra a quantidade desejada
            amostra = grupo.sample(n=n_amostras, random_state=random_state)
        else:
            # Se não tiver registros suficientes, usa todos os disponíveis
            amostra = grupo
            logging.warning(f"Loja {loja} possui apenas {len(grupo)} registros, menos que os {n_amostras} solicitados.")
        
        resultado.append(amostra)
    
    df_resultado = pd.concat(resultado, ignore_index=True)
    
    # Log do tempo e resultado
    fim = time.time()
    logging.info(f"Amostragem concluída em {fim - inicio:.2f} segundos")
    logging.info(f"Total de registros após amostragem: {len(df_resultado)}")
    
    return df_resultado


def filtrar_por_data(df, data_inicio=None, data_fim=None):
    """
    Filtra os dados por um intervalo de datas.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        data_inicio (str ou datetime): Data de início no formato 'YYYY-MM-DD'
        data_fim (str ou datetime): Data de fim no formato 'YYYY-MM-DD'
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    if df is None:
        logging.error("DataFrame é None, não é possível filtrar por data")
        return pd.DataFrame()
        
    if 'Date' not in df.columns:
        logging.error("Coluna 'Date' não encontrada no DataFrame")
        return df
    
    # Convertendo string para datetime se necessário
    if isinstance(data_inicio, str):
        data_inicio = pd.to_datetime(data_inicio)
    
    if isinstance(data_fim, str):
        data_fim = pd.to_datetime(data_fim)
    
    # Aplicando filtros
    if data_inicio and data_fim:
        logging.info(f"Filtrando dados entre {data_inicio.strftime('%Y-%m-%d')} e {data_fim.strftime('%Y-%m-%d')}...")
        df_filtrado = df[(df['Date'] >= data_inicio) & (df['Date'] <= data_fim)]
    elif data_inicio:
        logging.info(f"Filtrando dados a partir de {data_inicio.strftime('%Y-%m-%d')}...")
        df_filtrado = df[df['Date'] >= data_inicio]
    elif data_fim:
        logging.info(f"Filtrando dados até {data_fim.strftime('%Y-%m-%d')}...")
        df_filtrado = df[df['Date'] <= data_fim]
    else:
        logging.warning("Nenhum filtro de data especificado, retornando DataFrame original")
        return df
    
    logging.info(f"Total de registros após filtro de data: {len(df_filtrado)}")
    
    return df_filtrado


def carregar_dados(
    modo='amostra',
    n_amostras=N_AMOSTRAS_PADRAO,
    data_inicio=None,
    data_fim=None,
    force_reprocess=False,
    random_state=42
):
    """
    Função principal para carregar e processar os dados.
    
    Args:
        modo (str): Modo de carregamento: 'amostra' ou 'data'
        n_amostras (int): Número de amostras por loja (para modo='amostra')
        data_inicio (str ou datetime): Data de início (para modo='data')
        data_fim (str ou datetime): Data de fim (para modo='data')
        force_reprocess (bool): Se True, força o reprocessamento dos dados brutos
        random_state (int): Seed para reprodutibilidade na amostragem
        
    Returns:
        dict: Dicionário com todos os DataFrames e métricas necessárias para o dashboard
    """
    # Verifica se os diretórios existem
    verificar_diretorios()
    
    # Carrega/processa o DataFrame completo
    df_completo = processar_dados_brutos(force_reprocess)
    
    # Se não conseguiu processar os dados, retorna um dicionário com dados vazios
    # mas que ainda tem todas as chaves necessárias para não quebrar o dashboard
    if df_completo is None:
        logging.error("Não foi possível carregar ou processar os dados. Retornando dados vazios.")
        return {
        "df_principal": pd.DataFrame(),
        "df_vendas_original": pd.DataFrame(),
        "df_lojas_original": pd.DataFrame(),
        "distancia_max_global": 0,
        "contagem_vendas_antes": 0,
        "media_vendas_antes": 0,
        "contagem_vendas_depois": 0,
        "media_vendas_depois": 0,
        "df_vendas_antes_preprocessamento": pd.DataFrame(),
        "df_vendas_depois_preprocessamento": pd.DataFrame(),
            "df_lojas_tratado": pd.DataFrame(),
            "df_principal_json": "{}"
        }
    
    # Métricas sobre o dataset completo
    contagem_registros_original = len(df_completo)
    media_vendas_original = df_completo['Sales'].mean() if 'Sales' in df_completo.columns else 0
    
    # Aplica o modo de seleção
    if modo.lower() == 'amostra':
        df_reduzido = amostrar_por_loja(df_completo, n_amostras, random_state)
    elif modo.lower() == 'data':
        df_reduzido = filtrar_por_data(df_completo, data_inicio, data_fim)
    else:
        logging.error(f"Modo '{modo}' não reconhecido. Usando modo 'amostra' como padrão.")
        df_reduzido = amostrar_por_loja(df_completo, n_amostras, random_state)
    
    # Métricas sobre o dataset reduzido
    contagem_registros_reduzido = len(df_reduzido)
    media_vendas_reduzido = df_reduzido['Sales'].mean() if 'Sales' in df_reduzido.columns else 0
    
    # DataFrame para tabela original de vendas e lojas
    df_vendas_original = df_reduzido[['Store', 'Date', 'Sales', 'Customers', 'DayOfWeek', 'StateHoliday', 'SchoolHoliday']].copy() if not df_reduzido.empty else pd.DataFrame()
    
    df_lojas_original = None
    if not df_reduzido.empty:
        colunas_lojas = [col for col in df_reduzido.columns if col not in ['Date', 'Sales', 'Customers', 'DayOfWeek', 'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Day', 'WeekOfYear', 'SalesPerCustomer']]
        if 'Store' not in colunas_lojas:
            colunas_lojas.append('Store')
        
        # Obter apenas os valores únicos por loja
        df_lojas_original = df_reduzido[colunas_lojas].drop_duplicates(subset=['Store']).copy()
    else:
        df_lojas_original = pd.DataFrame()
    
    # Tratamento dos dados como no código original
    df_lojas_tratado = df_lojas_original.copy() if df_lojas_original is not None else pd.DataFrame()
    
    # Prepara dados para os histogramas comparativos
    df_vendas_antes_preprocessamento = df_vendas_original.copy() if not df_vendas_original.empty else pd.DataFrame()
    df_vendas_depois_preprocessamento = df_vendas_original.copy() if not df_vendas_original.empty else pd.DataFrame()
    
    # Prepara o resultado como um dicionário similar ao original
    dados = {
        "df_principal": df_reduzido,
        "df_vendas_original": df_vendas_original,
        "df_lojas_original": df_lojas_original,
        "distancia_max_global": df_reduzido['CompetitionDistance'].max() if not df_reduzido.empty and 'CompetitionDistance' in df_reduzido.columns else 0,
        "contagem_vendas_antes": contagem_registros_original,
        "media_vendas_antes": media_vendas_original,
        "contagem_vendas_depois": contagem_registros_reduzido,
        "media_vendas_depois": media_vendas_reduzido,
        "df_vendas_antes_preprocessamento": df_vendas_antes_preprocessamento,
        "df_vendas_depois_preprocessamento": df_vendas_depois_preprocessamento,
        "df_lojas_tratado": df_lojas_tratado
    }
    
    # Converte o DataFrame principal para JSON para armazenar no dcc.Store
    if not df_reduzido.empty:
        dados["df_principal_json"] = df_reduzido.to_json(date_format='iso', orient='split')
    else:
        dados["df_principal_json"] = "{}"
    
    logging.info(f"Carregamento de dados concluído no modo '{modo}'")
    logging.info(f"Registros no DataFrame reduzido: {contagem_registros_reduzido}")

    return dados

# Nova função para obter estados dos dados (bruto, limpo e amostrado)
def get_data_states(use_samples=False, n_amostras=N_AMOSTRAS_PADRAO, random_state=42):
    """
    Carrega os dados e retorna um dicionário com DataFrames nos estados:
    'antes' (bruto), 'depois' (após limpeza) e 'amostrado' (após amostragem).
    """
    # Carrega dados brutos
    df_vendas_raw, df_lojas_raw = carregar_dados_brutos()
    if df_vendas_raw is None or df_lojas_raw is None:
        return {"antes": pd.DataFrame(), "depois": pd.DataFrame(), "amostrado": pd.DataFrame()}

    # Estado 1: Antes da Limpeza
    df_antes = pd.merge(df_vendas_raw, df_lojas_raw, how='left', on='Store')
    
    # Estado 2: Depois da Limpeza (usando processar_dados_brutos)
    df_depois = processar_dados_brutos(force_reprocess=False)
    
    # Estado 3: Depois da Amostragem (usando get_principal_dataset)
    df_amostrado = get_principal_dataset(use_samples, n_amostras, random_state)

    return {"antes": df_antes, "depois": df_depois, "amostrado": df_amostrado}

def get_principal_dataset(use_samples=False, n_amostras=N_AMOSTRAS_PADRAO, random_state=42):
    """
    Retorna o DataFrame principal (limpo ou amostrado) em memória, cacheado para evitar recálculos.
    """
    key = (use_samples, n_amostras)
    if key in _principal_cache:
        return _principal_cache[key]
    
    # Obter o dataset limpo (carregado do arquivo processado)
    df_base = processar_dados_brutos(force_reprocess=False)
    
    # Aplicar amostragem se solicitado
    if use_samples and df_base is not None:
        df_princ = amostrar_por_loja(df_base, n_amostras=n_amostras, random_state=random_state)
    else:
        df_princ = df_base
        
    _principal_cache[key] = df_princ
    return df_princ