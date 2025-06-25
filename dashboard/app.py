# dashboard/app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
from flask_caching import Cache
import warnings
import pandas as pd
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir avisos específicos do pandas para melhorar a legibilidade dos logs
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning, message=".*observed=False.*")
warnings.filterwarnings('ignore', category=FutureWarning, message=".*When grouping with a length-1 list-like.*")

# Importar layouts e dados com caminho absoluto
from dashboard.layouts import (
    barra_lateral,
    criar_layout_contextualizacao,
    criar_layout_limpeza_dados,
    criar_layout_analise_preliminar,
    criar_layout_dashboard_analise,
    criar_layout_previsao_vendas,
    criar_layout_analise_lojas,
    criar_layout_analise_3d
)
from dashboard.data_loader import carregar_dados, N_AMOSTRAS_PADRAO
from dashboard.callbacks import registrar_callbacks

# ==============================================================================
# Inicialização do Aplicativo
# ==============================================================================
# Configuração da porta
port = int(os.environ.get('PORT', 8050))

aplicativo = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        # Adicione aqui o caminho para seu CSS customizado se tiver um
        '/assets/css/estilos_customizados.css',
        '/assets/css/estilos_barra_lateral.css'
    ],
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    title='Dashboard Rossmann',
    update_title='Atualizando...'
)

# Configuração do servidor
server = aplicativo.server
aplicativo.title = "Rossmann Sales Dashboard"

# Configuração do cache
cache = Cache(aplicativo.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 600
})

# Cache para dados processados
@cache.memoize()
def get_cached_data(modo, n_amostras, data_inicio, data_fim, force_reprocess):
    return carregar_dados(
        modo=modo,
        n_amostras=n_amostras,
        data_inicio=data_inicio,
        data_fim=data_fim,
        force_reprocess=force_reprocess
    )

# Cache para layouts
layout_cache = {}

def get_layout(pathname, dados):
    if pathname not in layout_cache:
        if pathname == '/':
            layout_cache[pathname] = criar_layout_contextualizacao(dados)
        elif pathname == '/limpeza-dados':
            layout_cache[pathname] = criar_layout_limpeza_dados(dados)
        elif pathname == '/analise-preliminar':
            layout_cache[pathname] = criar_layout_analise_preliminar(dados)
        elif pathname == '/dashboard':
            layout_cache[pathname] = criar_layout_dashboard_analise(dados)
        elif pathname == '/analise-lojas':
            layout_cache[pathname] = criar_layout_analise_lojas(dados)
        elif pathname == '/analise-3d':
            layout_cache[pathname] = criar_layout_analise_3d(dados)
        elif pathname == '/previsao-vendas':
            layout_cache[pathname] = criar_layout_previsao_vendas()
    return layout_cache[pathname]

# ==============================================================================
# Configurações de carregamento dos dados
# ==============================================================================
modo_carregamento = os.environ.get('MODO_CARREGAMENTO', 'amostra')
n_amostras = int(os.environ.get('N_AMOSTRAS', '50'))
data_inicio = os.environ.get('DATA_INICIO', None)
data_fim = os.environ.get('DATA_FIM', None)
force_reprocess = os.environ.get('FORCE_REPROCESS', 'False').lower() == 'true'

logger.info(f"Carregando dados no modo '{modo_carregamento}'...")
logger.info(f"  - Amostras por loja: {n_amostras if modo_carregamento == 'amostra' else 'N/A'}")
if modo_carregamento == 'data':
    logger.info(f"  - Intervalo de datas: {data_inicio} a {data_fim if data_fim else 'hoje'}")
logger.info(f"  - Forçar reprocessamento: {force_reprocess}")

# Carregar dados usando cache
dados = get_cached_data(
    modo_carregamento,
    n_amostras,
    data_inicio,
    data_fim,
    force_reprocess
)

# Definir estado inicial para o store (modo e nº amostras)
initial_store = {'modo': 'amostras', 'n_amostras': 50}

# ==============================================================================
# Layout do Aplicativo
# ==============================================================================
# Carregar todos os layouts no início
# A visibilidade será controlada por um callback que altera o 'display'
aplicativo.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Componente de armazenamento para o DataFrame principal
    dcc.Store(id='armazenamento-df-principal', data=initial_store),
    barra_lateral,
    html.Div(
        id='conteudo-pagina',
        className='content',
        children=[
            html.Div(get_layout('/', dados), id='conteudo-pagina-/', style={'display': 'block'}),
            html.Div(get_layout('/limpeza-dados', dados), id='conteudo-pagina-/limpeza-dados', style={'display': 'none'}),
            html.Div(get_layout('/analise-preliminar', dados), id='conteudo-pagina-/analise-preliminar', style={'display': 'none'}),
            html.Div(get_layout('/dashboard', dados), id='conteudo-pagina-/dashboard', style={'display': 'none'}),
            html.Div(get_layout('/analise-lojas', dados), id='conteudo-pagina-/analise-lojas', style={'display': 'none'}),
            html.Div(get_layout('/analise-3d', dados), id='conteudo-pagina-/analise-3d', style={'display': 'none'}),
            html.Div(get_layout('/previsao-vendas', dados), id='conteudo-pagina-/previsao-vendas', style={'display': 'none'}),
        ]
    )
])

# ==============================================================================
# Registro de Callbacks
# ==============================================================================
registrar_callbacks(aplicativo, dados)

# ==============================================================================
# Execução do Aplicativo
# ==============================================================================
if __name__ == '__main__':
    aplicativo.run(host='localhost', port=port, debug=False)

