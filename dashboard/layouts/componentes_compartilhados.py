import dash_bootstrap_components as dbc
from dash import dcc, html

# Importar as constantes com os novos nomes
from ..utils import criar_figura_vazia, criar_icone_informacao # Refatorar nomes de funções
from ..config import AZUL_ESCURO, ALTURA_GRAFICO

def gerar_titulo_secao(titulo, subtitulo): # Refatorar nome da função
    """Gera um título de seção padronizado com subtítulo."""
    return dbc.Row(
        dbc.Col(
            html.Div([
                html.H2(titulo, className="section-title"),
                html.P(subtitulo, className="section-subtitle")
            ]),
            width=12
        ),
        className="mb-4 mt-4" # Adiciona margem para separação visual
    )

# ==============================================================================
# Armazenamento de estado da barra lateral
# ==============================================================================
# dcc.Store para armazenar o estado (colapsado ou não) da barra lateral
# storage_type='session' para persistir na sessão do usuário
armazenamento_estado_barra_lateral = dcc.Store(id='armazenamento-estado-barra-lateral', storage_type='session', data='expanded') # Refatorar nome da variável e ID


# ==============================================================================
# FUNÇÃO AUXILIAR PARA CARDS DE GRÁFICO
# ==============================================================================

def criar_card_grafico(id_grafico, id_analise, largura_md=6, classe_card="", controles_extras=None): # Refatorar nome da função e parâmetros
    """
    Cria um card padrão para exibir um gráfico com uma área para análise textual.

    Args:
        id_grafico: ID do gráfico
        id_analise: ID do elemento de análise
        largura_md: Largura do card em colunas MD
        classe_card: Classes CSS adicionais para o card
        controles_extras: Elementos extras a serem adicionados antes do gráfico (ex: filtros)
    """
    filhos_corpo_card = [] # Refatorar nome da variável

    # Adiciona controles extras se fornecidos
    if controles_extras:
        filhos_corpo_card.append(controles_extras) # Usar nova variável refatorada

    # Adiciona o gráfico
    filhos_corpo_card.append( # Usar nova variável refatorada
        dcc.Loading(
            children=[
                dcc.Graph(
                    id=id_grafico,
                    config={'displayModeBar': False},
                    style={'height': f'{ALTURA_GRAFICO}px'} # Usar nova constante
                )
            ]
        )
    )

    # Adiciona a área de análise
    filhos_corpo_card.append( # Usar nova variável refatorada
        html.Div([
            html.P([
                html.I(className="fas fa-lightbulb me-2"),
                html.Strong("Análise: "),
                html.Span(id=id_analise)
            ])
        ], className="analise-text-box mt-3") # Refatorar classe CSS
    )

    return dbc.Col(
        dbc.Card(
            dbc.CardBody(filhos_corpo_card), # Usar nova variável refatorada
            className=f"custom-card {classe_card}"
        ),
        md=largura_md,
        className="graph-card-col" # Refatorar classe CSS
    )

def criar_card_grafico_3d(id_grafico, titulo, id_analise=None, children=None): # Refatorar nome da função e parâmetros
    """Cria um card customizado para os gráficos 3D."""
    filhos_corpo_card = [ # Refatorar nome da variável
        dcc.Loading(children=[dcc.Graph(
            id=id_grafico,
            # Gráfico começa visível com uma figura vazia
            style={'height': '65vh'},
            config={'displayModeBar': False}
        )])
    ]

    # Adiciona a caixa de análise se um ID for fornecido
    if id_analise:
        caixa_analise = html.Div([ # Refatorar nome da variável
            html.P([
                html.I(className="fas fa-lightbulb me-2"),
                html.Strong("Análise: "),
                html.Span(id=id_analise)
            ])
        ], className="mt-3 p-3 analise-text-box") # Refatorar classe CSS
        filhos_corpo_card.append(caixa_analise) # Usar nova variável refatorada

    if children:
        # Adiciona elementos extras (como dropdown) no início do corpo do card
        filhos_corpo_card.insert(0, children) # Usar nova variável refatorada

    return dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(html.H5(titulo, className="m-0 p-1 text-center fw-bold", style={'color': AZUL_ESCURO})), # Usar nova constante
                dbc.CardBody(filhos_corpo_card, style={'padding': '0.5rem'}) # Usar nova variável refatorada
            ],
            className="custom-card h-100"
        ),
        md=12,
        className="mb-4"
    )

# ==============================================================================
# COMPONENTES REUTILIZÁVEIS
# ==============================================================================

def criar_botoes_cabecalho(nome_pagina): # Refatorar nome da função e parâmetro
    """Cria os botões de ação do cabeçalho da página (REUTILIZÁVEL)."""
    return html.Div([
        # Os IDs agora são dicionários para funcionar com callbacks MATCH
        html.Button(html.I(className="fas fa-expand-alt"), id={'type': 'botao-tela-cheia', 'index': nome_pagina}, className='btn btn-outline-secondary me-2', title='Tela Cheia'), # Refatorar ID
        dcc.Store(id={'type': 'saida-tela-cheia', 'index': nome_pagina}) # Refatorar ID
    ], className="d-flex align-items-center")

# --- Barra de Navegação Lateral Elegante ---
barra_lateral = html.Div( # Refatorar nome da variável
    [
        armazenamento_estado_barra_lateral, # Adiciona o Store ao layout da sidebar # Usar nova variável refatorada
        # Cabeçalho da Sidebar
        html.Div(
            [
                html.Img(
                    src="/assets/images/rossmann_logo2.png", # Usa o logo de expansão como padrão
                    id="img-logo-barra-lateral", # Adiciona um ID para o callback # Refatorar ID
                    className="sidebar-logo", # Refatorar classe CSS
                    style={'marginBottom': '1rem'}
                ),
            ],
            className="sidebar-header" # Refatorar classe CSS
        ),

        # Links de Navegação
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-info-circle fa-fw me-2"), html.Span("Contextualização", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-broom fa-fw me-2"), html.Span("Limpeza dos Dados", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/limpeza-dados",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-chart-line fa-fw me-2"), html.Span("Análise Preliminar", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/analise-preliminar",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-tachometer-alt fa-fw me-2"), html.Span("Visão Geral", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/dashboard",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-store fa-fw me-2"), html.Span("Análise de Lojas", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/analise-lojas",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-cube fa-fw me-2"), html.Span("Análise 3D", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/analise-3d",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-bullseye fa-fw me-2"), html.Span("Previsão de Vendas", className="sidebar-text-content")], # Refatorar classe CSS
                    href="/previsao-vendas",
                    active="exact",
                    className="nav-link-custom" # Refatorar classe CSS
                ),
            ],
            vertical=True,
            pills=True,
            className="flex-grow-1"
        ),

        # Rodapé e botão de toggle
        html.Div(
            [
                html.Div([
                    html.Hr(className="sidebar-hr"), # Refatorar classe CSS
                    # O texto do rodapé agora está em um Div para ser escondido
                    html.Div([
                        html.P("Desenvolvido por Micarlos", className="text-white-50 small text-center"),
                        html.P("UFRN - 2025", className="text-white-50 small text-center mt-n2")
                    ], className="sidebar-text-content"), # Refatorar classe CSS
                ]),

                # Botão para colapsar/expandir a barra lateral
                html.Div(
                    dbc.Button(
                        html.I(id="icone-alternar-barra-lateral", className="fas fa-angle-double-left"), # Refatorar ID
                        id="alternar-barra-lateral", # Refatorar ID
                        className="sidebar-toggle-button", # Refatorar classe CSS
                        n_clicks=0
                    ),
                    className="sidebar-toggle-wrapper" # Refatorar classe CSS
                )
            ],
            className="sidebar-footer"
        )
    ],
    id="barra-lateral", # Refatorar ID
    # A classe inicial é determinada pelo callback, mas podemos começar com 'sidebar'
    className="sidebar d-flex flex-column", # Refatorar classe CSS
)


# ==============================================================================
# PAINEL DE FILTROS REUTILIZÁVEL
# ==============================================================================
def criar_card_filtros(df_principal, prefix=''): # Refatorar nome da função e adiciona prefixo para IDs
    """Cria o painel de filtros que pode ser usado em múltiplas páginas, com namespace opcional."""
    return dbc.Card([
        dbc.CardHeader(html.H4([html.I(className="fas fa-filter me-2"), "Painel de Filtros"], className="m-0 p-2 text-center fw-bold")),
        dbc.CardBody([
            dbc.Row([
                # Coluna 1: Filtros principais de escopo
                dbc.Col([
                    html.Div([
                        dbc.Label("Período de Análise", html_for=prefix+"filtro-data", className="fw-bold"),
                        dcc.DatePickerRange(
                            id=prefix+"filtro-data",
                            min_date_allowed=df_principal['Date'].min().date(),
                            max_date_allowed=df_principal['Date'].max().date(),
                            start_date=df_principal['Date'].min().date(),
                            end_date=df_principal['Date'].max().date(),
                            display_format='DD/MM/YYYY',
                            className="date-picker-custom w-100"
                        ),
                    ], className="mb-3"),
                    html.Div([
                        dbc.Label("Tipo(s) de Loja", html_for=prefix+"filtro-tipo-loja", className="fw-bold"),
                        dcc.Dropdown(
                            id=prefix+"filtro-tipo-loja",
                            options=[{'label': t, 'value': t} for t in sorted(df_principal['StoreType'].unique())],
                            value=sorted(df_principal['StoreType'].unique()),
                            multi=True,
                            placeholder="Selecione tipos de loja",
                            className="dropdown-dash"
                        ),
                    ], className="mb-3"),
                    html.Div([
                        dbc.Label("Loja(s) Específica(s)", html_for=prefix+"filtro-loja-especifica", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id=prefix+"filtro-loja-especifica",
                            options=[{'label': str(s), 'value': s} for s in sorted(df_principal['Store'].unique())],
                            value=[],
                            multi=True,
                            placeholder="Busque por uma ou mais lojas...",
                            className="dropdown-dash"
                        ),
                    ], className="mb-3"),
                ], md=6, className="border-end-md py-3 px-4"),

                # Coluna 2: Filtros de análise e feriados
                dbc.Col([
                    html.Div([
                        dbc.Label("Métrica Principal", html_for=prefix+"filtro-metrica-temporal", className="fw-bold mb-2"),
                        dbc.RadioItems(
                            id=prefix+"filtro-metrica-temporal",
                            options=[
                                {'label': 'Vendas', 'value': 'Sales'},
                                {'label': 'Clientes', 'value': 'Customers'},
                                {'label': 'Ticket Médio', 'value': 'SalesPerCustomer'}
                            ],
                            value='Sales',
                            inline=True,
                            className="grupo-radio",
                            labelClassName="radio-item-custom",
                            inputClassName="radio-input-custom"
                        ),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Feriado Estadual", html_for=prefix+"filtro-feriado-estadual", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id=prefix+"filtro-feriado-estadual",
                                options=[
                                    {'label': 'Todos', 'value': 'all'},
                                    {'label': 'Nenhum (0)', 'value': '0'},
                                    {'label': 'Público (a)', 'value': 'a'},
                                    {'label': 'Páscoa (b)', 'value': 'b'},
                                    {'label': 'Natal (c)', 'value': 'c'}
                                ],
                                value='all',
                                multi=False,
                                className="dropdown-dash",
                                clearable=False
                            ),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Feriado Escolar", html_for=prefix+"filtro-feriado-escolar", className="fw-bold mb-2"),
                            dcc.Dropdown(
                                id=prefix+"filtro-feriado-escolar",
                                options=[
                                    {'label': 'Todos', 'value': 'all'},
                                    {'label': 'Sim (1)', 'value': '1'},
                                    {'label': 'Não (0)', 'value': '0'}
                                ],
                                value='all',
                                multi=False,
                                className="dropdown-dash",
                                clearable=False
                            ),
                        ], md=6)
                    ])
                ], md=6, className="py-3 px-4"),
            ])
        ])
    ], className="custom-card mb-4")

def criar_card_filtros_3d(df_principal): # Refatorar nome da função e parâmetro
    """Cria um painel de filtros simplificado para a página 3D, sem controles de loja."""
    return dbc.Card([
        dbc.CardHeader(html.H4([html.I(className="fas fa-filter me-2"), "Filtros Gerais"], className="m-0 p-2 text-center fw-bold")),
        dbc.CardBody([
            dbc.Row([
                # Coluna 1: Filtros de data
                dbc.Col([
                    html.Div([
                        dbc.Label("Período de Análise", html_for="filtro-data", className="fw-bold"),
                        dcc.DatePickerRange(
                            id='filtro-data-3d', # ID Único para a página 3D
                            min_date_allowed=df_principal['Date'].min().date(), # Usar novo DataFrame
                            max_date_allowed=df_principal['Date'].max().date(), # Usar novo DataFrame
                            start_date=df_principal['Date'].min().date(), # Usar novo DataFrame
                            end_date=df_principal['Date'].max().date(), # Usar novo DataFrame
                            display_format='DD/MM/YYYY',
                            className="date-picker-custom w-100" # Refatorar classe CSS
                        ),
                    ]),
                ], md=6, className="border-end-md py-3 px-4"),

                # Coluna 2: Filtros de feriados
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Feriado Estadual", html_for="filtro-feriado-estadual-3d", className="fw-bold mb-2"), # Refatorar ID
                            dcc.Dropdown(
                                id='filtro-feriado-estadual-3d', # Refatorar ID
                                options=[
                                    {'label': 'Todos', 'value': 'all'},
                                    {'label': 'Nenhum (0)', 'value': '0'},
                                    {'label': 'Público (a)', 'value': 'a'},
                                    {'label': 'Páscoa (b)', 'value': 'b'},
                                    {'label': 'Natal (c)', 'value': 'c'}
                                ],
                                value='all',
                                multi=False, className="dropdown-dash", clearable=False # Refatorar classe CSS
                            ),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Feriado Escolar", html_for="filtro-feriado-escolar-3d", className="fw-bold mb-2"), # Refatorar ID
                            dcc.Dropdown(
                                id='filtro-feriado-escolar-3d', # Refatorar ID
                                options=[
                                    {'label': 'Todos', 'value': 'all'},
                                    {'label': 'Sim (1)', 'value': '1'},
                                    {'label': 'Não (0)', 'value': '0'}
                                ],
                                value='all',
                                multi=False, className="dropdown-dash", clearable=False # Refatorar classe CSS
                            ),
                        ], md=6)
                    ])
                ], md=6, className="py-3 px-4 d-flex flex-column justify-content-center"),
            ])
        ])
    ], className="custom-card mb-4")

def criar_card_filtros_analise_lojas(df_principal): # Refatorar nome da função e parâmetro
    """Cria um painel de filtros otimizado para a página de Análise de Lojas."""
    return dbc.Card([
        dbc.CardHeader(html.H4([html.I(className="fas fa-filter me-2"), "Painel de Filtros"], className="m-0 p-2 text-center fw-bold")),
        dbc.CardBody([
            dbc.Row([
                # Coluna 1: Filtros de escopo
                dbc.Col([
                    html.Div([
                        dbc.Label("Período de Análise", html_for="filtro-data", className="fw-bold"),
                        dcc.DatePickerRange(
                            id='filtro-data',
                            min_date_allowed=df_principal['Date'].min().date(), # Usar novo DataFrame
                            max_date_allowed=df_principal['Date'].max().date(), # Usar novo DataFrame
                            start_date=df_principal['Date'].min().date(), # Usar novo DataFrame
                            end_date=df_principal['Date'].max().date(), # Usar novo DataFrame
                            display_format='DD/MM/YYYY',
                            className="date-picker-custom w-100" # Refatorar classe CSS
                        ),
                    ], className="mb-3"),
                    html.Div([
                        dbc.Label("Tipo(s) de Loja", html_for="filtro-tipo-loja", className="fw-bold"), # Refatorar ID
                        dcc.Dropdown(
                            id='filtro-tipo-loja', # Refatorar ID
                            options=[{'label': t, 'value': t} for t in sorted(df_principal['StoreType'].unique())], # Usar novo DataFrame
                            value=sorted(df_principal['StoreType'].unique()), # Usar novo DataFrame
                            multi=True,
                            placeholder="Selecione tipos de loja",
                            className="dropdown-dash" # Refatorar classe CSS
                        ),
                    ], className="mb-3"),
                ], md=6, className="border-end-md py-3 px-4"),

                # Coluna 2: Filtros de loja e feriados
                dbc.Col([
                    html.Div([
                        dbc.Label("Loja(s) Específica(s)", html_for="filtro-loja-especifica", className="fw-bold mb-2"), # Refatorar ID
                        dcc.Dropdown(
                            id='filtro-loja-especifica', # Refatorar ID
                            options=[{'label': str(s), 'value': s} for s in sorted(df_principal['Store'].unique())], # Usar novo DataFrame
                            value=[],
                            multi=True,
                            placeholder="Busque por uma ou mais lojas...",
                            className="dropdown-dash" # Refatorar classe CSS
                        ),
                    ], className="mb-4"), # Aumentar margem inferior

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Feriado Estadual", html_for="filtro-feriado-estadual", className="fw-bold mb-2"), # Refatorar ID
                            dcc.Dropdown(
                                id='filtro-feriado-estadual', # Refatorar ID
                                options=[
                                    {'label': 'Todos', 'value': 'all'},
                                    {'label': 'Nenhum (0)', 'value': '0'},
                                    {'label': 'Público (a)', 'value': 'a'},
                                    {'label': 'Páscoa (b)', 'value': 'b'},
                                    {'label': 'Natal (c)', 'value': 'c'}
                                ],
                                value='all',
                                multi=False,
                                className="dropdown-dash", # Refatorar classe CSS
                                clearable=False
                            ),
                        ], md=6),
                        dbc.Col([
                            dbc.Label("Feriado Escolar", html_for="filtro-feriado-escolar", className="fw-bold mb-2"), # Refatorar ID
                            dcc.Dropdown(
                                id='filtro-feriado-escolar', # Refatorar ID
                                options=[
                                    {'label': 'Todos', 'value': 'all'},
                                    {'label': 'Sim (1)', 'value': '1'},
                                    {'label': 'Não (0)', 'value': '0'}
                                ],
                                value='all',
                                multi=False,
                                className="dropdown-dash", # Refatorar classe CSS
                                clearable=False
                            ),
                        ], md=6)
                    ])
                ], md=6, className="py-3 px-4"),
            ])
        ])
    ], className="custom-card mb-4")