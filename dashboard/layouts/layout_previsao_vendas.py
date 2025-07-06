import dash_bootstrap_components as dbc
from dash import dcc, html

from .componentes_compartilhados import criar_botoes_cabecalho, criar_card_filtros # Importar o componente de filtros

def criar_layout_previsao_vendas(): # Refatorar nome da função
    nome_pagina = "previsao-vendas" # Refatorar nome da variável
    return dbc.Container(
        [
            # Cabeçalho da Página
            dbc.Row(
                [
                    dbc.Col(html.H1("Modelagem e Previsão de Vendas", className="page-title"), md=8),
                    dbc.Col(criar_botoes_cabecalho(nome_pagina), md=4, className="d-flex justify-content-end"), # Usar nova função e variável refatorada
                ],
                align="center",
                className="mb-4"
            ),
            # Painel de Filtros
            dbc.Card([
                dbc.CardHeader(html.H4([html.I(className="fas fa-filter me-2"), "Painel de Filtros"], className="m-0 p-2 text-center fw-bold")),
                dbc.CardBody([
                    dbc.Row([
                        # Coluna 1: Filtros de Previsão
                        dbc.Col([
                            html.Div([
                                dbc.Label("Métrica de Previsão", className="fw-bold mb-2"),
                                dbc.RadioItems(
                                    id='radio-metrica-previsao',
                                    options=[
                                        {'label': 'Vendas', 'value': 'Sales'},
                                        {'label': 'Clientes', 'value': 'Customers'},
                                        {'label': 'Ticket Médio', 'value': 'SalesPerCustomer'}
                                    ],
                                    value='Sales',
                                    className="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                ),
                                dbc.Label("Modelo de Previsão", className="fw-bold mb-2 mt-3"),
                                dcc.Dropdown(
                                    id='dropdown-modelo-previsao',
                                    options=[
                                        {'label': 'Prophet', 'value': 'prophet'},
                                        {'label': 'Random Forest', 'value': 'random_forest'},
                                        {'label': 'ARIMA', 'value': 'arima'},
                                        {'label': 'XGBoost', 'value': 'xgboost'},
                                        {'label': 'LightGBM', 'value': 'lightgbm'},
                                        {'label': 'Ensemble', 'value': 'ensemble'}
                                    ],
                                    value='prophet',
                                    clearable=False,
                                    className="mb-3"
                                ),
                                dbc.Label("Horizonte de Previsão (períodos)", className="fw-bold mb-2"),
                                dcc.Slider(
                                    id='slider-horizonte-previsao', 
                                    min=1, 
                                    max=30, 
                                    step=1, 
                                    value=7,
                                    marks={i: {'label': str(i), 'style': {'color': '#77b0b1'}} for i in range(1, 31, 5)},
                                    className="mb-3"
                                ),
                                dbc.Label("Granularidade", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='dropdown-granularidade-previsao',
                                    options=[
                                        {'label': 'Diária', 'value': 'diaria'},
                                        {'label': 'Semanal', 'value': 'semanal'},
                                        {'label': 'Mensal', 'value': 'mensal'}
                                    ],
                                    value='diaria',
                                    clearable=False,
                                    className="mb-1"
                                )
                            ], className="mb-3")
                        ], md=4),
                        
                        # Coluna 2: Filtros de Lojas
                        dbc.Col([
                            html.Div([
                                dbc.Label("Tipo(s) de Loja", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='dropdown-tipo-loja',
                                    options=[
                                        {'label': 'Tipo A', 'value': 'a'},
                                        {'label': 'Tipo B', 'value': 'b'},
                                        {'label': 'Tipo C', 'value': 'c'},
                                        {'label': 'Tipo D', 'value': 'd'}
                                    ],
                                    value=['a', 'b', 'c', 'd'],
                                    multi=True,
                                    placeholder='Selecione tipos de loja',
                                    className='dropdown-dash'
                                ),
                                dbc.Label("Loja(s) Específica(s)", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='dropdown-lojas-previsao',
                                    options=[], 
                                    multi=True,
                                    value=[],
                                    placeholder='Busque por uma ou mais lojas...',
                                    className='dropdown-dash'
                                )
                            ])
                        ], md=4),
                        
                        # Coluna 3: Filtros Temporais
                        dbc.Col([
                            html.Div([
                                dbc.Label("Dias da Semana", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='checklist-dias-semana',
                                    options=[
                                        {'label': 'Todos os dias', 'value': 'todos'},
                                        {'label': 'Segunda-feira', 'value': 1},
                                        {'label': 'Terça-feira', 'value': 2},
                                        {'label': 'Quarta-feira', 'value': 3},
                                        {'label': 'Quinta-feira', 'value': 4},
                                        {'label': 'Sexta-feira', 'value': 5},
                                        {'label': 'Sábado', 'value': 6},
                                        {'label': 'Domingo', 'value': 7}
                                    ],
                                    value='todos',
                                    multi=True,
                                    placeholder='Selecione os dias da semana',
                                    className="mb-3"
                                ),
                                dbc.Label("Status de Promoção", className="fw-bold mb-2"),
                                dcc.Dropdown(
                                    id='dropdown-promocao',
                                    options=[
                                        {'label': 'Todos', 'value': 'todos'},
                                        {'label': 'Com promoção', 'value': 1},
                                        {'label': 'Sem promoção', 'value': 0}
                                    ],
                                    value='todos',
                                    clearable=False,
                                    className="mb-1"
                                )
                            ])
                        ], md=4)
                    ])
                ])
            ], className="mb-4"),
            
            # Parâmetros Específicos de Modelo
            html.Div(id='parametros-modelo', children=[
                html.Div(id='parametros-arima', children=[
                    html.Label('ARIMA Order (p,d,q)'),
                    dcc.Input(id='arima-p', type='number', placeholder='p', value=1, min=0, style={'width':'60px'}),
                    dcc.Input(id='arima-d', type='number', placeholder='d', value=1, min=0, style={'width':'60px', 'marginLeft':'10px'}),
                    dcc.Input(id='arima-q', type='number', placeholder='q', value=1, min=0, style={'width':'60px', 'marginLeft':'10px'})
                ], style={'display':'none', 'marginBottom':'20px'}),
                html.Div(id='parametros-xgboost', children=[
                    html.Label('XGBoost n_estimators'),
                    dcc.Slider(id='xgb-estimators', min=10, max=500, step=10, value=100,
                               marks={i:str(i) for i in range(50,501,50)}),
                    html.Label('XGBoost learning_rate', style={'marginTop':'10px'}),
                    dcc.Input(id='xgb-lr', type='number', placeholder='0.1', value=0.1, step=0.01)
                ], style={'display':'none', 'marginBottom':'20px'}),
                html.Div(id='parametros-lightgbm', children=[
                    html.Label('LightGBM n_estimators'),
                    dcc.Slider(id='lgbm-estimators', min=10, max=500, step=10, value=100,
                               marks={i:str(i) for i in range(50,501,50)})
                ], style={'display':'none', 'marginBottom':'20px'}),
                html.Div(id='parametros-ensemble', children=[
                    html.Label('Pesos do Ensemble'),
                    html.P('Disponível após seleção dos modelos Prophet e RF'),
                ], style={'display':'none', 'marginBottom':'20px'})
            ], className='mb-4'),
            
            # Gráfico de Previsão
            dbc.Card([
                dbc.CardBody([
                    # Gráfico de Previsão com botão nativo de download (ícone de câmera)
                    dcc.Graph(
                        id='grafico-previsao',
                        config={
                            'modeBarButtonsToAdd': ['toImage'],
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'grafico_previsao',
                                'height': 600,
                                'width': 800,
                                'scale': 1
                            }
                        }
                    ),
                    # Área de resultados: Tabela e Principais Informações
                    dbc.Row([
                        # Tabela de Previsão com scroll dinâmico
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5("Tabela de Previsão", className="card-title fw-bold m-0")),
                                dbc.CardBody(
                                    dcc.Loading(type="circle", children=html.Div(id="tabela-previsao")),
                                    style={'max-height': 'calc(60vh)', 'overflow-y': 'auto'},
                                    className="p-3"
                                )
                            ], className="custom-card"),
                            md=6, className="mb-4"
                        ),
                        # Principais Informações Calculadas
                        dbc.Col(
                            dbc.Card([
                                dbc.CardHeader(html.H5("Principais Informações", className="card-title fw-bold m-0")),
                                dbc.CardBody(
                                    dcc.Loading(
                                        type="circle", 
                                        children=html.Div(id="informacoes-previsao"),
                                    )
                                )
                            ], className="custom-card"),
                            md=6, className="mb-4"
                        )
                    ])
                ])
            ], className='custom-card'),
            # Seção de Inteligência de Negócio e Impacto Operacional
            html.Div([
                html.H3('Inteligência de Negócio e Impacto Operacional', className='section-subtitle'),
                dbc.Row([
                    # Módulo de Otimização de Estoque
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.H5('Módulo de Otimização de Estoque')),
                            dbc.CardBody(html.Div(id='otimizacao-estoque-conteudo'))
                        ], className='mb-4'),
                        width=12
                    ),
                    # Calculadora de ROI de Marketing
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.H5('Calculadora de ROI de Marketing')),
                            dbc.CardBody([
                                html.Label('Custo da Campanha (€)'),
                                dcc.Input(id='input-custo-campanha', type='number', min=0, step=0.01, value=0),
                                html.Br(),
                                html.Label('Período de Simulação (dias)'),
                                dcc.Input(id='input-periodo-campanha', type='number', min=1, step=1, value=7),
                                html.Br(),
                                dbc.Button('Simular Campanha', id='btn-simular-campanha', color='primary', className='mt-2'),
                                html.Div(id='roi-marketing-output', className='mt-3')
                            ])
                        ], className='mb-4'),
                        width=12
                    ),
                    # Análise de Elasticidade de Preço
                    dbc.Col(
                        dbc.Card([
                            dbc.CardHeader(html.H5('Análise de Elasticidade de Preço')),
                            dbc.CardBody([
                                dbc.Button('Abrir Análise de Elasticidade', id='btn-open-modal-elasticidade', color='secondary'),
                                dbc.Modal([
                                    dbc.ModalHeader('Análise de Elasticidade de Preço'),
                                    dbc.ModalBody([
                                        html.Label('% de Alteração de Preço'),
                                        dcc.Input(id='input-alteracao-preco', type='number', step=0.01, value=0),
                                        html.Br(),
                                        dbc.Button('Simular Alteração', id='btn-simular-preco', color='primary', className='mt-2'),
                                        html.Div([
                                            dcc.Graph(id='grafico-elasticidade-vendas'),
                                            dcc.Graph(id='grafico-elasticidade-receita'),
                                        ], className='mt-4 d-flex justify-content-between')
                                    ]),
                                    dbc.ModalFooter(
                                        dbc.Button('Fechar', id='btn-close-modal-elasticidade')
                                    )
                                ], id='modal-elasticidade', is_open=False)
                            ])
                        ], className='mb-4'),
                        width=12
                    )
                ])
            ])
        ],
        fluid=True,
        className="p-4 page-content"
    )