import dash_bootstrap_components as dbc
from dash import dcc, html

from .componentes_compartilhados import criar_botoes_cabecalho

def criar_layout_previsao_vendas():
    nome_pagina = "previsao-vendas"
    return dbc.Container(
        [
            # Cabeçalho mais compacto
            dbc.Row(
                [
                    dbc.Col(html.H1("Modelagem e Previsão de Vendas", className="page-title mb-3"), md=8),
                    dbc.Col(criar_botoes_cabecalho(nome_pagina), md=4, className="d-flex justify-content-end"),
                ],
                align="center",
                className="mb-3"
            ),
            
            # Painel de Filtros Compacto
            dbc.Card([
                dbc.CardHeader(
                    dbc.Row([
                        dbc.Col(html.H5("Filtros", className="m-0"), className="d-flex align-items-center"),
                        dbc.Col(
                            dbc.Button(
                                html.I(className="fas fa-chevron-up"),
                                id="btn-toggle-filtros",
                                color="link",
                                className="p-0 float-end",
                                n_clicks=0
                            ),
                            width="auto"
                        )
                    ], className="g-0")
                ),
                dbc.Collapse(
                    dbc.CardBody([
                        dbc.Row([
                            # Coluna 1: Filtros principais
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Métrica", className="small fw-bold mb-1"),
                                        dbc.RadioItems(
                                            id='radio-metrica-previsao',
                                            options=[
                                                {'label': 'Vendas', 'value': 'Sales'},
                                                {'label': 'Clientes', 'value': 'Customers'},
                                                {'label': 'Ticket Médio', 'value': 'SalesPerCustomer'}
                                            ],
                                            value='Sales',
                                            className="btn-group btn-group-sm",
                                            inputClassName="btn-check",
                                            labelClassName="btn btn-outline-primary btn-sm"
                                        )
                                    ], width=12, className="mb-2"),
                                    dbc.Col([
                                        html.Label("Modelo", className="small fw-bold mb-1"),
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
                                            className="dropdown-sm"
                                        )
                                    ], width=12, className="mb-2")
                                ])
                            ], md=3),
                            
                            # Coluna 2: Filtros de tempo
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Horizonte", className="small fw-bold mb-1"),
                                        dcc.Slider(
                                            id='slider-horizonte-previsao',
                                            min=1,
                                            max=30,
                                            step=1,
                                            value=7,
                                            marks={i: str(i) for i in range(1, 31, 5)},
                                            className="mb-3"
                                        )
                                    ], width=12),
                                    dbc.Col([
                                        html.Label("Granularidade", className="small fw-bold mb-1"),
                                        dcc.Dropdown(
                                            id='dropdown-granularidade-previsao',
                                            options=[
                                                {'label': 'Diária', 'value': 'diaria'},
                                                {'label': 'Semanal', 'value': 'semanal'},
                                                {'label': 'Mensal', 'value': 'mensal'}
                                            ],
                                            value='diaria',
                                            clearable=False,
                                            className="dropdown-sm"
                                        )
                                    ], width=12)
                                ])
                            ], md=3),
                            
                            # Coluna 3: Filtros de loja
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Tipo de Loja", className="small fw-bold mb-1"),
                                        dcc.Dropdown(
                                            id='dropdown-tipo-loja',
                                            options=[
                                                {'label': 'Tipo A', 'value': 'a'},
                                                {'label': 'Tipo B', 'value': 'b'},
                                                {'label': 'Tipo C', 'value': 'c'},
                                                {'label': 'Tipo D', 'value': 'd'}
                                            ],
                                            value=['a'],
                                            multi=True,
                                            className="dropdown-sm"
                                        )
                                    ], width=12, className="mb-2"),
                                    dbc.Col([
                                        html.Label("Lojas Específicas", className="small fw-bold mb-1"),
                                        dcc.Dropdown(
                                            id='dropdown-lojas-previsao',
                                            options=[],
                                            multi=True,
                                            placeholder="Selecione...",
                                            className="dropdown-sm"
                                        )
                                    ], width=12)
                                ])
                            ], md=3),
                            
                            # Coluna 4: Outros filtros
                            dbc.Col([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Dias da Semana", className="small fw-bold mb-1"),
                                        dcc.Dropdown(
                                            id='checklist-dias-semana',
                                            options=[
                                                {'label': 'Todos', 'value': 'todos'},
                                                {'label': 'Segunda', 'value': 1},
                                                {'label': 'Terça', 'value': 2},
                                                {'label': 'Quarta', 'value': 3},
                                                {'label': 'Quinta', 'value': 4},
                                                {'label': 'Sexta', 'value': 5},
                                                {'label': 'Sábado', 'value': 6},
                                                {'label': 'Domingo', 'value': 7}
                                            ],
                                            value='todos',
                                            multi=True,
                                            className="dropdown-sm"
                                        )
                                    ], width=12, className="mb-2"),
                                    dbc.Col([
                                        html.Label("Promoção", className="small fw-bold mb-1"),
                                        dcc.Dropdown(
                                            id='dropdown-promocao',
                                            options=[
                                                {'label': 'Todos', 'value': 'todos'},
                                                {'label': 'Com Promoção', 'value': 1},
                                                {'label': 'Sem Promoção', 'value': 0}
                                            ],
                                            value='todos',
                                            clearable=False,
                                            className="dropdown-sm"
                                        )
                                    ], width=12)
                                ])
                            ], md=3)
                        ])
                    ]),
                    id="collapse-filtros",
                    is_open=True
                )
            ], className="mb-3"),

            # Parâmetros de Modelo (inicialmente ocultos)
            html.Div(id='parametros-modelo', className="mb-3", children=[
                html.Div(id='parametros-arima', style={'display': 'none'}, children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label('ARIMA (p,d,q)', className="small fw-bold"),
                            dbc.Row([
                                dbc.Col(dcc.Input(id='arima-p', type='number', placeholder='p', value=1, min=0, className="form-control form-control-sm"), width=4),
                                dbc.Col(dcc.Input(id='arima-d', type='number', placeholder='d', value=1, min=0, className="form-control form-control-sm"), width=4),
                                dbc.Col(dcc.Input(id='arima-q', type='number', placeholder='q', value=1, min=0, className="form-control form-control-sm"), width=4)
                            ])
                        ], width=6)
                    ])
                ]),
                html.Div(id='parametros-xgboost', style={'display': 'none'}, children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label('XGBoost', className="small fw-bold"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("N° Estimadores", className="small"),
                                    dcc.Slider(id='xgb-estimators', min=10, max=500, step=10, value=100)
                                ], width=8),
                                dbc.Col([
                                    html.Label("Learning Rate", className="small"),
                                    dcc.Input(id='xgb-lr', type='number', value=0.1, step=0.01, className="form-control form-control-sm")
                                ], width=4)
                            ])
                        ])
                    ])
                ]),
                html.Div(id='parametros-lightgbm', style={'display': 'none'}, children=[
                    dbc.Row([
                        dbc.Col([
                            html.Label('LightGBM', className="small fw-bold"),
                            html.Label("N° Estimadores", className="small"),
                            dcc.Slider(id='lgbm-estimators', min=10, max=500, step=10, value=100)
                        ])
                    ])
                ]),
                html.Div(id='parametros-ensemble', style={'display': 'none'})
            ]),

            # Grid de Visualizações
            dbc.Row([
                # Coluna da Esquerda (8/12)
                dbc.Col([
                    # Gráfico Principal
                    dbc.Card([
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading-grafico-previsao",
                                type="circle",
                                children=dcc.Graph(
                                    id='grafico-previsao',
                                    config={'displayModeBar': True, 'scrollZoom': True}
                                )
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Grid de Gráficos Secundários
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Distribuição por Dia da Semana", className="card-title"),
                                    dcc.Graph(id='grafico-empilhado', config={'displayModeBar': False})
                                ])
                            ]),
                            md=6
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Heatmap de Previsão", className="card-title"),
                                    dcc.Graph(id='heatmap-calendario', config={'displayModeBar': False})
                                ])
                            ]),
                            md=6
                        )
                    ], className="mb-3"),
                    
                    # Gráfico Comparativo
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Comparativo de Métricas", className="card-title"),
                            dcc.Graph(id='grafico-comparativo', config={'displayModeBar': False})
                        ])
                    ])
                ], md=8),
                
                # Coluna da Direita (4/12)
                dbc.Col([
                    # Painel de Informações
                    dbc.Card([
                        dbc.CardHeader(html.H6("Principais Informações", className="m-0")),
                        dbc.CardBody(
                            dcc.Loading(
                                id="loading-info",
                                type="circle",
                                children=html.Div(id="informacoes-previsao", className="info-panel-grid")
                            )
                        )
                    ], className="mb-3"),
                    
                    # Tabela de Previsão
                    dbc.Card([
                        dbc.CardHeader(html.H6("Tabela de Previsão", className="m-0")),
                        dbc.CardBody(
                            dcc.Loading(
                                id="loading-tabela",
                                type="circle",
                                children=html.Div(id="tabela-previsao", className="tabela-compacta")
                            ),
                            style={"maxHeight": "400px", "overflowY": "auto"}
                        )
                    ], className="mb-3"),
                    
                    # Simulador What-If
                    dbc.Card([
                        dbc.CardHeader(html.H6("Simulador What-If", className="m-0")),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Variação de Preço (%)", className="small fw-bold mb-1"),
                                    dcc.Slider(id='slider-whatif-preco', min=-50, max=50, value=0, marks={i: f"{i}%" for i in range(-50, 51, 10)})
                                ], width=12, className="mb-3"),
                                dbc.Col([
                                    html.Label("Variação de Promoção (p.p.)", className="small fw-bold mb-1"),
                                    dcc.Slider(id='slider-whatif-promo', min=-100, max=100, value=0, marks={i: f"{i}%" for i in range(-100, 101, 20)})
                                ], width=12, className="mb-3"),
                                dbc.Col([
                                    dbc.Button("Simular", id="btn-simular-whatif", color="primary", size="sm", className="w-100")
                                ], width=12, className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Total Base:", className="small text-muted"),
                                        html.Div(id="kpi-total-base", className="h6")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div("Total Simulado:", className="small text-muted"),
                                        html.Div(id="kpi-total-sim", className="h6")
                                    ], width=6)
                                ]),
                                dbc.Col(
                                    dcc.Graph(id='grafico-whatif', config={'displayModeBar': False}),
                                    width=12
                                )
                            ])
                        ])
                    ])
                ], md=4)
            ])
        ],
        fluid=True,
        className="previsao-vendas-container"
    )