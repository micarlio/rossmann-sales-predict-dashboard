import dash_bootstrap_components as dbc
from dash import dcc, html

from .componentes_compartilhados import criar_botoes_cabecalho

def criar_layout_previsao_vendas():
    nome_pagina = "previsao-vendas"
    return dbc.Container(
        [
            # (Remover todas as linhas com ids ou tooltips relacionados ao simulador what-if e cenários)

            dcc.Store(id='armazenamento-metrica-selecionada'), # Armazena a métrica selecionada
            dcc.Store(id='armazenamento-forecast-diario'), # Armazena o forecast diário para cálculos consistentes
            dcc.Store(id='armazenamento-hist-diario'),
            
            # Cabeçalho mais compacto
            dbc.Row(
                [
                    dbc.Col(html.H1("Modelagem e Previsão", className="page-title mb-3"), md=8),
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
                                html.I(className="fas fa-chevron-up", id="icone-toggle-filtros"),
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
                        # Linha 1 de Filtros
                        dbc.Row([
                            # Coluna 1: Métrica
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
                                    inline=True,
                                    className="grupo-radio",
                                    labelClassName="radio-item-custom",
                                    inputClassName="radio-input-custom"
                                )
                            ], md=3),

                            # Coluna 2: Horizonte
                            dbc.Col([
                                html.Label("Horizonte (Semanas)", className="small fw-bold mb-1"),
                                dcc.Slider(
                                    id='slider-horizonte-previsao',
                                    min=1, max=52, step=1, value=4,
                                    marks={i: str(i) for i in [1] + list(range(13, 53, 13))},
                                )
                            ], md=3),

                            # Coluna 3: Tipo de Loja
                            dbc.Col([
                                html.Label("Tipo de Loja", className="small fw-bold mb-1"),
                                dcc.Dropdown(
                                    id='dropdown-tipo-loja',
                                    options=[
                                        {'label': 'Todos', 'value': 'todos'},
                                        {'label': 'Tipo A', 'value': 'a'},
                                        {'label': 'Tipo B', 'value': 'b'},
                                        {'label': 'Tipo C', 'value': 'c'},
                                        {'label': 'Tipo D', 'value': 'd'}
                                    ],
                                    value=['todos'], multi=True, className="dropdown-sm"
                                )
                            ], md=3),

                            # Coluna 4: Dias da Semana
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
                                    value='todos', multi=True, className="dropdown-sm"
                                )
                            ], md=3),
                        ], className="mb-3"), # Espaçamento entre as linhas de filtros

                        # Linha 2 de Filtros
                        dbc.Row([
                            # Coluna 2: Granularidade
                            dbc.Col([
                                html.Label("Granularidade", className="small fw-bold mb-1"),
                                dcc.Dropdown(
                                    id='dropdown-granularidade-previsao',
                                    options=[
                                        {'label': 'Diária', 'value': 'diaria'},
                                        {'label': 'Semanal', 'value': 'semanal'}
                                    ],
                                    value='semanal', clearable=False, className="dropdown-sm"
                                )
                            ], md=3),

                            # Coluna 3: Lojas Específicas
                            dbc.Col([
                                html.Label("Lojas Específicas", className="small fw-bold mb-1"),
                                dcc.Dropdown(
                                    id='dropdown-lojas-previsao',
                                    options=[], multi=True, placeholder="Selecione...",
                                    className="dropdown-sm"
                                )
                            ], md=3),

                            # Coluna 4: Promoção
                            dbc.Col([
                                html.Label("Promoção", className="small fw-bold mb-1"),
                                dcc.Dropdown(
                                    id='dropdown-promocao',
                                    options=[
                                        {'label': 'Todos', 'value': 'todos'},
                                        {'label': 'Com Promoção', 'value': 1},
                                        {'label': 'Sem Promoção', 'value': 0}
                                    ],
                                    value='todos', clearable=False, className="dropdown-sm"
                                )
                            ], md=3),
                        ])
                    ]),
                    id="collapse-filtros",
                    is_open=True
                )
            ], className="mb-3"),

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
                                children=html.Div([
                                    dcc.Graph(
                                        id='grafico-previsao',
                                        config={'displayModeBar': False}
                                    ),
                                    html.Div(id='metricas-previsao', className="mt-2 ms-1")
                                ])
                            )
                        ])
                    ], className="mb-3"),
                    
                    # Grid de Gráficos Secundários
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Histórico e Previsão Semanal", className="card-title"),
                                    dcc.Graph(id='grafico-media-global', config={'displayModeBar': False})
                                ])
                            ]),
                            md=6
                        ),
                        dbc.Col(
                            dbc.Card([
                                dbc.CardBody([
                                    html.H6("Previsão por Tipo de Loja", className="card-title"),
                                    dcc.Graph(id='grafico-media-tipo-loja', config={'displayModeBar': False})
                                ])
                            ]),
                            md=6
                        )
                    ], className="mb-3"),
                    
                    # (Remover todo o bloco do Simulador What-If e elementos relacionados, incluindo ids, tooltips, stores, botões, sliders, gráficos, KPIs, dropdowns, modais, toasts e seções de comparação de cenários)
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
                            ),
                            style={"maxHeight": "500px", "overflowY": "auto"}
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
                    ])
                ], md=4)
            ])
        ],
        fluid=True,
        className="previsao-vendas-container"
    )