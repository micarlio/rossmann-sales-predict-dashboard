import dash_bootstrap_components as dbc
from dash import dcc, html

from .componentes_compartilhados import criar_botoes_cabecalho

def criar_layout_previsao_vendas():
    nome_pagina = "previsao-vendas"
    return dbc.Container(
        [
            # Modal para salvar cenário
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle("Salvar Cenário")),
                dbc.ModalBody([
                    html.Div([
                        dbc.Label("Nome do Cenário"),
                        dbc.Input(
                            type="text",
                            id="input-nome-cenario",
                            placeholder="Ex: Aumento de Preços Q4",
                            className="mb-3"
                        ),
                        dbc.FormText(
                            "Digite um nome descritivo para identificar este cenário posteriormente.",
                            color="secondary"
                        )
                    ])
                ]),
                dbc.ModalFooter([
                    dbc.Button(
                        "Cancelar", 
                        id="btn-cancelar-salvar-cenario", 
                        className="me-2",
                        color="secondary"
                    ),
                    dbc.Button(
                        "Salvar", 
                        id="btn-confirmar-salvar-cenario",
                        color="primary"
                    )
                ])
            ], id="modal-salvar-cenario", is_open=False),

            # Toast para feedback
            dbc.Toast(
                "",
                id="toast-feedback-cenario",
                header="Notificação",
                is_open=False,
                dismissable=True,
                duration=4000,
                style={"position": "fixed", "top": 66, "right": 10, "width": 350}
            ),

            # Toast de feedback
            dbc.Toast(
                id="toast-whatif",
                header="Cenário Salvo",
                is_open=False,
                dismissable=True,
                duration=4000,
                icon="success",
                style={"position": "fixed", "top": 66, "right": 10, "width": 350}
            ),

            dcc.Store(id='armazenamento-metrica-selecionada'), # Armazena a métrica selecionada
            dcc.Store(id='armazenamento-forecast-diario'), # Armazena o forecast diário para cálculos consistentes
            dcc.Store(id='armazenamento-hist-diario'),
            
            # Tooltip de ajuda
            dbc.Tooltip(
                """
                Simule diferentes cenários alterando variáveis como:
                • Preço e promoções
                • Competição
                • Sazonalidade
                • Operação da loja
                
                Crie e salve seus próprios cenários para comparação posterior.
                """,
                target="tooltip-whatif",
                placement="left"
            ),
            
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
                                    min=1, max=54, step=1, value=4,
                                    marks={i: str(i) for i in range(1, 55, 9)},
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
                            # Coluna 1: Modelo
                            dbc.Col([
                                html.Label("Modelo", className="small fw-bold mb-1"),
                                dcc.Dropdown(
                                    id='dropdown-modelo-previsao',
                                    options=[
                                        {'label': 'Prophet', 'value': 'prophet'},
                                        {'label': 'Random Forest', 'value': 'random_forest'},
                                        {'label': 'XGBoost', 'value': 'xgboost'},
                                        {'label': 'LightGBM', 'value': 'lightgbm'},
                                        {'label': 'Ensemble', 'value': 'ensemble'}
                                    ],
                                    value='prophet', clearable=False, className="dropdown-sm"
                                )
                            ], md=3),

                            # Coluna 2: Granularidade
                            dbc.Col([
                                html.Label("Granularidade", className="small fw-bold mb-1"),
                                dcc.Dropdown(
                                    id='dropdown-granularidade-previsao',
                                    options=[
                                        {'label': 'Diária', 'value': 'diaria'},
                                        {'label': 'Semanal', 'value': 'semanal'},
                                        {'label': 'Mensal', 'value': 'mensal'}
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
                                children=html.Div([
                                    dcc.Graph(
                                        id='grafico-previsao',
                                        config={'displayModeBar': True, 'scrollZoom': True}
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
                                    html.H6("Média Semanal Prevista", className="card-title"),
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
                    ], className="mb-3"),
                    
                    # Simulador What-If
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Row([
                                dbc.Col(html.H6("Simulador What-If", className="m-0"), width="auto"),
                                dbc.Col(
                                    dbc.Button(
                                        html.I(className="fas fa-question-circle"),
                                        id="tooltip-whatif",
                                        color="link",
                                        size="sm",
                                        className="p-0 ms-2"
                                    ),
                                    width="auto"
                                )
                            ], align="center", className="g-0")
                        ]),
                        dbc.CardBody([
                            # Removendo a estrutura de abas e mantendo apenas o conteúdo do cenário personalizado
                            html.Div([
                                dbc.Row([
                                    # Variáveis de Simulação - Coluna 1
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-tags me-2"),
                                            "Var. de Preço"
                                        ], className="whatif-section-title"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Variação de Preço (%)", className="whatif-slider-label"),
                                                dcc.Slider(
                                                    id='slider-whatif-preco',
                                                    min=-50,
                                                    max=50,
                                                    value=0,
                                                    marks={
                                                        -50: '-50%',
                                                        -25: '-25%',
                                                        0: '0%',
                                                        25: '25%',
                                                        50: '50%'
                                                    },
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className="whatif-slider"
                                                )
                                            ], className="whatif-slider-container"),
                                            html.Div([
                                                html.Label("Variação de Promoção (p.p.)", className="whatif-slider-label"),
                                                dcc.Slider(
                                                    id='slider-whatif-promo',
                                                    min=-100,
                                                    max=100,
                                                    value=0,
                                                    marks={
                                                        -100: '-100',
                                                        -50: '-50',
                                                        0: '0',
                                                        50: '50',
                                                        100: '100'
                                                    },
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className="whatif-slider"
                                                )
                                            ], className="whatif-slider-container")
                                        ], className="whatif-section")
                                    ], md=6, className="pe-2"),
                                    
                                    # Variáveis de Simulação - Coluna 2
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-store me-2"),
                                            "Var. de Competição"
                                        ], className="whatif-section-title"),
                                        html.Div([
                                            html.Div([
                                                html.Label("Dist. do Competidor (km)", className="whatif-slider-label"),
                                                dcc.Slider(
                                                    id='slider-whatif-comp-dist',
                                                    min=0,
                                                    max=10,
                                                    value=5,
                                                    marks={
                                                        0: '0',
                                                        2: '2',
                                                        5: '5',
                                                        8: '8',
                                                        10: '10'
                                                    },
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className="whatif-slider"
                                                )
                                            ], className="whatif-slider-container"),
                                            html.Div([
                                                html.Label("Promoção do Competidor (%)", className="whatif-slider-label"),
                                                dcc.Slider(
                                                    id='slider-whatif-comp-promo',
                                                    min=0,
                                                    max=100,
                                                    value=0,
                                                    marks={
                                                        0: '0%',
                                                        25: '25%',
                                                        50: '50%',
                                                        75: '75%',
                                                        100: '100%'
                                                    },
                                                    tooltip={"placement": "bottom", "always_visible": True},
                                                    className="whatif-slider"
                                                )
                                            ], className="whatif-slider-container")
                                        ], className="whatif-section")
                                    ], md=6, className="ps-2"),
                                ], className="mb-3 g-0"),
                                
                                # Variáveis de Sazonalidade
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-calendar-alt me-2"),
                                            "Variáveis de Sazonalidade"
                                        ], className="whatif-section-title"),
                                        dbc.Row([
                                            dbc.Col([
                                                html.Label("Feriados", className="whatif-slider-label"),
                                                dbc.Checklist(
                                                    id='check-whatif-feriados',
                                                    options=[
                                                        {"label": "Considerar Feriados", "value": 1}
                                                    ],
                                                    value=[],
                                                    switch=True,
                                                )
                                            ], width=6, className="pe-2"),
                                            dbc.Col([
                                                html.Label("Eventos Especiais", className="whatif-slider-label"),
                                                dbc.Select(
                                                    id='select-whatif-eventos',
                                                    options=[
                                                        {"label": "Nenhum", "value": "none"},
                                                        {"label": "Volta às Aulas", "value": "back_to_school"},
                                                        {"label": "Natal", "value": "christmas"},
                                                        {"label": "Páscoa", "value": "easter"}
                                                    ],
                                                    value="none",
                                                    size="sm"
                                                )
                                            ], width=6, className="ps-2")
                                        ], className="g-0")
                                    ], className="whatif-section")
                                ], className="g-0"),
                            ], className="mb-3"),
                            
                            # Botões de Ação
                            dbc.Row([
                                dbc.Col(
                                    dbc.Button([
                                        html.I(className="fas fa-play me-2"),
                                        "Simular"
                                    ], id="btn-simular-whatif", 
                                       color="primary",
                                       size="sm",
                                       className="btn-whatif w-100"),
                                    width=6
                                ),
                                dbc.Col(
                                    dbc.Button([
                                        html.I(className="fas fa-save me-2"),
                                        "Salvar Cenário"
                                    ], id="btn-salvar-whatif", 
                                       color="outline-primary",
                                       size="sm",
                                       className="btn-whatif w-100"),
                                    width=6
                                )
                            ], className="mb-3"),
                            
                            # Resultados da Simulação
                            html.Div([
                                # KPIs Principais
                                dbc.Row([
                                    dbc.Col([
                                        html.Div("Total Base:", className="small text-muted"),
                                        html.Div(id="kpi-total-base", className="h6")
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Total Simulado:", className="small text-muted"),
                                        html.Div(id="kpi-total-sim", className="h6")
                                    ], width=4),
                                    dbc.Col([
                                        html.Div("Variação:", className="small text-muted"),
                                        html.Div(id="kpi-variacao-sim", className="h6")
                                    ], width=4)
                                ], className="mb-3"),
                                
                                # Gráfico Comparativo
                                dcc.Graph(id='grafico-whatif', config={'displayModeBar': False}),
                                
                                # Insights Automáticos
                                html.Div(id="insights-whatif", className="mt-3")
                            ], id="resultados-whatif", style={"display": "none"}),
                            
                            # Nova seção: Comparação de Cenários
                            html.Div([
                                html.Div([
                                    html.I(className="fas fa-chart-line me-2"),
                                    "Comparação de Cenários"
                                ], className="whatif-section-title mt-4 mb-3"),
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Cenários Salvos:", className="whatif-slider-label mb-2"),
                                            dcc.Dropdown(
                                                id="dropdown-cenarios",
                                                multi=True,
                                                placeholder="Selecione os cenários para comparar",
                                                className="mb-3"
                                            )
                                        ], width=12),
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dbc.Button([
                                                html.I(className="fas fa-chart-line me-2"),
                                                "Comparar Cenários"
                                            ], id="btn-comparar-cenarios",
                                               color="outline-primary",
                                               size="sm",
                                               className="btn-whatif w-100"),
                                            width=6
                                        ),
                                        dbc.Col(
                                            dbc.Button([
                                                html.I(className="fas fa-trash-alt me-2"),
                                                "Limpar Cenários"
                                            ], id="btn-limpar-cenarios",
                                               color="outline-danger",
                                               size="sm",
                                               className="btn-whatif w-100"),
                                            width=6
                                        )
                                    ]),
                                    html.Div(id="resultados-comparacao", style={"display": "none"}),
                                    dcc.ConfirmDialog(
                                        id='confirm-limpar-cenarios',
                                        message='Você tem certeza que deseja apagar TODOS os cenários salvos? Esta ação não pode ser desfeita.'
                                    ),
                                    dcc.Store(id='store-cenarios-update-trigger')
                                ], className="comparacao-cenarios")
                            ])
                        ])
                    ])
                ], md=4)
            ])
        ],
        fluid=True,
        className="previsao-vendas-container"
    )