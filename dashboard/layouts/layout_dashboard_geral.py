import dash_bootstrap_components as dbc
from dash import dcc, html

from .componentes_compartilhados import criar_card_filtros, criar_card_grafico
from ..utils import criar_icone_informacao

def criar_layout_dashboard_analise(dados):
    nome_pagina = "dashboard"
    df_principal = dados["df_principal"]

    card_filtros = criar_card_filtros(df_principal, prefix='dashboard-')

    return html.Div([
        html.Div([
            html.H1("Visão Geral e Análise de Vendas", className="page-title"), # Título atualizado
            html.Div([
                html.Button([html.I(className="fas fa-download me-2"), "Exportar Parquet"], id='dashboard-botao-baixar-dados-filtrados', className='btn btn-primary-custom'),
                dcc.Download(id="download-df-parquet"),
                html.Button(html.I(className="fas fa-sync-alt"), id='dashboard-botao-resetar-filtros', className='btn btn-outline-secondary ms-2', title='Resetar Filtros'),
                html.Button(html.I(className="fas fa-expand-alt"), id={'type': 'botao-tela-cheia', 'index': nome_pagina}, className='btn btn-outline-secondary ms-2', title='Tela Cheia'),
                dcc.Store(id={'type': 'saida-tela-cheia', 'index': nome_pagina})
            ], className="d-flex align-items-center"),
        ], className="d-flex justify-content-between align-items-center mb-4"),
        card_filtros,

        html.H3("KPIs Globais de Desempenho", className="section-subtitle mt-4"),
        dbc.Row(id='linha-kpi-dashboard', className='mb-4 g-4'),
        html.Div(id='alerta-vendas-clientes-zero', className="mb-4 p-3 alert-box", style={'display': 'none'}),

        html.H3("Desempenho por Tipo de Loja", className="section-subtitle mt-4"),
        dbc.Row(id='linha-kpi-tipo-loja', className='mb-4 g-4'),

        html.H2(["Tendências Temporais", criar_icone_informacao("info-temporal", "Analisam a performance ao longo do tempo (diário, semanal, mensal) para identificar sazonalidades e tendências de longo prazo.")], className="page-title mt-5"),
        html.Hr(className="section-hr"),
        dbc.Row([
            criar_card_grafico(
                'grafico-vendas-clientes-tempo-dashboard',
                'analise-vendas-clientes-tempo',
                12,
                controles_extras=html.Div([
                    dbc.Label("Granularidade Temporal", html_for="filtro-granularidade", className="fw-bold mb-2"),
                    dbc.RadioItems(
                        id='filtro-granularidade',
                        options=[
                            {'label': 'Mensal', 'value': 'M'},
                            {'label': 'Semanal', 'value': 'W'},
                            {'label': 'Diário (Suavizado)', 'value': 'D'}
                        ],
                        value='M',
                        inline=True,
                        className="grupo-radio",
                        labelClassName="radio-item-custom",
                        inputClassName="radio-input-custom"
                    ),
                ], className="mb-4")
            )
        ], className='mb-4'),
        dbc.Row([
            criar_card_grafico('grafico-vendas-clientes-mensal-dashboard', 'analise-vendas-clientes-mensal', 6),
            criar_card_grafico('grafico-vendas-clientes-anual-dashboard', 'analise-vendas-clientes-anual', 6)
        ], className='mb-4 g-4'),

        html.H2(["Padrões Semanais e Mensais", criar_icone_informacao("info-padroes", "Examina a performance em ciclos curtos, como dias da semana ou dias do mês, para encontrar padrões de consumo recorrentes.")], className="page-title mt-5"),
        html.Hr(className="section-hr"),
        dbc.Row([
            criar_card_grafico('grafico-dia-semana-dashboard', 'analise-dia-semana', 6),
            criar_card_grafico('grafico-dia-dashboard', 'analise-dia', 6)
        ], className='mb-4 g-4'),

        html.H2(["Impacto de Promoções", criar_icone_informacao("info-promocoes", "Avalia a eficácia das promoções diárias (Promo) nas métricas de vendas, comparando dias com e sem promoção.")], className="page-title mt-5"),
        html.Hr(className="section-hr"),
        dbc.Row([
            criar_card_grafico('grafico-promocao-por-tipo-loja-dashboard', 'analise-promocao-por-tipo-loja-dashboard', 6),
            criar_card_grafico('grafico-impacto-promocao-por-tipo-loja-boxplot', 'analise-impacto-promocao-por-tipo-loja-boxplot', 6)
        ], className='mb-4 g-4'),
        dbc.Row([
            criar_card_grafico('grafico-impacto-promocao-geral-boxplot', 'analise-impacto-promocao-boxplot', 6),
            criar_card_grafico('grafico-impacto-promocao-geral-hist', 'analise-impacto-promocao-hist', 6)
        ], className='mb-4 g-4'),

        html.H2(["Análise de Comportamento do Cliente", criar_icone_informacao("info-comportamento", "Estes gráficos focam em como os clientes se comportam (quantos vêm à loja e quanto gastam em média) em vez de apenas olhar para as vendas totais.")], className="page-title mt-5"),
        html.Hr(className="section-hr"),
        dbc.Row([
            # Coluna 1: Comportamento por Promoção (Filtros removidos aqui)
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H5("Impacto da Promoção no Comportamento", className="m-0 p-1 text-center fw-bold")),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id='seletor-metrica-comportamento-promocao',
                            options=[
                                {'label': 'Ticket Médio', 'value': 'SalesPerCustomer'},
                                {'label': 'Nº de Clientes', 'value': 'Customers'},
                                {'label': 'Vendas', 'value': 'Sales'},
                            ],
                            value='SalesPerCustomer',
                            inline=True,
                            className="grupo-radio",
                            labelClassName="radio-item-custom",
                            inputClassName="radio-input-custom"
                        ),
                        dcc.Loading(dcc.Graph(id='grafico-comportamento-promocao-boxplot', config={'displayModeBar': False})),
                        html.Div([
                            html.P([
                                html.I(className="fas fa-lightbulb me-2"),
                                html.Strong("Análise: "),
                                html.Span(id='analise-comportamento-promocao-boxplot')
                            ])
                        ], className="analise-text-box mt-3")
                    ])
                ], className="custom-card h-100"),
                md=6, className="graph-card-col"
            ),

            # Coluna 2: Comportamento por Sortimento (Filtros removidos aqui)
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(html.H5("Métricas por Tipo de Sortimento", className="m-0 p-1 text-center fw-bold")),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id='seletor-metrica-sortimento',
                            options=[
                                {'label': 'Ticket Médio', 'value': 'SalesPerCustomer'},
                                {'label': 'Nº de Clientes', 'value': 'Customers'},
                                {'label': 'Vendas', 'value': 'Sales'},
                            ],
                            value='SalesPerCustomer',
                            inline=True,
                            className="grupo-radio",
                            labelClassName="radio-item-custom",
                            inputClassName="radio-input-custom"
                        ),
                        dcc.Loading(dcc.Graph(id='grafico-comportamento-sortimento-barras', config={'displayModeBar': False})),
                        html.Div([
                            html.P([
                                html.I(className="fas fa-lightbulb me-2"),
                                html.Strong("Análise: "),
                                html.Span(id='analise-comportamento-sortimento-barras')
                            ])
                        ], className="analise-text-box mt-3")
                    ])
                ], className="custom-card h-100"),
                md=6, className="graph-card-col"
            ),
        ]),

        html.Hr(className="my-5"),

        html.H2(["Fatores de Loja e Feriados", criar_icone_informacao("info-fatores-loja", "Explora como características intrínsecas da loja (distância do concorrente, tipo de sortimento, etc.) e feriados impactam o desempenho.")], className="page-title mt-5"),
        html.Hr(className="section-hr"),
        dbc.Row([
            criar_card_grafico('grafico-impacto-distancia-concorrencia', 'analise-impacto-distancia-concorrencia', 6),
            criar_card_grafico('grafico-impacto-promo2', 'analise-impacto-promo2', 6)
        ], className='mb-4 g-4'),
        dbc.Row([
            criar_card_grafico('grafico-impacto-sortimento', 'analise-impacto-sortimento', 6),
            criar_card_grafico('grafico-vendas-por-tipo-feriado', 'analise-vendas-por-tipo-feriado', 6)
        ], className='mb-4 g-4'),
    ])