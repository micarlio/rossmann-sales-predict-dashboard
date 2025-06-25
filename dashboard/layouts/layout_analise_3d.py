import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import numpy as np

from .componentes_compartilhados import criar_botoes_cabecalho, criar_card_filtros_3d, criar_card_grafico_3d # Refatorar nomes de módulos e funções

def criar_layout_analise_3d(dados): # Refatorar nome da função e parâmetro
    """Cria o layout da página de Análise 3D com filtros por gráfico."""
    nome_pagina = "analise-3d" # Refatorar nome da variável
    df_principal = dados['df_principal'] # Usar o novo nome do DataFrame principal

    # Opções para os filtros
    opcoes_tipo_loja = [{'label': f'Tipo {t.upper()}', 'value': t} for t in sorted(df_principal['StoreType'].unique())] # Refatorar nome da variável e DataFrame
    todos_tipos_loja = sorted(df_principal['StoreType'].unique()) # Refatorar nome da variável e DataFrame
    todas_lojas = sorted(df_principal['Store'].unique()) # Refatorar nome da variável e DataFrame
    opcoes_loja = [{'label': f'Loja {s}', 'value': s} for s in todas_lojas] # Refatorar nome da variável

    return dbc.Container(
        [
            dcc.Store(id='armazenamento-dados-base-3d'), # Refatorar ID

            # Cabeçalho da Página
            dbc.Row(
                [
                    dbc.Col(html.H2("Análise Multidimensional", className="page-title"), md=8),
                    dbc.Col(criar_botoes_cabecalho(nome_pagina), md=4, className="d-flex justify-content-end"), # Usar nova função e variável refatorada
                ],
                align="center",
                className="mb-4"
            ),
            # Filtros Globais (apenas data e feriados)
            criar_card_filtros_3d(df_principal), # Usar nova função e DataFrame

            # Card para o Gráfico de Correlação 3D (agora com filtros)
            dbc.Row([
                criar_card_grafico_3d( # Usar nova função
                    id_grafico='grafico-correlacao-3d', # Refatorar ID
                    titulo="Análise de Correlação 3D: Vendas vs. Clientes vs. Promoções",
                    id_analise='analise-correlacao-3d', # Refatorar ID
                    children=html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Filtrar por Tipo de Loja:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='filtro-tipo-loja-correlacao', # Refatorar ID
                                    options=opcoes_tipo_loja, # Usar nova variável refatorada
                                    value=todos_tipos_loja, # Usar nova variável refatorada
                                    multi=True,
                                    className="dash-dropdown"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Filtrar por Loja Específica:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='filtro-loja-especifica-correlacao', # Refatorar ID
                                    options=opcoes_loja, # Usar nova variável refatorada
                                    multi=True,
                                    className="dash-dropdown",
                                    placeholder="Deixe em branco para usar o filtro de tipo"
                                )
                            ], width=6)
                        ])
                    ], className="p-3")
                )
            ]),

            # Gráficos 3D empilhados com filtros individuais
            dbc.Row([
                criar_card_grafico_3d( # Usar nova função
                    id_grafico='grafico-superficie-3d', # Refatorar ID
                    titulo="Superfície de Sazonalidade (Vendas por Dia da Semana e Mês)",
                    id_analise='analise-superficie-3d', # Refatorar ID
                    children=html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Filtrar por Tipo de Loja:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='filtro-tipo-loja-superficie', # Refatorar ID
                                    options=opcoes_tipo_loja, # Usar nova variável refatorada
                                    value=todos_tipos_loja, # Usar nova variável refatorada
                                    multi=True,
                                    className="dash-dropdown"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Filtrar por Loja Específica:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='filtro-loja-especifica-superficie', # Refatorar ID
                                    options=opcoes_loja, # Usar nova variável refatorada
                                    multi=True,
                                    className="dash-dropdown",
                                    placeholder="Deixe em branco para usar o filtro de tipo"
                                )
                            ], width=6)
                        ])
                    ], className="p-3")
                ),
            ]),
            dbc.Row([
                criar_card_grafico_3d( # Usar nova função
                    id_grafico='grafico-dispersao-3d', # Refatorar ID
                    titulo="Dinâmica da Concorrência (Vendas vs. Clientes vs. Distância)",
                    id_analise='analise-dispersao-3d', # Refatorar ID
                    children=html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Filtrar por Tipo de Loja:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='filtro-tipo-loja-fatores', # Refatorar ID
                                    options=opcoes_tipo_loja, # Usar nova variável refatorada
                                    value=todos_tipos_loja, # Usar nova variável refatorada
                                    multi=True,
                                    className="dash-dropdown"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Filtrar por Loja Específica:", className="fw-bold"),
                                dcc.Dropdown(
                                    id='filtro-loja-especifica-fatores', # Refatorar ID
                                    options=opcoes_loja, # Usar nova variável refatorada
                                    multi=True,
                                    className="dash-dropdown",
                                    placeholder="Deixe em branco para usar o filtro de tipo"
                                )
                            ], width=6)
                        ])
                    ], className="p-3")
                ),
            ]),
            dbc.Row(
                [
                     criar_card_grafico_3d( # Usar nova função
                        id_grafico='grafico-dinamica-promocao-3d', # Refatorar ID
                        titulo="Dinâmica das Promoções (Clientes vs. Ticket Médio)",
                        id_analise='analise-dinamica-promocao-3d', # Refatorar ID
                        children=html.Div([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Filtrar por Tipo de Loja:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='filtro-tipo-loja-promocao', # Refatorar ID
                                        options=opcoes_tipo_loja, # Usar nova variável refatorada
                                        value=todos_tipos_loja, # Usar nova variável refatorada
                                        multi=True,
                                        className="dash-dropdown"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Filtrar por Loja Específica:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='filtro-loja-especifica-promocao', # Refatorar ID
                                        options=opcoes_loja, # Usar nova variável refatorada
                                        multi=True,
                                        className="dash-dropdown",
                                        placeholder="Deixe em branco para usar o filtro de tipo"
                                    )
                                ], width=6)
                            ])
                        ], className="p-3")
                    )
                ]
            )
        ],
        fluid=True,
        className="p-4 page-content"
    )