import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import numpy as np
import pandas as pd

from .componentes_compartilhados import criar_botoes_cabecalho # Refatorar nome do módulo e da função
from ..utils import criar_figura_vazia # Refatorar nome do módulo e da função
from ..config import (
    VERMELHO_ROSSMANN, CINZA_NEUTRO, COLUNAS_NUMERICAS_VENDAS, # Importar novas constantes
    COLUNAS_NUMERICAS_LOJAS_PARA_PLOTAR # Importar novas constantes
)

def criar_layout_analise_preliminar(dados): # Refatorar nome da função e parâmetro
    nome_pagina = "analise-preliminar" # Refatorar nome da variável
    df_principal = dados["df_principal"] # Usar o novo nome do DataFrame principal

    # --- Lógica para o Gráfico de Barras de Correlação (Melhorado) ---
    # Removida a criação estática da matriz de correlação, agora gerada dinamicamente pelo callback

    opcoes_colunas_vendas = [{'label': col, 'value': col} for col in COLUNAS_NUMERICAS_VENDAS if col != 'Open'] # Refatorar nome da variable e constante
    opcoes_colunas_lojas = [{'label': col, 'value': col} for col in COLUNAS_NUMERICAS_LOJAS_PARA_PLOTAR] # Refatorar nome da variable e constante

    return html.Div([
        html.Div([
            html.H1("Análise Preliminar e Estatística", className="page-title"),
            criar_botoes_cabecalho(nome_pagina) # Usar nova função e variável refatorada
        ], className="d-flex justify-content-between align-items-center mb-4"),
        dbc.Card([
            dbc.CardBody([
                dcc.Markdown("""
                    Com os dados limpos e unificados, exploramos as relações entre as variáveis numéricas. A matriz de correlação permite quantificar a força e a direção dessas relações.
                    * **`Sales` e `Customers`:** A correlação mais evidente é entre Vendas e Número de Clientes (0.82), indicando que o fluxo de clientes é o principal motor das vendas.
                    * **`Promo` e `Sales`:** Observa-se um impacto positivo das promoções diárias nas vendas (0.37), confirmando sua eficácia.
                    * **`DayOfWeek` e `Promo2`:** Estas variáveis mostram uma correlação negativa fraca com as vendas, sugerindo que dias de fim de semana (onde algumas lojas fecham) ou a presença de promoções contínuas (`Promo2`) podem estar associadas a padrões de vendas ligeiramente diferentes.
                """, className="mb-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-matriz-correlacao'), width="auto") # Matriz gerada dinamicamente pelo callback
                ], className="mb-5 justify-content-center"),
                # Texto explicativo mantendo referência à matriz de correlação (gráfico removido)
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H4("Correlação com Vendas", className="text-center mb-3"),
                            dcc.Markdown("""
                                A matriz de correlação acima mostra a relação entre cada variável e as **vendas (Sales)**.
                                * **Correlação Positiva**: quando a variável aumenta, as vendas tendem a aumentar.
                                * **Correlação Negativa**: quando a variável aumenta, as vendas tendem a diminuir.
                                * **Força da Correlação**: valores próximos de ±1 indicam uma relação mais forte.
                            """, className="mb-0")
                        ], className="border-start border-4 border-primary p-4")
                    ], md=12)
                ]),
                html.Hr(className="my-5"),
                html.Div([
                    html.H4("Visualização de Correlação entre Variáveis", className="fw-bold text-center"),
                    html.P("Clique em uma célula na Matriz de Correlação para visualizar o gráfico de dispersão correspondente.", className="text-muted text-center mb-4"),
                    dcc.Graph(id='grafico-dispersao-correlacao', figure=criar_figura_vazia("Clique em uma célula da matriz")) # Refatorar ID e função
                ])
            ])
        ], className="custom-card"),
        html.H2("Análise de Distribuição das Variáveis", className="section-subtitle"),
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5("Dataset de Vendas", className="fw-bold"),
                        html.P("Distribuição antes e depois da limpeza de dados.", className="text-muted"),
                        dcc.Dropdown(id='dropdown-histograma-vendas', options=opcoes_colunas_vendas, value='Sales', multi=False, className="dash-dropdown mb-3") # Refatorar ID e variável
                    ]),
                    dbc.Col([
                        html.H5("Dataset de Lojas", className="fw-bold"),
                        html.P("Distribuição antes e depois do tratamento de dados.", className="text-muted"),
                        dcc.Dropdown(id='dropdown-histograma-lojas', options=opcoes_colunas_lojas, value='CompetitionDistance', multi=False, className="dash-dropdown mb-3") # Refatorar ID e variável
                    ])
                ]),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='histograma-vendas-comparativo'), md=6), # Refatorar ID
                    dbc.Col(dcc.Graph(id='histograma-lojas'), md=6) # Refatorar ID
                ], className="g-4"),
                html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-estatisticas-vendas'), md=6), # Refatorar ID
                    dbc.Col(dcc.Graph(id='grafico-estatisticas-lojas'), md=6) # Refatorar ID
                ], className="g-4")
            ])
        ], className="custom-card")
    ])