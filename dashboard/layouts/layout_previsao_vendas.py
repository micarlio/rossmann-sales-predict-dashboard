import dash_bootstrap_components as dbc
from dash import html

from .componentes_compartilhados import criar_botoes_cabecalho # Refatorar nome do módulo e da função

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
            dbc.Alert(
                [
                    html.I(className="fas fa-tools me-2"),
                    html.Strong("Página em Construção: "),
                    "Esta seção abrigará os resultados dos modelos de Machine Learning para prever as vendas futuras."
                ],
                color="info"
            )
        ],
        fluid=True,
        className="p-4 page-content"
    )