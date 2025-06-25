import dash_bootstrap_components as dbc
from dash import dcc, html

from .componentes_compartilhados import criar_botoes_cabecalho, criar_card_filtros_analise_lojas # Refatorar nomes de módulos e funções

def criar_layout_analise_lojas(dados): # Refatorar nome da função e parâmetro
    """Cria o layout da página de Análise Comparativa de Lojas."""
    df_principal = dados['df_principal'] # Usar o novo nome do DataFrame principal

    return dbc.Container([
        # --- Cabeçalho da Página ---
        dbc.Row([
            dbc.Col(html.H1("Análise Comparativa de Lojas", className="page-title"), md=10),
            dbc.Col(criar_botoes_cabecalho('analise-lojas'), md=2, className="d-flex justify-content-end align-items-center") # Usar nova função
        ], className="mb-4 align-items-center"),

        # --- Filtros Otimizados para esta página ---
        dbc.Row([
            dbc.Col(criar_card_filtros_analise_lojas(df_principal), width=12) # Usar nova função e DataFrame
        ], className="mb-4"),

        # --- Controles Específicos da Página ---
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Métrica para Ranking", html_for="seletor-metrica-ranking", className="fw-bold"), # Refatorar ID
                        dcc.Dropdown(
                            id='seletor-metrica-ranking', # Refatorar ID
                            options=[
                                {'label': 'Vendas Totais', 'value': 'Sales_sum'},
                                {'label': 'Vendas Médias por Dia', 'value': 'Sales_mean'},
                                {'label': 'Clientes Totais', 'value': 'Customers_sum'},
                                {'label': 'Clientes Médios por Dia', 'value': 'Customers_mean'},
                                {'label': 'Ticket Médio', 'value': 'SalesPerCustomer_mean'}
                            ],
                            value='Sales_mean',
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Ordem do Ranking", html_for="seletor-ordem-ranking", className="fw-bold"), # Refatorar ID
                        dbc.RadioItems(
                            id='seletor-ordem-ranking', # Refatorar ID
                            options=[
                                {'label': 'Melhores', 'value': 'desc'},
                                {'label': 'Piores', 'value': 'asc'},
                            ],
                            value='desc',
                            inline=True,
                            className="grupo-radio", # Refatorar classe CSS
                            labelClassName="radio-item-custom",
                            inputClassName="radio-input-custom"
                        )
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Número de Lojas a Exibir", html_for="slider-contagem-ranking", className="fw-bold"), # Refatorar ID
                        dcc.Slider(id='slider-contagem-ranking', min=5, max=50, step=5, value=10, # Refatorar ID
                                   marks={i: str(i) for i in range(5, 51, 5)})
                    ], md=4)
                ])
            ]),
            className="mb-4"
        ),

        # --- Área de Resultados: Ranking e Detalhes ---
        dbc.Row([
            # Coluna para a Tabela de Ranking
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Ranking de Lojas", className="card-title fw-bold m-0")),
                        dbc.CardBody([
                            html.P("Selecione uma loja na tabela para ver seus detalhes ao lado. Se clicar em duas, os dados das duas serão comparados.", className="card-subtitle mb-3 text-muted"),
                            dcc.Loading(type="circle", children=html.Div(id="tabela-ranking-lojas")) # Refatorar ID
                        ])
                    ], className="custom-card"
                ),
                md=5,
                className="mb-4"
            ),

            # Coluna para os Detalhes da Loja Selecionada
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H5("Detalhes da Loja Selecionada", className="card-title fw-bold m-0")),
                        dbc.CardBody(
                            dcc.Loading(type="circle", children=html.Div(id="conteudo-detalhe-loja")), # Refatorar ID
                            # O estilo de rolagem e altura máxima é aplicado diretamente no corpo do card.
                            # A altura é calculada para descontar a altura do cabeçalho.
                            style={'max-height': 'calc(85vh - 58px)', 'overflow-y': 'auto'},
                            className="p-3" # Mantém o padding para o conteúdo.
                        )
                    ],
                    className="custom-card",
                ),
                md=7,
                className="mb-4"
            )
        ], align="start"),

        # Stores para gerenciamento de estado da página
        dcc.Store(id='armazenamento-dados-ranking'), # Refatorar ID
        dcc.Store(id='armazenamento-id-loja-selecionada', data=[]) # Refatorar ID

    ], fluid=True, className="p-4 page-content")