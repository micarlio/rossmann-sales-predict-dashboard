import dash_bootstrap_components as dbc
from dash import dcc, html

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
            # Controles de Previsão de Vendas
            html.Div([
                html.H3('Controles de Previsão', className='section-subtitle'),
                dbc.Row([
                    dbc.Col([
                        html.Label('Métrica de Previsão'),
                        dcc.RadioItems(
                            id='radio-metrica-previsao',
                            options=[
                                {'label': 'Vendas', 'value': 'Sales'},
                                {'label': 'Clientes', 'value': 'Customers'}
                            ],
                            value='Sales',
                            inline=True,
                            labelStyle={'margin-right': '10px'}
                        )
                    ], width=12)
                ], className='g-4 mb-4'),
                dbc.Row([
                    dbc.Col([
                        html.Label('Horizonte de Previsão (períodos)'),
                        dcc.Slider(id='slider-horizonte-previsao', min=1, max=30, step=1, value=7,
                                   marks={i: str(i) for i in range(1, 31, 7)})
                    ], md=4),
                    dbc.Col([
                        html.Label('Granularidade'),
                        dcc.Dropdown(
                            id='dropdown-granularidade-previsao',
                            options=[
                                {'label': 'Diária', 'value': 'diaria'},
                                {'label': 'Semanal', 'value': 'semanal'},
                                {'label': 'Mensal', 'value': 'mensal'}
                            ],
                            value='diaria',
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label('Seleção de Loja(s)'),
                        dcc.Dropdown(
                            id='dropdown-lojas-previsao',
                            options=[], multi=True,
                            placeholder='Selecione loja(s)'
                        )
                    ], md=4)
                ], className='g-4 mb-4'),
                dbc.Row([
                    dbc.Col([
                        html.Label('Intervalo de Dados Históricos'),
                        dcc.DatePickerRange(
                            id='date-picker-historico-previsao',
                            start_date_placeholder_text='Data Início',
                            end_date_placeholder_text='Data Fim'
                        )
                    ], md=6),
                    dbc.Col([
                        html.Label('Modelo de Previsão'),
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
                            clearable=False
                        )
                    ], md=6)
                ], className='g-4')
            ], className='mb-4'),
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
                    # Botão de download de CSV
                    html.Div([
                        dbc.Button('Baixar CSV', id='btn-download-csv', color='primary')
                    ], className='mt-3 mb-4'),
                    html.Div(id='cards-metricas', className='mt-4'),
                    html.Div(id='tabela-previsao', className='mt-4')
                ])
            ], className='custom-card')
        ],
        fluid=True,
        className="p-4 page-content"
    )