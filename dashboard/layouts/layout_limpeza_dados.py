import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import pandas as pd

from .componentes_compartilhados import criar_botoes_cabecalho # Refatorar nome do módulo e da função
from ..config import VERMELHO_ROSSMANN, FUNDO_CINZA_CLARO, AZUL_ESCURO, CINZA_NEUTRO
from ..data_loader import CAMINHO_ARQUIVO_LOJAS_BRUTO, reduzir_uso_memoria  # para análise de valores ausentes e memória

def criar_layout_limpeza_dados(dados): # Refatorar nome da função e parâmetro
    nome_pagina = "limpeza" # Refatorar nome da variável
    media_vendas_antes = dados["media_vendas_antes"] # Refatorar nome da variável
    media_vendas_depois = dados["media_vendas_depois"] # Refatorar nome da variável
    contagem_vendas_antes = dados["contagem_vendas_antes"] # Refatorar nome da variável
    contagem_vendas_depois = dados["contagem_vendas_depois"] # Refatorar nome da variável

    df_limpeza_media = pd.DataFrame({'Estado do Dataset': ['Antes da Limpeza', 'Após Limpeza (Open==1)'], 'Vendas Médias (€)': [media_vendas_antes, media_vendas_depois]}) # Refatorar nome da variável
    df_limpeza_contagem = pd.DataFrame({'Estado do Dataset': ['Antes da Limpeza', 'Após Limpeza (Open==1)'], 'Número de Registros': [contagem_vendas_antes, contagem_vendas_depois]}) # Refatorar nome da variável

    fig_limpeza_media = px.bar(df_limpeza_media, x='Estado do Dataset', y='Vendas Médias (€)', title="Impacto da Limpeza: Média de Vendas", text_auto='.2f', color='Estado do Dataset', color_discrete_map={'Antes da Limpeza': CINZA_NEUTRO, 'Após Limpeza (Open==1)': VERMELHO_ROSSMANN}) # Refatorar nome da variável e constantes
    fig_limpeza_contagem = px.bar(df_limpeza_contagem, x='Estado do Dataset', y='Número de Registros', title="Impacto da Limpeza: Contagem de Registros", text_auto=',', color='Estado do Dataset', color_discrete_map={'Antes da Limpeza': CINZA_NEUTRO, 'Após Limpeza (Open==1)': VERMELHO_ROSSMANN}) # Refatorar nome da variável e constantes

    registros_removidos = contagem_vendas_antes - contagem_vendas_depois # Refatorar nome da variável
    registros_originais = contagem_vendas_antes # Refatorar nome da variável

    # --- Seção de Tratamento de Valores Ausentes ---
    # Carrega dados brutos de lojas para análise de valores ausentes
    df_lojas_bruto = pd.read_parquet(CAMINHO_ARQUIVO_LOJAS_BRUTO)
    colunas_nulos = ['CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
    percent_missing = df_lojas_bruto[colunas_nulos].isnull().mean() * 100
    estrategias = [
        {'coluna': 'CompetitionDistance', 'missing_pct': percent_missing['CompetitionDistance'], 'estrategia': 'Preenchido com a média', 'justificativa': 'Poucos valores ausentes; a média não distorce a distribuição significativamente.'},
        {'coluna': 'CompetitionOpenSinceMonth', 'missing_pct': percent_missing['CompetitionOpenSinceMonth'], 'estrategia': 'Preenchido com 0', 'justificativa': 'Assume-se que a ausência de data indica que não há competidor próximo.'},
        {'coluna': 'CompetitionOpenSinceYear', 'missing_pct': percent_missing['CompetitionOpenSinceYear'], 'estrategia': 'Preenchido com 0', 'justificativa': 'Assume-se que a ausência de data indica que não há competidor próximo.'},
        {'coluna': 'Promo2SinceWeek', 'missing_pct': percent_missing['Promo2SinceWeek'], 'estrategia': 'Preenchido com 0', 'justificativa': 'A ausência de data está diretamente ligada a Promo2 = 0.'},
        {'coluna': 'Promo2SinceYear', 'missing_pct': percent_missing['Promo2SinceYear'], 'estrategia': 'Preenchido com 0', 'justificativa': 'A ausência de data está diretamente ligada a Promo2 = 0.'},
        {'coluna': 'PromoInterval', 'missing_pct': percent_missing['PromoInterval'], 'estrategia': 'Preenchido com "Nenhum"', 'justificativa': 'Categoria explícita para lojas que não participam da Promo2.'}
    ]
    tabela_estrategias = html.Table(
        children=[
            html.Thead(html.Tr([html.Th("Coluna"), html.Th("% Valores Ausentes"), html.Th("Estratégia"), html.Th("Justificativa")])),
            html.Tbody([
                html.Tr([html.Td(e['coluna']), html.Td(f"{e['missing_pct']:.1f}%"), html.Td(e['estrategia']), html.Td(e['justificativa'])])
                for e in estrategias
            ])
        ],
        className="table table-sm table-striped"
    )

    # --- Seção de Matriz de Valores Ausentes ---
    matrix_nulos = df_lojas_bruto[colunas_nulos].isnull()
    fig_nulos = px.imshow(
        matrix_nulos.T,
        color_continuous_scale=[FUNDO_CINZA_CLARO, VERMELHO_ROSSMANN],
        labels={'x': 'Índice', 'y': 'Coluna', 'color': 'Ausente'},
        aspect='auto',
        title="Matriz de Valores Ausentes (store.parquet)"
    ).update_xaxes(showticklabels=False)

    # --- Seção de Impacto da Limpeza ---
    mem_bruto = df_lojas_bruto.memory_usage(deep=True).sum() / 1024**2
    df_lojas_otimizado = reduzir_uso_memoria(df_lojas_bruto.copy(), nome_df="df_lojas_bruto_dashboard")
    mem_otimizado = df_lojas_otimizado.memory_usage(deep=True).sum() / 1024**2
    red_pct = (1 - mem_otimizado / mem_bruto) * 100
    raw_dist = df_lojas_bruto['CompetitionDistance'].dropna()
    tratado_dist = dados['df_lojas_tratado']['CompetitionDistance']
    df_hist = pd.DataFrame({
        'CompetitionDistance': pd.concat([raw_dist, tratado_dist], ignore_index=True),
        'Status': ['Bruto'] * len(raw_dist) + ['Tratado'] * len(tratado_dist)
    })
    fig_hist_dist = px.histogram(
        df_hist, x='CompetitionDistance', color='Status',
        barmode='overlay', nbins=50, opacity=0.7,
        title='Distribuição de CompetitionDistance: Bruto vs Tratado'
    ).update_layout(xaxis_title='Distance (m)', yaxis_title='Contagem')

    # --- Seção de Análise de Outliers ---
    fig_box_sales = px.box(dados['df_principal'], y='Sales', title="Boxplot de Vendas (Sales)", color_discrete_sequence=[AZUL_ESCURO])
    fig_box_customers = px.box(dados['df_principal'], y='Customers', title="Boxplot de Clientes (Customers)", color_discrete_sequence=[AZUL_ESCURO])

    return html.Div([
        html.Div([
            html.H1("Processo de Limpeza e Pré-processamento", className="page-title"),
            criar_botoes_cabecalho(nome_pagina) # Usar nova função e variável refatorada
        ], className="d-flex justify-content-between align-items-center mb-4"),
        # Controles de Limpeza e Amostragem
        html.Div([
            html.H3("Controles de Limpeza e Amostragem", style={'borderLeft': f'5px solid {VERMELHO_ROSSMANN}', 'paddingLeft': '10px'}),
            html.Div([
                dcc.RadioItems(
                    id='seletor-modo-dados',
                    options=[
                        {'label': 'Dataset Completo (Limpo)', 'value': 'completo'},
                        {'label': 'Amostras por Loja', 'value': 'amostras'},
                    ],
                    value='amostras',
                    labelStyle={'display': 'inline-block', 'margin-right': '20px'}
                ),
                html.Div(id='container-input-amostras', children=[
                    dcc.Input(
                        id='input-numero-amostras',
                        type='number',
                        placeholder='Nº de amostras por loja',
                        value=50,
                        min=1,
                        step=1,
                    ),
                    html.Span("Atualização automática", style={'marginLeft': '10px', 'color': AZUL_ESCURO, 'fontStyle': 'italic'})
                ], style={'display': 'none', 'marginTop': '10px'})
            ], style={'padding': '10px', 'border': f'1px solid {FUNDO_CINZA_CLARO}', 'borderRadius': '5px', 'marginBottom': '20px'})
        ], className='mb-4'),
        dbc.Card([
            dbc.CardBody([
                dcc.Markdown(f"""
                    A primeira etapa do projeto envolveu a limpeza e a preparação dos dados brutos. As ações mais significativas foram:
                    * **União dos Datasets:** Foi realizada a junção dos dados de vendas (`train.parquet`) com as informações detalhadas sobre cada loja (`store.parquet`) utilizando o `Store ID` como chave comum.
                    * **Tratamento de Lojas Fechadas:** A decisão mais impactante foi a remoção dos {registros_removidos:,.0f} registros diários onde as lojas estavam fechadas (`Open == 0`), de um total de {registros_originais:,.0f} registros originais. Isso garante que a análise se concentre apenas nos dias de operação efetiva.
                    * **Tratamento de Dados Faltantes em `store_df`:** Preenchemos valores ausentes (`NaN`) em colunas como `CompetitionDistance` (com a média da coluna) e em campos relacionados a `Promo2` e `CompetitionOpenSince` (com 0, indicando "não aplicável" ou "desconhecido" para facilitar a modelagem futura).
                """, className="mb-4"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-impacto-media'), md=6),
                    dbc.Col(dcc.Graph(id='grafico-impacto-contagem'), md=6)
                ], className="g-4")
            ])
        ], className="custom-card"),
        # Seção de Valores Ausentes
        dbc.Card([
            dbc.CardBody([
                html.H3("Tratamento de Valores Ausentes", style={'borderLeft': f'5px solid {VERMELHO_ROSSMANN}', 'paddingLeft': '10px'}),
                tabela_estrategias
            ])
        ], className="custom-card mt-4"),
        # Seção de Matriz de Valores Ausentes
        dbc.Card([
            dbc.CardBody([
                html.H3("Matriz de Valores Ausentes", style={'borderLeft': f'5px solid {VERMELHO_ROSSMANN}', 'paddingLeft': '10px'}),
                dcc.Graph(figure=fig_nulos)
            ])
        ], className="custom-card mt-4"),
        # Seção de Outliers
        dbc.Card([
            dbc.CardBody([
                html.H3("Análise de Outliers", style={'borderLeft': f'5px solid {VERMELHO_ROSSMANN}', 'paddingLeft': '10px'}),
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_box_sales), md=6),
                    dbc.Col(dcc.Graph(figure=fig_box_customers), md=6)
                ], className="g-4"),
                dcc.Markdown("""
                    **Decisão de Tratamento:** Mantivemos os outliers nas colunas *Sales* e *Customers* 
                    pois eles representam eventos reais de vendas pontuais (como feriados e promoções). 
                    A remoção desses pontos poderia reduzir a capacidade do modelo de capturar picos de demanda.
                """)
            ])
        ], className="custom-card mt-4")
    ])