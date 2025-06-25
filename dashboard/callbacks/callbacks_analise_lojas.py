# dashboard/callbacks/callbacks_analise_lojas.py
import dash
from dash import Input, Output, State, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from io import StringIO
import json

from ..utils import criar_figura_vazia, filtrar_dataframe # Importar as funções utilitárias refatoradas
from ..data_loader import get_principal_dataset, N_AMOSTRAS_PADRAO
from ..config import VERMELHO_ROSSMANN, AZUL_ESCURO, CINZA_NEUTRO, MAPEAMENTO_DIAS_SEMANA, ORDEM_DIAS_SEMANA # Importar as novas constantes
from ..config import AZUL_DESTAQUE, PALETA_CORES_GRAFICO # Importar as novas constantes

# Função auxiliar para deserializar o DataFrame do JSON
def deserializar_df(store_data):
    """
    Obtém o DataFrame principal a partir do dcc.Store.
    Se store_data for dict, utiliza get_principal_dataset; caso contrário, desserializa o JSON.
    """
    # Uso de cache no servidor quando store_data é dict
    if isinstance(store_data, dict) and 'modo' in store_data:
        modo = store_data.get('modo', 'completo')
        n_amostras = store_data.get('n_amostras', N_AMOSTRAS_PADRAO)
        use_samples = (modo == 'amostras')
        df = get_principal_dataset(use_samples=use_samples, n_amostras=n_amostras)
        # Garantir que a coluna 'Date' seja datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    # Caso contrário, assume que é JSON string
    if not store_data:
        return None
    df = pd.read_json(StringIO(store_data), orient='split')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def registrar_callbacks_analise_lojas(aplicativo):
    """Registra os callbacks para a página de análise de lojas."""
    
    @aplicativo.callback(
        Output('armazenamento-dados-ranking', 'data'),
        [Input('url', 'pathname'),
         Input('armazenamento-df-principal', 'data'),
         Input('filtro-data', 'start_date'),
         Input('filtro-data', 'end_date'),
         Input('filtro-tipo-loja', 'value'),
         Input('filtro-loja-especifica', 'value'),
         Input('filtro-feriado-estadual', 'value'),
         Input('filtro-feriado-escolar', 'value'),
         Input('seletor-metrica-ranking', 'value'),
         Input('seletor-ordem-ranking', 'value')]  # Novo input para centralizar a lógica
    )
    def atualizar_dados_ranking(caminho_pagina, df_principal_json, data_inicio, data_fim, tipos_loja, lojas_especificas,
                            feriado_estadual, feriado_escolar, metrica, ordem):  # Novo parâmetro 'ordem'
        if caminho_pagina != '/analise-lojas':
            return dash.no_update

        # Usar a função auxiliar para deserializar o DataFrame
        df_principal = deserializar_df(df_principal_json)
        if df_principal is None:
            return dash.no_update
        
        # Ranking é calculado para todas as lojas que obedecem aos filtros globais
        # Passamos None para o filtro de lojas específicas para que o ranking seja calculado
        # com base em TODOS os dados que passam pelos filtros globais, ignorando a seleção específica de lojas
        df_filtrado = filtrar_dataframe(df_principal, data_inicio, data_fim, tipos_loja, None, feriado_estadual, feriado_escolar)

        if df_filtrado.empty:
            return pd.DataFrame().to_json(date_format='iso', orient='split')

        mapeamento_metrica = {
            'Sales_sum': ('Sales', 'sum'), 'Sales_mean': ('Sales', 'mean'),
            'Customers_sum': ('Customers', 'sum'), 'Customers_mean': ('Customers', 'mean'),
            'SalesPerCustomer_mean': ('SalesPerCustomer', 'mean')
        }
        coluna_metrica, funcao_agg = mapeamento_metrica[metrica]

        df_ranking_loja = df_filtrado.groupby('Store').agg({
            coluna_metrica: funcao_agg,
            'StoreType': 'first',
            'Assortment': 'first'
        }).rename(columns={coluna_metrica: 'Métrica'}).reset_index()
        
        # --- NOVA LÓGICA CENTRALIZADA ---
        # Ordenação e atribuição do ranking já acontecem aqui
        if not df_ranking_loja.empty:
            eh_ascendente = ordem == 'asc'
            df_ranking_loja = df_ranking_loja.sort_values(by='Métrica', ascending=eh_ascendente)
            df_ranking_loja['Ranking'] = range(1, len(df_ranking_loja) + 1)
            
            # Salvar uma cópia do valor numérico para cálculo de barra de progresso
            df_ranking_loja['MetricValue'] = df_ranking_loja['Métrica'].copy()
        # ---------------------------------

        return df_ranking_loja.to_json(date_format='iso', orient='split')

    @aplicativo.callback(
        Output('tabela-ranking-lojas', 'children'),
        [Input('armazenamento-dados-ranking', 'data'),
         Input('slider-contagem-ranking', 'value'),
         Input('armazenamento-id-loja-selecionada', 'data'),
         Input('seletor-metrica-ranking', 'value'),
         Input('filtro-loja-especifica', 'value')],
        [State('seletor-ordem-ranking', 'value')]  # Adicionado State para acessar a ordem sem recalcular o callback
    )
    def atualizar_tabela_ranking_lojas(dados_json, contagem, ids_lojas_selecionadas, metrica, lojas_especificas, ordem):
        if not dados_json:
            return dash.no_update

        # O DataFrame já vem ordenado e com as colunas 'Ranking' e 'MetricValue'
        ranking_lojas = pd.read_json(StringIO(dados_json), orient='split')
        if ranking_lojas.empty:
            return dbc.Alert("Nenhuma loja encontrada para os filtros selecionados.", color="warning")

        nomes_cabecalho = {
            'Sales_sum': 'Vendas Totais', 'Sales_mean': 'Vendas Médias',
            'Customers_sum': 'Clientes Totais', 'Customers_mean': 'Clientes Médios',
            'SalesPerCustomer_mean': 'Ticket Médio'
        }

        # Decide qual subconjunto de lojas exibir
        df_a_exibir = ranking_lojas
        if lojas_especificas:
            # Filtra pelas lojas selecionadas, mas mantém a coluna 'Ranking' original
            df_a_exibir = ranking_lojas[ranking_lojas['Store'].isin(lojas_especificas)]
        else:
            # Mostra o Top N
            df_a_exibir = ranking_lojas.head(contagem)

        # Formatação da métrica para exibição
        if "Sales" in metrica or "SalesPerCustomer" in metrica:
            df_a_exibir['Métrica'] = df_a_exibir['Métrica'].apply(lambda x: f"€ {x:,.2f}")
        else:
            df_a_exibir['Métrica'] = df_a_exibir['Métrica'].apply(lambda x: f"{x:,.0f}")

        # Máximo para normalizar a barra de progresso (do DataFrame completo)
        valor_max_metrica = ranking_lojas['MetricValue'].max() if not ranking_lojas.empty else 0
        medalhas = {1: "🥇", 2: "🥈", 3: "🥉"}

        cabecalho_tabela = [html.Thead(html.Tr([
            html.Th("#"), html.Th("Loja"), html.Th("Tipo"), html.Th("Sortimento"), html.Th(nomes_cabecalho.get(metrica, "Métrica"))
        ], className="table-dark"))]

        corpo_tabela = [html.Tbody([
            html.Tr([
                # Coluna posição com medalhas
                html.Td(f"{medalhas.get(rank['Ranking'], '')} {rank['Ranking']}"),
                html.Td(rank['Store']),
                html.Td(rank['StoreType'].upper()), html.Td(rank['Assortment'].upper()),
                # Coluna métrica + barra de progresso
                html.Td([
                    html.Span(rank['Métrica'], className="d-block"),
                    dbc.Progress(value= (rank['MetricValue']/valor_max_metrica*100) if valor_max_metrica else 0,
                                 color='success' if ordem == 'desc' else 'danger',  # Corrigido para usar o parâmetro ordem
                                 style={'height':'6px'})
                ])
            ], id={'type': 'linha-ranking', 'index': rank['Store']}, n_clicks=0,
               className="ranking-row ranking-row-selected" if rank['Store'] in ids_lojas_selecionadas else "ranking-row")
            for _, rank in df_a_exibir.iterrows()
        ])]

        return dbc.Table(cabecalho_tabela + corpo_tabela, bordered=True, striped=True, hover=True, responsive=True, className="mt-3")

    # ==============================================================================
    # HELPER FUNCTIONS PARA A PÁGINA DE ANÁLISE DE LOJAS
    # ==============================================================================

    def gerar_visualizacao_loja_unica(id_loja, dados_json, ordem_ranking, data_inicio, data_fim, feriado_estadual, feriado_escolar, metrica_ranking, df_principal_json):
        """Gera o layout completo de detalhes para uma única loja."""
        
        # Verificar se temos os dados do DataFrame principal
        if not df_principal_json:
            return dbc.Alert("Erro interno: DataFrame principal não encontrado.", color="danger")
            
        # Usar a função auxiliar para deserializar o DataFrame
        df_principal = deserializar_df(df_principal_json)
        if df_principal is None:
            return dbc.Alert("Erro interno: Falha ao processar os dados.", color="danger")

        # Para obter o StoreType para o filtro, precisamos do df original ou do df principal filtrado por data
        tipo_loja_para_filtro = df_principal[df_principal['Store'] == id_loja]['StoreType'].iloc[0]

        df_filtrado_loja = filtrar_dataframe(df_principal, data_inicio, data_fim, [tipo_loja_para_filtro], [id_loja], feriado_estadual, feriado_escolar)
        if df_filtrado_loja.empty:
            return dbc.Alert(f"Não foram encontrados dados para a loja {id_loja} com os filtros atuais.", color="warning")

        # CORREÇÃO CRÍTICA: Ordenar o DataFrame por data antes de plotar a série temporal
        df_filtrado_loja = df_filtrado_loja.sort_values(by='Date')

        # Geração de cards e gráficos
        media_vendas = df_filtrado_loja['Sales'].mean()
        media_clientes = df_filtrado_loja['Customers'].mean()
        media_ticket = df_filtrado_loja['SalesPerCustomer'].mean() if df_filtrado_loja['SalesPerCustomer'].count() > 0 else 0
        
        # Layout base para os gráficos
        layout_base = {
            'height': 280, 
            'margin': dict(t=50, b=40, l=40, r=20),
            'plot_bgcolor': '#ffffff',
            'paper_bgcolor': '#ffffff'
        }
        
        # Resto do código para gerar os cards
        card_kpi = dbc.Card(dbc.CardBody([
            html.H4(f"Detalhes da Loja: {id_loja}", className="card-title"),
            dbc.Row([
                dbc.Col(html.Div([html.P("Vendas Médias/Dia"), html.H5(f"€ {media_vendas:,.2f}")])),
                dbc.Col(html.Div([html.P("Clientes Médios/Dia"), html.H5(f"{media_clientes:,.0f}")])),
                dbc.Col(html.Div([html.P("Ticket Médio"), html.H5(f"€ {media_ticket:,.2f}")])),
            ], className="text-center")
        ], className="p-3"), className="mb-3")

        # Código para gerar o card de detalhes estáticos...
        info_loja = df_principal[df_principal['Store'] == id_loja].iloc[0]
        str_ranking = "N/A"
        if dados_json:
            df_ranking = pd.read_json(StringIO(dados_json), orient='split')
            if not df_ranking.empty:
                info_ranking_loja = df_ranking[df_ranking['Store'] == id_loja]
                if not info_ranking_loja.empty:
                    rank = info_ranking_loja['Ranking'].iloc[0]
                    total = len(df_ranking)
                    str_ranking = f"{rank}º de {total}"
                    
        str_distancia = f"{info_loja['CompetitionDistance']:,.0f} m" if pd.notna(info_loja['CompetitionDistance']) else "N/A"
        status_promo2 = "Sim" if info_loja['Promo2'] == 1 else "Não"
        if info_loja['Promo2'] == 1 and info_loja.get('PromoInterval'):
            status_promo2 += f" ({info_loja['PromoInterval']})"
        mapeamento_sortimento = {'a': 'Básico', 'b': 'Extra', 'c': 'Estendido'}
        str_sortimento = mapeamento_sortimento.get(info_loja['Assortment'], info_loja['Assortment'])
        ano_abertura_concorrente = info_loja.get('CompetitionOpenSinceYear')
        mes_abertura_concorrente = info_loja.get('CompetitionOpenSinceMonth')
        str_abertura_concorrente = f"{int(mes_abertura_concorrente)}/{int(ano_abertura_concorrente)}" if pd.notna(ano_abertura_concorrente) and ano_abertura_concorrente > 0 else "N/A"
        
        def criar_item_info(titulo, valor, icone):
            return dbc.Col(html.Div([
                html.P([html.I(className=f"fas {icone} me-2"), titulo], className="text-muted"),
                html.H5(valor, className="fw-bold")
            ]), md=4, className="text-center mb-3")
            
        card_detalhes_estaticos = dbc.Card(dbc.CardBody([
            html.H5("Características e Ranking da Loja", className="card-title text-center mb-4"),
            dbc.Row([
                criar_item_info("Ranking", str_ranking, "fa-trophy"),
                criar_item_info("Tipo de Loja", info_loja['StoreType'].upper(), "fa-tag"),
                criar_item_info("Sortimento", str_sortimento, "fa-boxes-stacked"),
                criar_item_info("Dist. Concorrente", str_distancia, "fa-road"),
                criar_item_info("Abertura Concorrente", str_abertura_concorrente, "fa-calendar-alt"),
                criar_item_info("Promoção Contínua", status_promo2, "fa-bullhorn"),
            ], className="justify-content-center")
        ], className="p-1"), className="mb-1")

        mapeamento_metrica = {'Sales_sum': 'Sales', 'Sales_mean': 'Sales', 'Customers_sum': 'Customers', 'Customers_mean': 'Customers', 'SalesPerCustomer_mean': 'SalesPerCustomer'}
        mapeamento_rotulo = {'Sales': 'Vendas', 'Customers': 'Clientes', 'SalesPerCustomer': 'Ticket Médio'}
        mapeamento_titulo_eixo_y = {'Sales': 'Vendas Diárias (€)', 'Customers': 'Nº de Clientes Diário', 'SalesPerCustomer': 'Ticket Médio Diário (€)'}
        coluna_metrica = mapeamento_metrica.get(metrica_ranking, 'Sales')
        rotulo_metrica = mapeamento_rotulo.get(coluna_metrica, 'Vendas')
        titulo_eixo_y = mapeamento_titulo_eixo_y.get(coluna_metrica, 'Vendas Diárias (€)')

        # Gráficos com layout base
        fig_ts = px.line(df_filtrado_loja, x='Date', y=coluna_metrica, title=f'Série Temporal de {rotulo_metrica} - Loja {id_loja}')
        fig_ts.update_traces(line=dict(color=VERMELHO_ROSSMANN))
        fig_ts.update_layout(**layout_base, yaxis_title=titulo_eixo_y)
        
        fig_promo = px.box(df_filtrado_loja, x='Promo', y=coluna_metrica, color='Promo', 
                          title=f'Impacto da Promoção em {rotulo_metrica}', 
                          labels={coluna_metrica: titulo_eixo_y, 'Promo': 'Promoção'}, 
                          color_discrete_map={0: CINZA_NEUTRO, 1: VERMELHO_ROSSMANN})
        fig_promo.update_layout(**layout_base, showlegend=False)
        fig_promo.update_xaxes(tickvals=[0, 1], ticktext=['Sem Promoção', 'Com Promoção'])
        
        df_dia_semana = df_filtrado_loja.groupby('DayOfWeek')[coluna_metrica].mean().reset_index()
        df_dia_semana['DayName'] = df_dia_semana['DayOfWeek'].map(MAPEAMENTO_DIAS_SEMANA)
        fig_dia_semana = px.bar(df_dia_semana, x='DayName', y=coluna_metrica, 
                               title=f'Média de {rotulo_metrica} por Dia da Semana', 
                               labels={coluna_metrica: f"Média de {rotulo_metrica}", 'DayName': 'Dia da Semana'}, 
                               color_discrete_sequence=[VERMELHO_ROSSMANN])
        fig_dia_semana.update_layout(**layout_base, xaxis={'categoryorder':'array', 'categoryarray': ORDEM_DIAS_SEMANA})
        
        fig_dist = px.histogram(df_filtrado_loja, x=coluna_metrica, nbins=50, 
                               title=f'Distribuição de {rotulo_metrica}', 
                               labels={coluna_metrica: rotulo_metrica}, 
                               color_discrete_sequence=[AZUL_DESTAQUE])
        fig_dist.update_layout(**layout_base)

        # Gráfico DNA da Loja
        fig_dna = px.scatter(
            df_filtrado_loja, x="Customers", y="Sales",
            title=f"DNA da Loja: Vendas vs. Clientes - Loja {id_loja}",
            labels={'Customers': 'Número de Clientes (por dia)', 'Sales': 'Vendas (por dia)'},
            trendline="ols", trendline_color_override=AZUL_ESCURO
        )
        fig_dna.update_layout(**layout_base)

        # Organiza gráficos em abas
        componente_abas = dbc.Tabs([
            dbc.Tab(dcc.Graph(figure=fig_ts),   label="Série Temporal",  tab_id="ts"),
            dbc.Tab(dcc.Graph(figure=fig_promo),label="Promoção",        tab_id="promo"),
            dbc.Tab(dcc.Graph(figure=fig_dia_semana), label="Dia-Semana", tab_id="dow"),
            dbc.Tab(dcc.Graph(figure=fig_dist), label="Distribuição",    tab_id="dist"),
            dbc.Tab(dcc.Graph(figure=fig_dna),  label="DNA da Loja",     tab_id="dna")
        ], id=f"abas-loja-{id_loja}", active_tab="ts", className="mt-3 rossmann-tabs")

        return html.Div([
            card_kpi,
            card_detalhes_estaticos,
            componente_abas
        ])

    def criar_coluna_comparacao(id_loja, df_filtrado_loja, str_ranking): # Refatorar nome da função e parâmetros
        media_vendas = df_filtrado_loja['Sales'].mean() # Refatorar nome da variable
        media_clientes = df_filtrado_loja['Customers'].mean() # Refatorar nome da variable
        media_ticket = df_filtrado_loja['SalesPerCustomer'].mean() if df_filtrado_loja['SalesPerCustomer'].count() > 0 else 0 # Refatorar nome da variable

        def criar_kpi(titulo, valor, eh_moeda=True): # Refatorar nome da função e parâmetro
            valor_formatado = f"€ {valor:,.2f}" if eh_moeda else f"{valor:,.0f}" # Refatorar nome da variable
            return html.Div(
                [
                    html.Div(
                        titulo,
                        className="text-muted",
                        style={
                            'fontSize': '0.9rem',
                            'marginBottom': '8px'
                        }
                    ),
                    html.Div(
                        valor_formatado, # Usar nova variável refatorada
                        style={
                            'fontSize': '1.75rem',
                            'fontFamily': 'monospace',
                            'fontWeight': '500',
                            'color': '#2c3e50'
                        }
                    )
                ],
                className="mb-4"
            )

        return dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Row(
                        [
                            dbc.Col(
                                html.H2(
                                    f"Loja {id_loja}",
                                    style={
                                        'fontSize': '1.75rem',
                                        'fontWeight': '600',
                                        'color': '#2c3e50',
                                        'marginBottom': '0'
                                    }
                                ),
                                width="auto"
                            ),
                            dbc.Col(
                                html.Div(
                                    f"Ranking: {str_ranking}", # Usar nova variável refatorada
                                    className="text-muted",
                                    style={
                                        'fontSize': '1.1rem',
                                        'textAlign': 'right'
                                    }
                                ),
                                width="auto",
                                className="ms-auto"
                            )
                        ],
                        className="align-items-center",
                        style={'marginBottom': '0'}
                    ),
                    style={
                        'backgroundColor': '#ffffff',
                        'borderBottom': '1px solid #e9ecef'
                    }
                ),
                dbc.CardBody(
                    [
                        criar_kpi("Vendas Médias/Dia", media_vendas), # Usar nova função e variável refatorada
                        criar_kpi("Clientes Médios/Dia", media_clientes, False), # Usar nova função e variável refatorada
                        criar_kpi("Ticket Médio", media_ticket) # Usar nova função e variável refatorada
                    ],
                    className="pt-3 px-3"
                )
            ],
            className="h-100 shadow-sm",
            style={'backgroundColor': '#ffffff'}
        )

    def gerar_comparacao_detalhada(id_loja1, id_loja2, df_filtrado1, df_filtrado2): # Refatorar nome da função e parâmetros
        """Gera o conteúdo detalhado da comparação entre lojas."""
        if df_filtrado1.empty or df_filtrado2.empty:
            return html.Div("Dados insuficientes para comparação.")

        # Calcular métricas para comparação
        metricas = { # Refatorar nome da variable
            'Vendas Médias/Dia': ('Sales', 'mean', '€ {:,.2f}'),
            'Clientes Médios/Dia': ('Customers', 'mean', '{:,.0f}'),
            'Ticket Médio': ('SalesPerCustomer', 'mean', '€ {:,.2f}'),
            'Vendas Totais': ('Sales', 'sum', '€ {:,.2f}')
        }

        metricas_loja1 = {nome: df_filtrado1[col].agg(func) for nome, (col, func, _) in metricas.items()} # Refatorar nome da variable
        metricas_loja2 = {nome: df_filtrado2[col].agg(func) for nome, (col, func, _) in metricas.items()} # Refatorar nome da variable

        # Calcular diferenças percentuais
        diferencas = {} # Refatorar nome da variable
        for metrica_nome in metricas.keys(): # Refatorar nome da variable
            val1, val2 = metricas_loja1[metrica_nome], metricas_loja2[metrica_nome] # Refatorar nome das variables
            if val2 != 0:  # Evitar divisão por zero
                pct_diferenca = ((val1 - val2) / val2) * 100 # Refatorar nome da variable
                diferencas[metrica_nome] = pct_diferenca

        def criar_linha_metrica(nome_metrica, formato): # Refatorar nome da função e parâmetros
            val1, val2 = metricas_loja1[nome_metrica], metricas_loja2[nome_metrica] # Refatorar nome das variables
            pct_diferenca = diferencas.get(nome_metrica, 0) # Refatorar nome da variable
            eh_positivo = pct_diferenca > 0 # Refatorar nome da variable

            return html.Div(
                [
                    html.H6(
                        nome_metrica,
                        className="text-muted mb-2",
                        style={
                            'color': '#2c3e50',
                            'fontSize': '0.85rem',
                            'fontWeight': '500'
                        }
                    ),
                    dbc.Row(
                        [
                            # Valor Loja 1
                            dbc.Col(
                                [
                                    html.Div(
                                        formato.format(val1), # Usar nova variável refatorada
                                        style={
                                            'fontSize': '1.4rem',
                                            'fontWeight': '500',
                                            'fontFamily': 'monospace',
                                            'color': '#2c3e50',
                                            'textAlign': 'center',
                                            'lineHeight': '1.2'
                                        }
                                    ),
                                    html.Div(
                                        f"Loja {id_loja1}",
                                        className="text-muted text-center",
                                        style={
                                            'fontSize': '0.75rem',
                                            'marginTop': '2px'
                                        }
                                    )
                                ],
                                width=5,
                                className="pe-0"
                            ),
                            # Diferença Percentual
                            dbc.Col(
                                html.Div(
                                    [
                                        html.I(
                                            className=f"fas fa-arrow-{'up' if eh_positivo else 'down'}", # Usar nova variável refatorada
                                            style={
                                                'color': 'green' if eh_positivo else 'red', # Usar nova variável refatorada
                                                'fontSize': '0.8rem'
                                            }
                                        ),
                                        html.Span(
                                            f" {abs(pct_diferenca):.1f}%", # Usar nova variável refatorada
                                            style={
                                                'fontFamily': 'monospace',
                                                'fontSize': '0.9rem'
                                            }
                                        )
                                    ],
                                    style={
                                        'color': 'green' if eh_positivo else 'red', # Usar nova variável refatorada
                                        'fontWeight': '500',
                                        'textAlign': 'center',
                                        'backgroundColor': f"rgba({0 if eh_positivo else 255}, {255 if eh_positivo else 0}, 0, 0.1)", # Usar nova variável refatorada
                                        'borderRadius': '12px',
                                        'padding': '2px 8px',
                                        'display': 'inline-block',
                                        'margin': '0 auto'
                                    }
                                ),
                                width=2,
                                className="d-flex align-items-center justify-content-center px-0"
                            ),
                            # Valor Loja 2
                            dbc.Col(
                                [
                                    html.Div(
                                        formato.format(val2), # Usar nova variável refatorada
                                        style={
                                            'fontSize': '1.4rem',
                                            'fontWeight': '500',
                                            'fontFamily': 'monospace',
                                            'color': '#2c3e50',
                                            'textAlign': 'center',
                                            'lineHeight': '1.2'
                                        }
                                    )
,
                                    html.Div(
                                        f"Loja {id_loja2}",
                                        className="text-muted text-center",
                                        style={
                                            'fontSize': '0.75rem',
                                            'marginTop': '2px'
                                        }
                                    )
                                ],
                                width=5,
                                className="ps-0"
                            )
                        ],
                        className="align-items-center g-0"
                    )
                ],
                className="mb-3 p-2",
                style={
                    'backgroundColor': '#ffffff',
                    'borderRadius': '8px',
                    'border': '1px solid #e9ecef',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.05)'
                }
            )

        # Criar linhas de comparação para cada métrica
        linhas_comparacao = [ # Refatorar nome da variable
            criar_linha_metrica(nome, fmt)
            for nome, (_, _, fmt) in metricas.items()
        ]

        # Análise de Desempenho
        analise_desempenho = [] # Refatorar nome da variable

        # Análise de Vendas
        diferenca_vendas = diferencas['Vendas Médias/Dia'] # Refatorar nome da variable
        diferenca_clientes = diferencas['Clientes Médios/Dia'] # Refatorar nome da variable
        diferenca_ticket = diferencas['Ticket Médio'] # Refatorar nome da variable

        # Gerar insights automáticos baseados nas diferenças
        insights = [] # Refatorar nome da variable

        # Insight sobre vendas
        if abs(diferenca_vendas) > 2:
            insights.append(
                html.Li([
                    "Vendas: ",
                    html.Span(
                        f"{'Superioridade' if diferenca_vendas > 0 else 'Inferioridade'} de {abs(diferenca_vendas):.1f}% ",
                        style={'color': 'green' if diferenca_vendas > 0 else 'red', 'fontWeight': '500'}
                    ),
                    "nas vendas médias diárias"
                ])
            )

        # Insight sobre clientes
        if abs(diferenca_clientes) > 2:
            insights.append(
                html.Li([
                    "Clientes: ",
                    html.Span(
                        f"{'Maior' if diferenca_clientes > 0 else 'Menor'} fluxo em {abs(diferenca_clientes):.1f}% ",
                        style={'color': 'green' if diferenca_clientes > 0 else 'red', 'fontWeight': '500'}
                    ),
                    "de clientes por dia"
                ])
            )

        # Insight sobre ticket médio
        if abs(diferenca_ticket) > 2:
            insights.append(
                html.Li([
                    "Ticket: ",
                    html.Span(
                        f"{'Superior' if diferenca_ticket > 0 else 'Inferior'} em {abs(diferenca_ticket):.1f}% ",
                        style={'color': 'green' if diferenca_ticket > 0 else 'red', 'fontWeight': '500'}
                    ),
                    "no valor médio por cliente"
                ])
            )

        if insights:
            analise_desempenho.extend([
                html.H6(
                    "Principais Diferenças",
                    style={
                        'fontSize': '0.9rem',
                        'fontWeight': '600',
                        'color': '#2c3e50',
                        'marginBottom': '8px'
                    }
                ),
                html.Ul(
                    insights,
                    style={
                        'paddingLeft': '20px',
                        'fontSize': '0.85rem',
                        'marginBottom': '0'
                    }
                )
            ])
        else:
            analise_desempenho.append(
                html.P(
                    "As lojas apresentam desempenho similar em todas as métricas analisadas.",
                    className="text-muted mb-0",
                    style={'fontSize': '0.85rem'}
                )
            )

        return html.Div([
            # Métricas Comparativas
            html.Div(
                linhas_comparacao, # Usar nova variável refatorada
                style={'marginBottom': '1rem'}
            ),
            # Análise de Desempenho
            html.Div(
                analise_desempenho, # Usar nova variável refatorada
                style={
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px',
                    'padding': '12px 16px'
                }
            )
        ])

    def gerar_visualizacao_comparacao(ids_lojas, dados_json, ordem_ranking, data_inicio, data_fim, feriado_estadual, feriado_escolar, metrica_ranking, df_principal_json):
        """Gera a visualização comparativa entre duas lojas."""
        if len(ids_lojas) != 2:
            return dash.no_update

        # Verificar se temos os dados do DataFrame principal
        if not df_principal_json:
            return dbc.Alert("Erro interno: DataFrame principal não encontrado.", color="danger")
            
        # Usar a função auxiliar para deserializar o DataFrame
        df_principal = deserializar_df(df_principal_json)
        if df_principal is None:
            return dbc.Alert("Erro interno: Falha ao processar os dados.", color="danger")

        id_loja1, id_loja2 = ids_lojas

        tipo_loja1 = df_principal[df_principal['Store'] == id_loja1]['StoreType'].iloc[0]
        tipo_loja2 = df_principal[df_principal['Store'] == id_loja2]['StoreType'].iloc[0]

        # Filtra dados para cada loja
        df_filtrado1 = filtrar_dataframe(df_principal, data_inicio, data_fim, [tipo_loja1], [id_loja1], feriado_estadual, feriado_escolar)
        df_filtrado2 = filtrar_dataframe(df_principal, data_inicio, data_fim, [tipo_loja2], [id_loja2], feriado_estadual, feriado_escolar)

        if df_filtrado1.empty or df_filtrado2.empty:
            return dbc.Alert("Não foram encontrados dados para uma ou ambas as lojas com os filtros atuais.", color="warning")

        # CORREÇÃO CRÍTICA: Ordenar os DataFrames por data antes de plotar as séries temporais
        df_filtrado1 = df_filtrado1.sort_values(by='Date')
        df_filtrado2 = df_filtrado2.sort_values(by='Date')

        # --- Mapeamento de Métricas e Labels ---
        mapeamento_metrica = {
            'Sales_sum': 'Sales',
            'Sales_mean': 'Sales',
            'Customers_sum': 'Customers',
            'Customers_mean': 'Customers',
            'SalesPerCustomer_mean': 'SalesPerCustomer'
        }
        mapeamento_rotulo = {
            'Sales': 'Vendas',
            'Customers': 'Clientes',
            'SalesPerCustomer': 'Ticket Médio'
        }
        coluna_metrica = mapeamento_metrica.get(metrica_ranking, 'Sales')
        rotulo_metrica = mapeamento_rotulo.get(coluna_metrica, 'Vendas')

        # --- Gera Colunas de Detalhes e KPIs ---
        # Ranking de cada loja
        str_ranking1, str_ranking2 = "N/A", "N/A"
        if dados_json:
            try:
                df_ranking = pd.read_json(StringIO(dados_json), orient='split')
                if not df_ranking.empty:
                    total = len(df_ranking)
                    def obter_ranking(id_loja):
                        linha = df_ranking[df_ranking['Store'] == id_loja]
                        if not linha.empty:
                            return f"{int(linha['Ranking'].iloc[0])}º de {total}"
                        return "N/A"
                    str_ranking1 = obter_ranking(id_loja1)
                    str_ranking2 = obter_ranking(id_loja2)
            except Exception:
                pass

        # Botão de comparação removido/oculto conforme solicitado
        botao_comparacao = dbc.Button(
            html.I(className="fas fa-chart-line"),
            id="abrir-modal-comparacao",
            style={'display': 'none'},
            n_clicks=0
        )

        # Modal para mostrar a comparação detalhada
        conteudo_comparacao = gerar_comparacao_detalhada(id_loja1, id_loja2, df_filtrado1, df_filtrado2)

        # KPI row com o botão de comparação ao lado do título
        linha_kpi = html.Div([
            # Título e botão na mesma linha
            dbc.Row(
                [
                    dbc.Col(
                        html.H1(
                            "Detalhes da Loja Selecionada",
                            className="mb-0",
                            style={
                                'fontSize': '1.75rem',
                                'fontWeight': '600',
                                'color': '#2c3e50'
                            }
                        ),
                        width=True,
                        className="d-flex align-items-center"
                    ),
                ],
                className="mb-3 align-items-center"
            ),
            # Cards das lojas
            dbc.Row(
                [
                    dbc.Col(criar_coluna_comparacao(id_loja1, df_filtrado1, str_ranking1), width=6),
                    dbc.Col(criar_coluna_comparacao(id_loja2, df_filtrado2, str_ranking2), width=6),
                    conteudo_comparacao
                ],
                className="g-3 mb-4"
            ),
            # Botão oculto incluído em div escondida para manter callback funcionando
            html.Div(botao_comparacao, style={'display': 'none'})
        ])

        # Configurações comuns para os gráficos
        layout_base = {
            'height': 350,
            'margin': dict(t=40, b=40, l=50, r=20),
            'plot_bgcolor': '#ffffff',
            'paper_bgcolor': '#ffffff',
            'font': dict(size=12),
            'showlegend': True,
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        }

        # Série Temporal
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=df_filtrado1['Date'],
            y=df_filtrado1[coluna_metrica],
            mode='lines',
            name=f'Loja {id_loja1}',
            line=dict(color=VERMELHO_ROSSMANN, width=2)
        ))
        fig_ts.add_trace(go.Scatter(
            x=df_filtrado2['Date'],
            y=df_filtrado2[coluna_metrica],
            mode='lines',
            name=f'Loja {id_loja2}',
            line=dict(color=AZUL_ESCURO, width=2)
        ))
        fig_ts.update_layout(
            title=f'Série Temporal de {rotulo_metrica}',
            xaxis_title='Data',
            yaxis_title=rotulo_metrica,
            **layout_base
        )
        fig_ts.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig_ts.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        # Impacto da Promoção
        df_filtrado1['Store_ID_Label'] = f'Loja {id_loja1}'
        df_filtrado2['Store_ID_Label'] = f'Loja {id_loja2}'
        df_combinado = pd.concat([df_filtrado1, df_filtrado2])
        fig_promo = px.box(
            df_combinado,
            x='Promo',
            y=coluna_metrica,
            color='Store_ID_Label',
            title=f'Impacto da Promoção em {rotulo_metrica}',
            color_discrete_map={
                f'Loja {id_loja1}': VERMELHO_ROSSMANN,
                f'Loja {id_loja2}': AZUL_ESCURO
            }
        )
        fig_promo.update_layout(**layout_base)
        fig_promo.update_xaxes(
            tickvals=[0, 1],
            ticktext=['Sem Promoção', 'Com Promoção'],
            title='Status da Promoção'
        )
        fig_promo.update_yaxes(title=rotulo_metrica)

        # Média por Dia da Semana
        df_dia_semana1 = df_filtrado1.groupby('DayOfWeek')[coluna_metrica].mean().reset_index()
        df_dia_semana1['DayName'] = df_dia_semana1['DayOfWeek'].map(MAPEAMENTO_DIAS_SEMANA)
        df_dia_semana2 = df_filtrado2.groupby('DayOfWeek')[coluna_metrica].mean().reset_index()
        df_dia_semana2['DayName'] = df_dia_semana2['DayOfWeek'].map(MAPEAMENTO_DIAS_SEMANA)

        fig_dia_semana = go.Figure()
        fig_dia_semana.add_trace(go.Bar(
            name=f'Loja {id_loja1}',
            x=df_dia_semana1['DayName'],
            y=df_dia_semana1[coluna_metrica],
            marker_color=VERMELHO_ROSSMANN
        ))
        fig_dia_semana.add_trace(go.Bar(
            name=f'Loja {id_loja2}',
            x=df_dia_semana2['DayName'],
            y=df_dia_semana2[coluna_metrica],
            marker_color=AZUL_ESCURO
        ))
        fig_dia_semana.update_layout(
            barmode='group',
            title=f'Média de {rotulo_metrica} por Dia da Semana',
            xaxis={'categoryorder': 'array', 'categoryarray': ORDEM_DIAS_SEMANA, 'title': 'Dia da Semana'},
            yaxis_title=rotulo_metrica,
            **layout_base
        )

        # DNA da Loja (Comparativo)
        fig_dna_comp = go.Figure()
        fig_dna_comp.add_trace(go.Scatter(
            x=df_filtrado1['Customers'],
            y=df_filtrado1['Sales'],
            mode='markers',
            name=f'Loja {id_loja1}',
            marker=dict(color=VERMELHO_ROSSMANN, opacity=0.5, size=8)
        ))
        fig_dna_comp.add_trace(go.Scatter(
            x=df_filtrado2['Customers'],
            y=df_filtrado2['Sales'],
            mode='markers',
            name=f'Loja {id_loja2}',
            marker=dict(color=AZUL_ESCURO, opacity=0.5, size=8)
        ))

        # Adicionar linhas de tendência
        for df_loja, id_loja_loop, cor in [(df_filtrado1, id_loja1, VERMELHO_ROSSMANN), (df_filtrado2, id_loja2, AZUL_ESCURO)]:
            X = sm.add_constant(df_loja['Customers'])
            model = sm.OLS(df_loja['Sales'], X).fit()
            fig_dna_comp.add_trace(go.Scatter(
                x=df_loja['Customers'],
                y=model.predict(X),
                mode='lines',
                name=f'Tendência Loja {id_loja_loop}',
                line=dict(color=cor, width=2, dash='dash')
            ))

        # Layout específico para o gráfico DNA
        layout_dna = layout_base.copy()
        layout_dna.update({
            'legend': dict(
                orientation='h',
                yanchor='top',
                y=-0.3,
                xanchor='center',
                x=0.5
            ),
            'margin': dict(t=60, b=110, l=50, r=20)
        })
        
        fig_dna_comp.update_layout(
            title='DNA da Loja: Relação entre Vendas e Clientes',
            xaxis_title='Número de Clientes (por dia)',
            yaxis_title='Vendas (por dia)',
            **layout_dna
        )
        fig_dna_comp.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        fig_dna_comp.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')

        # Container principal com todos os gráficos empilhados
        container_graficos = dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=fig_ts,
                            config={'displayModeBar': False}
                        ),
                        width=12,
                        className="mb-4"
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=fig_promo,
                            config={'displayModeBar': False}
                        ),
                        width=12,
                        className="mb-4"
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=fig_dia_semana,
                            config={'displayModeBar': False}
                        ),
                        width=12,
                        className="mb-4"
                    )
                ]),
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(
                            figure=fig_dna_comp,
                            config={'displayModeBar': False}
                        ),
                        width=12
                    )
                ])
            ]),
            className="shadow-sm",
            style={'backgroundColor': '#ffffff'}
        )

        # Inclui o modal no layout para que ele seja recriado a cada seleção de loja
        return html.Div([
            linha_kpi,
            container_graficos
        ])

    # Callback para controlar o modal e seu conteúdo
    @aplicativo.callback(
        [Output("modal-comparacao", "is_open"),
         Output("modal-comparacao", "children")],
        [Input("abrir-modal-comparacao", "n_clicks"),
         Input("fechar-modal-comparacao", "n_clicks"),
         Input('armazenamento-id-loja-selecionada', 'data'),
         Input('armazenamento-dados-ranking', 'data'),
         Input('filtro-data', 'start_date'),
         Input('filtro-data', 'end_date'),
         Input('filtro-feriado-estadual', 'value'),
         Input('filtro-feriado-escolar', 'value'),
         Input('armazenamento-df-principal', 'data')],
        [State("modal-comparacao", "is_open")]
    )
    def atualizar_modal(n1, n2, ids_lojas_selecionadas, dados_json, data_inicio, data_fim, feriado_estadual, feriado_escolar, df_principal_json, esta_aberto):
        contexto = dash.callback_context
        id_gatilho = contexto.triggered[0]['prop_id'].split('.')[0]

        # Se não houver duas lojas selecionadas, não abre o modal
        if len(ids_lojas_selecionadas) != 2:
            return False, None

        # Usar a função auxiliar para deserializar o DataFrame
        df_principal = deserializar_df(df_principal_json)
        if df_principal is None:
            return False, dbc.Alert("Erro interno: DataFrame principal não encontrado.", color="danger")

        # Obtém os IDs das lojas selecionadas
        id_loja1, id_loja2 = ids_lojas_selecionadas

        # Obtém os tipos das lojas
        tipo_loja1 = df_principal[df_principal['Store'] == id_loja1]['StoreType'].iloc[0]
        tipo_loja2 = df_principal[df_principal['Store'] == id_loja2]['StoreType'].iloc[0]

        # Filtra dados para cada loja
        df_filtrado1 = filtrar_dataframe(df_principal, data_inicio, data_fim, [tipo_loja1], [id_loja1], feriado_estadual, feriado_escolar)
        df_filtrado2 = filtrar_dataframe(df_principal, data_inicio, data_fim, [tipo_loja2], [id_loja2], feriado_estadual, feriado_escolar)

        # CORREÇÃO CRÍTICA: Ordenar os DataFrames por data antes de plotar as séries temporais
        df_filtrado1 = df_filtrado1.sort_values(by='Date')
        df_filtrado2 = df_filtrado2.sort_values(by='Date')

        # Gera o conteúdo do modal
        conteudo_modal = [
            dbc.ModalHeader(
                html.H5(
                    "Análise Comparativa Detalhada",
                    className="mb-0",
                    style={
                        'fontSize': '1.1rem',
                        'fontWeight': '600',
                        'color': '#2c3e50'
                    }
                ),
                close_button=True
            ),
            dbc.ModalBody(
                [
                    # Métricas Comparativas
                    html.Div(
                        gerar_comparacao_detalhada(id_loja1, id_loja2, df_filtrado1, df_filtrado2),
                        style={'marginBottom': '1rem'}
                    )
                ]
            ),
            dbc.ModalFooter(
                dbc.Button(
                    "Fechar",
                    id="fechar-modal-comparacao",
                    className="ms-auto",
                    n_clicks=0,
                    style={
                        'backgroundColor': '#6c757d',
                        'border': 'none',
                        'fontSize': '0.9rem',
                        'padding': '6px 16px'
                    }
                )
            )
        ]

        # Decide se deve abrir/fechar o modal
        deve_abrir = esta_aberto
        if id_gatilho in ["abrir-modal-comparacao", "fechar-modal-comparacao"]:
            deve_abrir = not esta_aberto
        elif id_gatilho in ['armazenamento-id-loja-selecionada', 'armazenamento-dados-ranking']:
            deve_abrir = esta_aberto

        return deve_abrir, conteudo_modal

    @aplicativo.callback(
        [Output('conteudo-detalhe-loja', 'children'),
         Output('armazenamento-id-loja-selecionada', 'data')],
        [Input({'type': 'linha-ranking', 'index': dash.ALL}, 'n_clicks'),
         Input('armazenamento-dados-ranking', 'data'),
         Input('filtro-loja-especifica', 'value'),
         Input('armazenamento-df-principal', 'data')],
        [State({'type': 'linha-ranking', 'index': dash.ALL}, 'id'),
         State('armazenamento-id-loja-selecionada', 'data'),
         State('filtro-data', 'start_date'), State('filtro-data', 'end_date'),
         State('filtro-tipo-loja', 'value'),
         State('filtro-feriado-estadual', 'value'), State('filtro-feriado-escolar', 'value'),
         State('seletor-metrica-ranking', 'value'),
         State('seletor-ordem-ranking', 'value')]
    )
    def atualizar_detalhes_loja_e_selecao(lista_n_clicks, dados_json, selecao_lojas_especificas, df_principal_json,
                                           lista_id, ids_lojas_selecionadas, data_inicio, data_fim,
                                           tipos_loja, feriado_estadual, feriado_escolar, metrica_ranking,
                                           ordem_ranking):
        contexto = dash.callback_context
        id_propriedade_gatilho = contexto.triggered[0]['prop_id']
        
        # Inicializa a lista se for None
        if ids_lojas_selecionadas is None:
            ids_lojas_selecionadas = []
        novos_ids_selecionados = ids_lojas_selecionadas.copy()

        # Cenário 1: Seleção via dropdown de busca
        if id_propriedade_gatilho == 'filtro-loja-especifica.value':
            # Sempre queremos as duas últimas escolhidas no dropdown
            if selecao_lojas_especificas:
                if len(selecao_lojas_especificas) > 2:
                    novos_ids_selecionados = selecao_lojas_especificas[-2:]
                else:
                    novos_ids_selecionados = selecao_lojas_especificas.copy()

        # Cenário 2: Clique na tabela
        elif 'linha-ranking' in id_propriedade_gatilho:
            try:
                id_clicado = json.loads(id_propriedade_gatilho.split('.')[0])
                id_loja_clicada = id_clicado['index']
                if id_loja_clicada in novos_ids_selecionados:
                    # Se já está selecionada, remove (toggle)
                    novos_ids_selecionados.remove(id_loja_clicada)
                else:
                    if len(novos_ids_selecionados) < 2:
                        novos_ids_selecionados.append(id_loja_clicada)
                    else:
                        # Já existem duas lojas; substitui a mais antiga (primeira) pela nova
                        novos_ids_selecionados.pop(0)
                        novos_ids_selecionados.append(id_loja_clicada)
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

        # Cenário 3: Filtros principais (data, tipo) foram alterados
        elif id_propriedade_gatilho == 'armazenamento-dados-ranking.data':
            # Limpa seleção para evitar mostrar uma loja que não pertence ao novo ranking
            novos_ids_selecionados = []
            if dados_json:
                df_ranking = pd.read_json(StringIO(dados_json), orient='split')
                if not df_ranking.empty and len(df_ranking) > 0:
                    # O ranking já vem ordenado do callback anterior
                    loja_topo = df_ranking.iloc[0]
                    novos_ids_selecionados.append(loja_topo['Store'])

        # Usar a função auxiliar para deserializar o DataFrame
        df_principal = deserializar_df(df_principal_json)
        if df_principal is None:
            conteudo = dbc.Alert("Erro interno: DataFrame principal não encontrado.", color="danger")
            return conteudo, novos_ids_selecionados

        # Decide qual view renderizar
        if len(novos_ids_selecionados) == 2:
            conteudo = gerar_visualizacao_comparacao(novos_ids_selecionados, dados_json, ordem_ranking, data_inicio, data_fim, feriado_estadual, feriado_escolar, metrica_ranking, df_principal_json)
        elif len(novos_ids_selecionados) == 1:
            conteudo = gerar_visualizacao_loja_unica(novos_ids_selecionados[0], dados_json, ordem_ranking, data_inicio, data_fim, feriado_estadual, feriado_escolar, metrica_ranking, df_principal_json)
        else:
            conteudo = html.Div([
                html.I(className="fas fa-tasks me-2"),
                html.H5("Selecione até duas lojas", className="d-inline-block"),
                html.P("Clique em uma loja no ranking ou use a busca para ver os detalhes.", className="text-muted")
            ], className="text-center mt-5")

        return conteudo, novos_ids_selecionados

