# dashboard/callbacks/callbacks_dashboard_geral.py
from dash import Input, Output, State, html
import dash
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from ..utils import criar_figura_vazia, filtrar_dataframe, parse_json_to_df
from ..config import (
    VERMELHO_ROSSMANN, AZUL_ESCURO, CINZA_NEUTRO, AZUL_DESTAQUE, VERDE_DESTAQUE,
    PALETA_CORES_GRAFICO, MAPEAMENTO_DIAS_SEMANA, ORDEM_DIAS_SEMANA,
    ALTURA_GRAFICO, ALTURA_GRAFICO_LARGURA_TOTAL
)

TITULOS_EIXO_Y = {
    'Sales': 'Vendas (€)',
    'Customers': 'Número de Clientes',
    'SalesPerCustomer': 'Ticket Médio (€)'
}

ROUTULOS_EIXO_Y = {
    'Sales': 'Vendas',
    'Customers': 'Clientes',
    'SalesPerCustomer': 'Ticket Médio'
}

def registrar_callbacks_dashboard_geral(aplicativo, dados):
    df_principal = dados["df_principal"]

    # --- Funções Auxiliares de Geração de Gráficos (Dashboard) ---
    def obter_grafico_serie_temporal(df_filtrado, tipo_granularidade, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y, lojas_especificas_selecionadas):
        df_temporal = df_filtrado.copy()

        chave_agrupamento = 'Store' if lojas_especificas_selecionadas else 'StoreType'
        entidade_titulo = "Loja" if lojas_especificas_selecionadas else "Tipo de Loja"

        if tipo_granularidade == 'M':
            df_temporal['Date_Period'] = df_temporal['Date'].dt.to_period('M').dt.to_timestamp()
            sufixo_titulo = 'Mensal'
        elif tipo_granularidade == 'W':
            df_temporal['Date_Period'] = df_temporal['Date'].dt.to_period('W').dt.to_timestamp()
            sufixo_titulo = 'Semanal'
        else:
            sufixo_titulo = 'Diária (Suavizado 7 dias)'

        if tipo_granularidade != 'D':
            df_agrupado = df_temporal.groupby(['Date_Period', chave_agrupamento], observed=False)[metrica].mean().reset_index()
            df_agrupado.rename(columns={metrica: 'Value'}, inplace=True)
        else:
            metrica_diaria = df_filtrado.groupby(['Date', chave_agrupamento], observed=False)[metrica].mean().unstack()
            metrica_suavizada = metrica_diaria.rolling(window=7, center=True, min_periods=1).mean()
            df_agrupado = metrica_suavizada.stack().reset_index(name='Value')
            df_agrupado.rename(columns={'Date': 'Date_Period'}, inplace=True)

        fig = px.line(df_agrupado, x='Date_Period', y='Value', color=chave_agrupamento, title=f'{texto_rotulo_eixo_y} por {entidade_titulo} ({sufixo_titulo})')
        fig.update_layout(
            height=ALTURA_GRAFICO_LARGURA_TOTAL,
            xaxis_title=f'Período ({sufixo_titulo})',
            yaxis_title=texto_titulo_eixo_y,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"O gráfico exibe a tendência de {texto_rotulo_eixo_y} por {entidade_titulo}. Ele permite observar a performance relativa e a sazonalidade de cada categoria ao longo do tempo, na granularidade selecionada ({sufixo_titulo})."
        return fig, texto_analise

    def obter_grafico_media_mensal(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_media_mensal = df_filtrado.groupby('Month', observed=False)[metrica].mean().reset_index()
        fig = px.line(df_media_mensal, x='Month', y=metrica, markers=True, title=f'Média de {texto_rotulo_eixo_y} por Mês', labels={metrica: texto_titulo_eixo_y, 'Month': 'Mês'}, color_discrete_sequence=[VERMELHO_ROSSMANN])
        fig.update_layout(
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13))),
            height=ALTURA_GRAFICO,
            yaxis_title=texto_titulo_eixo_y,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"Este gráfico mostra a sazonalidade anual da métrica '{texto_rotulo_eixo_y}'. Picos e vales podem indicar períodos de alta e baixa demanda, como festas de fim de ano ou meses de férias."
        return fig, texto_analise

    def obter_grafico_media_anual(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_media_anual = df_filtrado.groupby('Year', observed=False)[metrica].mean().reset_index()
        fig = px.line(df_media_anual, x='Year', y=metrica, markers=True, title=f'Média de {texto_rotulo_eixo_y} por Ano', labels={metrica: texto_titulo_eixo_y, 'Year': 'Ano'}, color_discrete_sequence=[VERMELHO_ROSSMANN])
        fig.update_layout(
            height=ALTURA_GRAFICO,
            yaxis_title=texto_titulo_eixo_y,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"A média de {texto_rotulo_eixo_y} por ano mostra a tendência geral ao longo do período selecionado. É útil para identificar crescimento, declínio ou estagnação no longo prazo."
        return fig, texto_analise

    def obter_grafico_promocao_tipo_loja(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_promo_tipo_loja = df_filtrado.groupby(['StoreType', 'Promo'], observed=True)[metrica].mean().reset_index()
        df_promo_tipo_loja['Promo'] = df_promo_tipo_loja['Promo'].map({0: 'Sem Promoção', 1: 'Com Promoção'})
        fig = px.bar(df_promo_tipo_loja, x='StoreType', y=metrica, color='Promo', barmode='group', text_auto='.0f', title=f'Promoção Vs {texto_rotulo_eixo_y} por Tipo de Loja', labels={metrica: texto_titulo_eixo_y, 'Promo': 'Status da Promoção', 'StoreType': 'Tipo de Loja'}, color_discrete_map={'Sem Promoção': CINZA_NEUTRO, 'Com Promoção': VERMELHO_ROSSMANN})
        fig.update_layout(height=ALTURA_GRAFICO, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        texto_analise = f"Este gráfico compara a média de {texto_rotulo_eixo_y} em dias com e sem promoção, para cada tipo de loja. É útil para avaliar a eficácia das promoções por segmento."
        return fig, texto_analise

    def obter_grafico_dia_semana(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_dia_semana = df_filtrado.groupby('DayOfWeek', observed=False)[metrica].mean().reset_index()
        df_dia_semana['DayName'] = df_dia_semana['DayOfWeek'].map(MAPEAMENTO_DIAS_SEMANA)
        fig = px.line(df_dia_semana, x='DayName', y=metrica, markers=True, title=f'{texto_rotulo_eixo_y} Médio por Dia da Semana', labels={metrica: texto_titulo_eixo_y, 'DayName': 'Dia da Semana'}, color_discrete_sequence=[VERMELHO_ROSSMANN])
        fig.update_layout(
            xaxis={'categoryorder':'array', 'categoryarray':ORDEM_DIAS_SEMANA},
            height=ALTURA_GRAFICO,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"Aqui vemos a variação média da métrica '{texto_rotulo_eixo_y}' ao longo da semana. Padrões podem indicar dias de maior movimento, como inícios de semana ou fins de semana."
        return fig, texto_analise

    def obter_grafico_dia_do_mes(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_dia_mes = df_filtrado.groupby('Day', observed=False)[metrica].mean().reset_index()
        fig = px.line(df_dia_mes, x='Day', y=metrica, markers=True, title=f'{texto_rotulo_eixo_y} Médio por Dia do Mês', labels={metrica: texto_titulo_eixo_y, 'Day': 'Dia do Mês'}, color_discrete_sequence=[VERMELHO_ROSSMANN])
        fig.update_layout(height=ALTURA_GRAFICO, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        texto_analise = f"Este gráfico revela o padrão de {texto_rotulo_eixo_y} ao longo do mês. Picos no início e no final do mês podem estar correlacionados com ciclos de pagamento de salários."
        return fig, texto_analise

    def obter_boxplot_promocao_tipo_loja(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        fig = px.box(df_filtrado, x='StoreType', y=metrica, color='Promo', title=f'Distribuição de {texto_rotulo_eixo_y} por Loja/Promo', labels={metrica: texto_titulo_eixo_y, 'StoreType': 'Tipo de Loja', 'Promo': 'Promoção Ativa?'}, color_discrete_map={0: CINZA_NEUTRO, 1: VERMELHO_ROSSMANN})
        fig.update_layout(
            boxmode='group',
            height=ALTURA_GRAFICO,
            yaxis_title=texto_titulo_eixo_y,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"O boxplot mostra a distribuição da métrica  '{texto_rotulo_eixo_y}' por tipo de loja, distinguindo dias com e sem promoção. Ajuda a entender a média e a consistência do impacto."
        return fig, texto_analise

    def obter_boxplot_promocao_geral(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        fig = go.Figure()
        fig.add_trace(go.Box(y=df_filtrado[df_filtrado['Promo'] == 0][metrica], name='Sem Promoção', marker_color=CINZA_NEUTRO))
        fig.add_trace(go.Box(y=df_filtrado[df_filtrado['Promo'] == 1][metrica], name='Com Promoção', marker_color=VERMELHO_ROSSMANN))
        fig.update_layout(
            title=f'Distribuição de {texto_rotulo_eixo_y} (Geral)',
            yaxis_title=texto_titulo_eixo_y,
            showlegend=True,
            legend_title_text='Promoção Ativa?',
            height=ALTURA_GRAFICO,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"Este boxplot compara a distribuição da métrica  '{texto_rotulo_eixo_y}' em dias com e sem promoção. Um deslocamento para cima na caixa 'Com Promoção' sugere impacto positivo"
        return fig, texto_analise

    def obter_histograma_promocao_geral(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_filtrado[df_filtrado['Promo'] == 0][metrica], name='Sem Promoção', marker_color=CINZA_NEUTRO, opacity=0.6, histnorm='density', nbinsx=50))
        fig.add_trace(go.Histogram(x=df_filtrado[df_filtrado['Promo'] == 1][metrica], name='Com Promoção', marker_color=VERMELHO_ROSSMANN, opacity=0.6, histnorm='density', nbinsx=50))
        fig.update_layout(
            barmode='overlay',
            title=f'Distribuição Comparativa de {texto_rotulo_eixo_y}',
            xaxis_title=texto_titulo_eixo_y,
            yaxis_title='Densidade',
            legend_title_text='Promoção Ativa?',
            height=ALTURA_GRAFICO,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        texto_analise = f"Este histograma de densidade compara a forma da distribuição de {texto_rotulo_eixo_y} para dias com e sem promoção. Curva vermelha à direita indica maiores valores com promoção."
        return fig, texto_analise

    def obter_grafico_impacto_distancia_concorrencia(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        """
        Gera um gráfico de dispersão (bubble chart) para analisar a relação entre
        a performance da loja, a distância do concorrente, o tipo de loja e o volume de clientes.
        """
        # Agrupar por loja para ter um ponto por loja no gráfico
        dados_nivel_loja = df_filtrado.groupby('Store', observed=False).agg(
            MetricValue=(metrica, 'mean'),
            CompetitionDistance=('CompetitionDistance', 'first'),
            StoreType=('StoreType', 'first'),
            AvgCustomers=('Customers', 'mean') # Usado para o tamanho da bolha
        ).dropna(subset=['CompetitionDistance', 'MetricValue'])

        if dados_nivel_loja.empty:
            return criar_figura_vazia("Sem dados suficientes para este gráfico."), "Não há lojas com dados de concorrência nos filtros selecionados."
        dados_nivel_loja['StoreType'] = dados_nivel_loja['StoreType'].cat.remove_unused_categories()
        fig = px.scatter(
            dados_nivel_loja,
            x='CompetitionDistance',
            y='MetricValue',
            size='AvgCustomers',
            color='StoreType',
            hover_name=dados_nivel_loja.index,
            title=f'Performance por Distância do Concorrente',
            labels={
                'CompetitionDistance': 'Distância do Concorrente (metros)',
                'MetricValue': f'Média de {texto_rotulo_eixo_y}',
                'StoreType': 'Tipo de Loja',
                'AvgCustomers': 'Média de Clientes'
            },
            size_max=60 # Define o tamanho máximo da bolha para não poluir o gráfico
        )

        fig.update_layout(height=ALTURA_GRAFICO, xaxis_title="Distância do Concorrente (metros)", yaxis_title=f"Média de {texto_rotulo_eixo_y}")
        texto_analise = f"Cada bolha representa uma loja. O gráfico mostra a relação entre a {texto_rotulo_eixo_y} (eixo y) e a distância do concorrente (eixo x). O tamanho da bolha indica o volume médio de clientes. É útil para identificar se lojas mais isoladas realmente performam melhor e para encontrar lojas atípicas (ex: perto de concorrentes, mas com alto volume e vendas)."
        return fig, texto_analise

    def obter_grafico_impacto_promo2(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_promo2_metrica = df_filtrado.groupby('Promo2', observed=False)[metrica].mean().reset_index()
        df_promo2_metrica['Promo2_Label'] = df_promo2_metrica['Promo2'].map({0: 'Não Participa', 1: 'Participa'})
        fig = px.bar(df_promo2_metrica, x='Promo2_Label', y=metrica, title=f'Média de {texto_rotulo_eixo_y} (Promo2)', labels={metrica: texto_titulo_eixo_y, 'Promo2_Label': 'Participação em Promo2'}, color='Promo2_Label', color_discrete_map={'Não Participa': CINZA_NEUTRO, 'Participa': VERMELHO_ROSSMANN})
        fig.update_layout(height=ALTURA_GRAFICO, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        texto_analise = f"Análise do impacto da 'Promo2' (promoção contínua) na média de {texto_rotulo_eixo_y}. Permite comparar o desempenho de lojas que participam deste programa com as que não participam."
        return fig, texto_analise

    def obter_grafico_impacto_sortimento(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_sortimento_metrica = df_filtrado.groupby('Assortment', observed=False)[metrica].mean().reset_index()
        fig = px.bar(df_sortimento_metrica, x='Assortment', y=metrica, title=f'{texto_rotulo_eixo_y} Médio por Tipo de Sortimento', labels={metrica: texto_titulo_eixo_y, 'Assortment': 'Tipo de Sortimento'}, color='Assortment')
        fig.update_layout(height=ALTURA_GRAFICO, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        texto_analise = f"O gráfico mostra como diferentes tipos de sortimento (a=básico, b=extra, c=estendido) se relacionam com a performance média da métrica '{texto_rotulo_eixo_y}'."
        return fig, texto_analise

    def obter_grafico_tipo_feriado(df_filtrado, metrica, texto_rotulo_eixo_y, texto_titulo_eixo_y):
        df_feriado_metrica = df_filtrado.groupby('StateHoliday', observed=False)[metrica].mean().reset_index()
        mapeamento_feriado = {'0': 'Dia Normal', 'a': 'Feriado Público', 'b': 'Páscoa', 'c': 'Natal'}
        df_feriado_metrica['StateHoliday_Label'] = df_feriado_metrica['StateHoliday'].map(mapeamento_feriado)
        ordem = [h for h in mapeamento_feriado.values() if h in df_feriado_metrica['StateHoliday_Label'].unique()]
        fig = px.bar(df_feriado_metrica, x='StateHoliday_Label', y=metrica, title=f'{texto_rotulo_eixo_y} Médio por Tipo de Dia/Feriado', labels={metrica: texto_titulo_eixo_y, 'StateHoliday_Label': 'Tipo de Feriado'}, color='StateHoliday_Label', color_discrete_map={'Dia Normal': CINZA_NEUTRO, 'Feriado Público': VERMELHO_ROSSMANN, 'Páscoa': AZUL_DESTAQUE, 'Natal': VERDE_DESTAQUE}, category_orders={'StateHoliday_Label': ordem})
        fig.update_layout(height=ALTURA_GRAFICO, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        texto_analise = f"Comparação da média de {texto_rotulo_eixo_y} em dias normais e feriados, destacando o impacto de feriados específicos, quando muitas lojas podem fechar."
        return fig, texto_analise

    def gerar_kpis(df_filtrado):
        """Gera os KPIs globais."""
        vendas_totais = df_filtrado['Sales'].sum()
        vendas_media_dia = df_filtrado['Sales'].mean()
        clientes_totais = df_filtrado['Customers'].sum()
        clientes_media_dia = df_filtrado['Customers'].mean()
        ticket_medio = df_filtrado['SalesPerCustomer'].mean() if df_filtrado['SalesPerCustomer'].count() > 0 else 0

        dados_kpi = [
            {"title": "Vendas Totais", "value": f"€{vendas_totais:,.0f}"},
            {"title": "Vendas Médias/Dia", "value": f"€{vendas_media_dia:,.2f}"},
            {"title": "Clientes Totais", "value": f"{clientes_totais:,.0f}"},
            {"title": "Clientes Médios/Dia", "value": f"{clientes_media_dia:,.0f}"},
            {"title": "Ticket Médio", "value": f"€{ticket_medio:,.2f}"}
        ]

        return [
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H6(kpi["title"], className="kpi-title text-muted"),
                            html.H3(kpi["value"], className="kpi-value fw-bold"),
                        ]
                    ),
                    className="kpi-card h-100 text-center shadow-sm"
                ),
                width=True
            ) for kpi in dados_kpi
        ]

    def gerar_kpis_por_tipo_loja(df_filtrado):
        """Gera os KPIs por tipo de loja."""
        colunas_kpi_tipo_loja = []
        tipos_loja_unicos = sorted(df_filtrado['StoreType'].unique())

        for tipo in tipos_loja_unicos:
            df_tipo_loja = df_filtrado[df_filtrado['StoreType'] == tipo]
            media_vendas_tipo = df_tipo_loja['Sales'].mean() if not df_tipo_loja.empty else 0
            media_clientes_tipo = df_tipo_loja['Customers'].mean() if not df_tipo_loja.empty else 0
            ticket_medio_tipo = df_tipo_loja['SalesPerCustomer'].mean() if df_tipo_loja['SalesPerCustomer'].count() > 0 else 0

            colunas_kpi_tipo_loja.append(
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader(f"Tipo {tipo.upper()}", className="kpi-header-storetype"),
                        dbc.CardBody([
                            html.P("Vendas Médias/Dia", className="kpi-label text-center"),
                            html.H4(f"€{media_vendas_tipo:,.0f}", className="kpi-value-storetype text-center mb-3"),
                            html.P("Clientes Médios/Dia", className="kpi-label text-center"),
                            html.H4(f"{media_clientes_tipo:,.0f}", className="kpi-value-storetype text-center mb-3"),
                            html.P("Ticket Médio", className="kpi-label text-center"),
                            html.H4(f"€{ticket_medio_tipo:,.2f}", className="kpi-value-storetype text-center")
                        ])
                    ], className="kpi-card h-100")
                )
            )

        return colunas_kpi_tipo_loja

    def verificar_valores_zero(df_filtrado):
        """Verifica se há lojas com vendas ou clientes zerados e retorna o alerta apropriado."""
        if df_filtrado['Sales'].mean() == 0 or df_filtrado['Customers'].mean() == 0:
            texto_alerta = html.P([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Atenção: Os dados filtrados incluem dias com Vendas ou Clientes zero. Isso pode indicar dias em que a loja estava aberta, mas sem registros de movimento. ",
                html.Span("Investigação adicional pode ser necessária.", style={'font-weight': 'bold'})
            ])
            estilo_alerta = {'display': 'block', 'background-color': '#F2C94C', 'color': AZUL_ESCURO, 'border': 'none'}
        else:
            texto_alerta = html.Div()
            estilo_alerta = {'display': 'none'}

        return texto_alerta, estilo_alerta

    def obter_grafico_comportamento_promocao(df_filtrado, metrica): # Removido o default 'SalesPerCustomer'
        """Gera um box plot comparando o comportamento do cliente em dias com e sem promoção."""

        titulos_eixo_y = {
            'SalesPerCustomer': 'Ticket Médio (€)',
            'Customers': 'Número de Clientes',
            'Sales': 'Vendas (€)'
        }
        titulo_eixo_y = titulos_eixo_y.get(metrica, 'Valor')

        if df_filtrado.empty or metrica not in df_filtrado.columns:
            return criar_figura_vazia("Sem dados para análise de comportamento."), html.P("Filtros selecionados não retornaram dados.")

        fig = px.box(
            df_filtrado,
            x='Promo',
            y=metrica,
            color='Promo',
            notched=True,
            labels={'Promo': 'Promoção Ativa?', metrica: titulo_eixo_y},
            category_orders={"Promo": [0, 1]},
            title=f'Distribuição de {titulo_eixo_y.replace(" (€)","")} por Promoção',
            color_discrete_map={0: CINZA_NEUTRO, 1: VERMELHO_ROSSMANN}
        )
        fig.update_layout(
            xaxis_tickvals=[0, 1],
            xaxis_ticktext=['Sem Promoção', 'Com Promoção'],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=ALTURA_GRAFICO-50
        )

        # Análise de texto genérica baseada na métrica
        textos_analise = {
            'SalesPerCustomer': "Este gráfico compara o ticket médio em dias com e sem promoção. Analise se os clientes tendem a gastar mais quando há promoções ativas.",
            'Customers': "Aqui, visualizamos a distribuição do número de clientes em dias promocionais e não promocionais. Permite avaliar se as promoções atraem um maior fluxo de pessoas.",
            'Sales': "O gráfico mostra a distribuição das vendas diárias. Compare os valores entre dias com e sem promoção para entender o impacto direto na receita."
        }
        texto_analise = textos_analise.get(metrica, "Análise do impacto da promoção na métrica selecionada.")

        return fig, texto_analise

    def obter_grafico_comportamento_sortimento(df_filtrado, metrica):
        """Gera um gráfico de barras comparando métricas por tipo de sortimento."""
        if metrica not in df_filtrado.columns or df_filtrado.empty:
            return criar_figura_vazia("Métrica não disponível para análise de comportamento."), html.P("Filtros selecionados não retornaram dados ou a métrica é inválida.")

        mapeamento_metrica = {
            'Sales': 'Vendas Médias',
            'Customers': 'Clientes Médios',
            'SalesPerCustomer': 'Ticket Médio'
        }
        rotulo_eixo_y = mapeamento_metrica.get(metrica, metrica)
        df_sortimento_metrica = df_filtrado.groupby('Assortment', observed=False)[metrica].mean().reset_index()
        fig = px.bar(df_sortimento_metrica, x='Assortment', y=metrica,
                     title=f'{rotulo_eixo_y} por Sortimento',
                     labels={metrica: f'{rotulo_eixo_y} (€)' if 'Sales' in metrica or 'SalesPerCustomer' in metrica else rotulo_eixo_y, 'Assortment': 'Tipo de Sortimento'},
                     color='Assortment'
        )
        fig.update_layout(height=ALTURA_GRAFICO-50, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # Análise de texto genérica baseada na métrica
        textos_analise = {
            'SalesPerCustomer': "Compare o ticket médio entre os diferentes tipos de sortimento (a=básico, b=extra, c=estendido). Isso ajuda a entender qual mix de produtos incentiva gastos maiores por cliente.",
            'Customers': "Este gráfico mostra a média de clientes por tipo de sortimento. Avalie qual sortimento é mais eficaz em atrair clientes para as lojas.",
            'Sales': "Analise as vendas médias para cada tipo de sortimento. Identifique qual estratégia de produtos gera maior receita para a rede."
        }
        texto_analise = textos_analise.get(metrica, "Comparativo da métrica selecionada entre os diferentes tipos de sortimento.")

        return fig, texto_analise

    # --- Callback Principal do Dashboard ---
    @aplicativo.callback(
        [
            Output('linha-kpi-dashboard', 'children'),
            Output('alerta-vendas-clientes-zero', 'children'),
            Output('alerta-vendas-clientes-zero', 'style'),
            Output('linha-kpi-tipo-loja', 'children'),
            Output('grafico-vendas-clientes-mensal-dashboard', 'figure'),
            Output('analise-vendas-clientes-mensal', 'children'),
            Output('grafico-vendas-clientes-anual-dashboard', 'figure'),
            Output('analise-vendas-clientes-anual', 'children'),
            Output('grafico-promocao-por-tipo-loja-dashboard', 'figure'),
            Output('analise-promocao-por-tipo-loja-dashboard', 'children'),
            Output('grafico-dia-semana-dashboard', 'figure'),
            Output('analise-dia-semana', 'children'),
            Output('grafico-dia-dashboard', 'figure'),
            Output('analise-dia', 'children'),
            Output('grafico-impacto-promocao-por-tipo-loja-boxplot', 'figure'),
            Output('analise-impacto-promocao-por-tipo-loja-boxplot', 'children'),
            Output('grafico-impacto-promocao-geral-boxplot', 'figure'),
            Output('analise-impacto-promocao-boxplot', 'children'),
            Output('grafico-impacto-promocao-geral-hist', 'figure'),
            Output('analise-impacto-promocao-hist', 'children'),
            Output('grafico-impacto-distancia-concorrencia', 'figure'),
            Output('analise-impacto-distancia-concorrencia', 'children'),
            Output('grafico-impacto-promo2', 'figure'),
            Output('analise-impacto-promo2', 'children'),
            Output('grafico-impacto-sortimento', 'figure'),
            Output('analise-impacto-sortimento', 'children'),
            Output('grafico-vendas-por-tipo-feriado', 'figure'),
            Output('analise-vendas-por-tipo-feriado', 'children')
        ],
        [
            Input('dashboard-filtro-data', 'start_date'),
            Input('dashboard-filtro-data', 'end_date'),
            Input('dashboard-filtro-tipo-loja', 'value'),
            Input('dashboard-filtro-loja-especifica', 'value'),
            Input('dashboard-filtro-metrica-temporal', 'value'),
            Input('dashboard-filtro-feriado-estadual', 'value'),
            Input('dashboard-filtro-feriado-escolar', 'value'),
            Input('armazenamento-df-principal', 'data')
        ]
    )
    def atualizar_pagina_dashboard(data_inicio, data_fim, tipos_loja_selecionados, lojas_especificas_selecionadas, metrica_temporal, feriado_estadual_selecionado, feriado_escolar_selecionado, df_principal_json):
        # Leitura dinâmica do df_principal a partir do dcc.Store
        try:
            df_principal = parse_json_to_df(df_principal_json)
        except Exception:
            df_principal = dados["df_principal"]

        if not data_inicio or not data_fim:
            figura_vazia = criar_figura_vazia("Período inválido")
            kpis_vazios = [html.Div("Selecione um período válido", className="alert alert-warning")] * 4
            return (
                kpis_vazios,                                     # linha-kpi-dashboard
                "Selecione um período válido",                   # alerta-vendas-clientes-zero children
                {'display': 'block'},                            # alerta-vendas-clientes-zero style
                kpis_vazios,                                     # linha-kpi-tipo-loja
                figura_vazia,                                    # grafico-vendas-clientes-mensal-dashboard
                "Selecione um período válido",                   # analise-vendas-clientes-mensal
                figura_vazia,                                    # grafico-vendas-clientes-anual-dashboard
                "Selecione um período válido",                   # analise-vendas-clientes-anual
                figura_vazia,                                    # grafico-promocao-por-tipo-loja-dashboard
                "Selecione um período válido",                   # analise-promocao-por-tipo-loja-dashboard
                figura_vazia,                                    # grafico-dia-semana-dashboard
                "Selecione um período válido",                   # analise-dia-semana
                figura_vazia,                                    # grafico-dia-dashboard
                "Selecione um período válido",                   # analise-dia
                figura_vazia,                                    # grafico-impacto-promocao-por-tipo-loja-boxplot
                "Selecione um período válido",                   # analise-impacto-promocao-por-tipo-loja-boxplot
                figura_vazia,                                    # grafico-impacto-promocao-geral-boxplot
                "Selecione um período válido",                   # analise-impacto-promocao-boxplot
                figura_vazia,                                    # grafico-impacto-promocao-geral-hist
                "Selecione um período válido",                   # analise-impacto-promocao-hist
                figura_vazia,                                    # grafico-impacto-distancia-concorrencia
                "Selecione um período válido",                   # analise-impacto-distancia-concorrencia
                figura_vazia,                                    # grafico-impacto-promo2
                "Selecione um período válido",                   # analise-impacto-promo2
                figura_vazia,                                    # grafico-impacto-sortimento
                "Selecione um período válido",                   # analise-impacto-sortimento
                figura_vazia,                                    # grafico-vendas-por-tipo-feriado
                "Selecione um período válido"                    # analise-vendas-por-tipo-feriado
            )

        if not all([metrica_temporal, feriado_estadual_selecionado, feriado_escolar_selecionado]):
            return dash.no_update # Evita erros durante a inicialização

        df_filtrado = filtrar_dataframe(df_principal, data_inicio, data_fim, tipos_loja_selecionados, lojas_especificas_selecionadas, feriado_estadual_selecionado, feriado_escolar_selecionado)

        if df_filtrado.empty:
            figura_vazia = criar_figura_vazia("Sem dados para os filtros selecionados")
            kpis_vazios = [html.Div("Sem dados para os filtros selecionados", className="alert alert-warning")] * 4
            return (
                kpis_vazios,                                     # linha-kpi-dashboard
                "Sem dados para os filtros selecionados",        # alerta-vendas-clientes-zero children
                {'display': 'block'},                            # alerta-vendas-clientes-zero style
                kpis_vazios,                                     # linha-kpi-tipo-loja
                figura_vazia,                                    # grafico-vendas-clientes-mensal-dashboard
                "Sem dados para os filtros selecionados",        # analise-vendas-clientes-mensal
                figura_vazia,                                    # grafico-vendas-clientes-anual-dashboard
                "Sem dados para os filtros selecionados",        # analise-vendas-clientes-anual
                figura_vazia,                                    # grafico-promocao-por-tipo-loja-dashboard
                "Sem dados para os filtros selecionados",        # analise-promocao-por-tipo-loja-dashboard
                figura_vazia,                                    # grafico-dia-semana-dashboard
                "Sem dados para os filtros selecionados",        # analise-dia-semana
                figura_vazia,                                    # grafico-dia-dashboard
                "Sem dados para os filtros selecionados",        # analise-dia
                figura_vazia,                                    # grafico-impacto-promocao-por-tipo-loja-boxplot
                "Sem dados para os filtros selecionados",        # analise-impacto-promocao-por-tipo-loja-boxplot
                figura_vazia,                                    # grafico-impacto-promocao-geral-boxplot
                "Sem dados para os filtros selecionados",        # analise-impacto-promocao-boxplot
                figura_vazia,                                    # grafico-impacto-promocao-geral-hist
                "Sem dados para os filtros selecionados",        # analise-impacto-promocao-hist
                figura_vazia,                                    # grafico-impacto-distancia-concorrencia
                "Sem dados para os filtros selecionados",        # analise-impacto-distancia-concorrencia
                figura_vazia,                                    # grafico-impacto-promo2
                "Sem dados para os filtros selecionados",        # analise-impacto-promo2
                figura_vazia,                                    # grafico-impacto-sortimento
                "Sem dados para os filtros selecionados",        # analise-impacto-sortimento
                figura_vazia,                                    # grafico-vendas-por-tipo-feriado
                "Sem dados para os filtros selecionados"         # analise-vendas-por-tipo-feriado
            )

        titulo_eixo_y = TITULOS_EIXO_Y[metrica_temporal]
        rotulo_eixo_y = ROUTULOS_EIXO_Y[metrica_temporal]

        # Gera os KPIs
        linha_kpis = gerar_kpis(df_filtrado)
        linha_kpis_tipo_loja = gerar_kpis_por_tipo_loja(df_filtrado)

        # Verifica se há lojas com vendas ou clientes zerados
        alerta_zero_filhos, estilo_alerta_zero = verificar_valores_zero(df_filtrado)

        # A variável filtro_loja_especifica_ativo não é mais necessária para o obter_grafico_serie_temporal,
        # pois a lógica de qual agrupamento usar foi movida para dentro da função.
        fig_vendas_clientes_mensal, analise_mensal_text = obter_grafico_media_mensal(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_vendas_clientes_anual, analise_vendas_clientes_anual_text = obter_grafico_media_anual(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_promocao_tipo_loja, analise_promocao_tipo_loja_text = obter_grafico_promocao_tipo_loja(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_dia_semana, analise_dia_semana_text = obter_grafico_dia_semana(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_dia, analise_dia_text = obter_grafico_dia_do_mes(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_impacto_promocao_tipo_loja_boxplot, analise_impacto_promocao_tipo_loja_boxplot_text = obter_boxplot_promocao_tipo_loja(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_impacto_promocao_geral_boxplot, analise_impacto_promocao_boxplot_text = obter_boxplot_promocao_geral(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_impacto_promocao_geral_hist, analise_impacto_promocao_hist_text = obter_histograma_promocao_geral(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_impacto_distancia_concorrencia, analise_impacto_distancia_concorrencia_text = obter_grafico_impacto_distancia_concorrencia(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_impacto_promo2, analise_impacto_promo2_text = obter_grafico_impacto_promo2(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_impacto_sortimento, analise_impacto_sortimento_text = obter_grafico_impacto_sortimento(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)
        fig_vendas_por_tipo_feriado, analise_vendas_por_tipo_feriado_text = obter_grafico_tipo_feriado(df_filtrado, metrica_temporal, rotulo_eixo_y, titulo_eixo_y)

        return (
            linha_kpis,                                    # linha-kpi-dashboard
            alerta_zero_filhos,                            # alerta-vendas-clientes-zero children
            estilo_alerta_zero,                            # alerta-vendas-clientes-zero style
            linha_kpis_tipo_loja,                          # linha-kpi-tipo-loja
            fig_vendas_clientes_mensal,                    # grafico-vendas-clientes-mensal-dashboard
            analise_mensal_text,                           # analise-vendas-clientes-mensal
            fig_vendas_clientes_anual,                     # grafico-vendas-clientes-anual-dashboard
            analise_vendas_clientes_anual_text,            # analise-vendas-clientes-anual
            fig_promocao_tipo_loja,                        # grafico-promocao-por-tipo-loja-dashboard
            analise_promocao_tipo_loja_text,               # analise-promocao-por-tipo-loja-dashboard
            fig_dia_semana,                                # grafico-dia-semana-dashboard
            analise_dia_semana_text,                       # analise-dia-semana
            fig_dia,                                       # grafico-dia-dashboard
            analise_dia_text,                              # analise-dia
            fig_impacto_promocao_tipo_loja_boxplot,        # grafico-impacto-promocao-por-tipo-loja-boxplot
            analise_impacto_promocao_tipo_loja_boxplot_text, # analise-impacto-promocao-por-tipo-loja-boxplot
            fig_impacto_promocao_geral_boxplot,            # grafico-impacto-promocao-geral-boxplot
            analise_impacto_promocao_boxplot_text,         # analise-impacto-promocao-boxplot
            fig_impacto_promocao_geral_hist,               # grafico-impacto-promocao-geral-hist
            analise_impacto_promocao_hist_text,            # analise-impacto-promocao-hist
            fig_impacto_distancia_concorrencia,            # grafico-impacto-distancia-concorrencia
            analise_impacto_distancia_concorrencia_text,   # analise-impacto-distancia-concorrencia
            fig_impacto_promo2,                            # grafico-impacto-promo2
            analise_impacto_promo2_text,                   # analise-impacto-promo2
            fig_impacto_sortimento,                        # grafico-impacto-sortimento
            analise_impacto_sortimento_text,               # analise-impacto-sortimento
            fig_vendas_por_tipo_feriado,                   # grafico-vendas-por-tipo-feriado
            analise_vendas_por_tipo_feriado_text           # analise-vendas-por-tipo-feriado
        )

    # --- Callbacks de Análise de Comportamento ---
    @aplicativo.callback(
        [Output('grafico-comportamento-promocao-boxplot', 'figure'),
         Output('analise-comportamento-promocao-boxplot', 'children')],
        [Input('armazenamento-df-principal', 'data'),
         Input('seletor-metrica-comportamento-promocao', 'value'),
         Input('dashboard-filtro-data', 'start_date'),
         Input('dashboard-filtro-data', 'end_date'),
         Input('dashboard-filtro-tipo-loja', 'value'),
         Input('dashboard-filtro-loja-especifica', 'value'),
         Input('dashboard-filtro-feriado-estadual', 'value'),
         Input('dashboard-filtro-feriado-escolar', 'value')]
    )
    def atualizar_grafico_comportamento_promocao(df_principal_json, metrica, data_inicio, data_fim, tipos_loja, lojas_especificas, feriado_estadual, feriado_escolar):
        if not df_principal_json:
            return dash.no_update, dash.no_update
        try:
            df_principal = parse_json_to_df(df_principal_json)
        except:
            df_principal = dados["df_principal"]
        if not metrica or not data_inicio or not data_fim:
            return dash.no_update, dash.no_update

        # Aplica os filtros globais ANTES de passar para a função do gráfico
        df_filtrado_global = filtrar_dataframe(df_principal, data_inicio, data_fim, tipos_loja, lojas_especificas, feriado_estadual, feriado_escolar)

        if df_filtrado_global.empty: # Verifica se há dados após o filtro
            return criar_figura_vazia("Sem dados para os filtros selecionados."), "Não há dados disponíveis para os filtros selecionados."

        fig, texto_analise = obter_grafico_comportamento_promocao(df_filtrado_global, metrica)
        return fig, texto_analise

    @aplicativo.callback(
        [Output('grafico-comportamento-sortimento-barras', 'figure'),
         Output('analise-comportamento-sortimento-barras', 'children')],
        [Input('armazenamento-df-principal', 'data'),
         Input('seletor-metrica-sortimento', 'value'),
         Input('dashboard-filtro-data', 'start_date'),
         Input('dashboard-filtro-data', 'end_date'),
         Input('dashboard-filtro-tipo-loja', 'value'),
         Input('dashboard-filtro-loja-especifica', 'value'),
         Input('dashboard-filtro-feriado-estadual', 'value'),
         Input('dashboard-filtro-feriado-escolar', 'value')]
    )
    def atualizar_grafico_comportamento_sortimento(df_principal_json, metrica, data_inicio, data_fim, tipos_loja, lojas_especificas, feriado_estadual, feriado_escolar):
        if not df_principal_json:
            return dash.no_update, dash.no_update
        try:
            df_principal = parse_json_to_df(df_principal_json)
        except:
            df_principal = dados["df_principal"]
        if not metrica or not data_inicio or not data_fim:
            return dash.no_update, dash.no_update

        # Aplica os filtros globais ANTES de passar para a função do gráfico
        df_filtrado_global = filtrar_dataframe(df_principal, data_inicio, data_fim, tipos_loja, lojas_especificas, feriado_estadual, feriado_escolar)

        if df_filtrado_global.empty: # Verifica se há dados após o filtro
            return criar_figura_vazia("Sem dados para os filtros selecionados."), "Não há dados disponíveis para os filtros selecionados."

        fig, texto_analise = obter_grafico_comportamento_sortimento(df_filtrado_global, metrica)
        return fig, texto_analise

    @aplicativo.callback(
        [Output('grafico-vendas-clientes-tempo-dashboard', 'figure'),
         Output('analise-vendas-clientes-tempo', 'children')],
        [Input('armazenamento-df-principal', 'data'),
         Input('filtro-granularidade', 'value'),
         Input('dashboard-filtro-metrica-temporal', 'value'),
         Input('dashboard-filtro-data', 'start_date'),
         Input('dashboard-filtro-data', 'end_date'),
         Input('dashboard-filtro-tipo-loja', 'value'),
         Input('dashboard-filtro-loja-especifica', 'value'),
         Input('dashboard-filtro-feriado-estadual', 'value'),
         Input('dashboard-filtro-feriado-escolar', 'value')]
    )
    def atualizar_grafico_serie_temporal(df_principal_json, granularidade, metrica, data_inicio, data_fim, tipos_loja, lojas_especificas, feriado_estadual, feriado_escolar):
        if not df_principal_json:
            return dash.no_update, dash.no_update
        try:
            df_principal = parse_json_to_df(df_principal_json)
        except:
            df_principal = dados["df_principal"]
        """Atualiza o gráfico de tendências temporais, respeitando os filtros globais."""
        # O filtro de granularidade é independente, mas os dados base já são filtrados globalmente
        df_filtrado = filtrar_dataframe(df_principal, data_inicio, data_fim, tipos_loja, lojas_especificas, feriado_estadual, feriado_escolar)

        if df_filtrado.empty:
            return criar_figura_vazia("Sem dados para o período selecionado."), "Não há dados disponíveis para os filtros selecionados."

        titulo_eixo_y = TITULOS_EIXO_Y[metrica]
        rotulo_eixo_y = ROUTULOS_EIXO_Y[metrica]

        # Passa a informação se lojas específicas foram selecionadas para a lógica de agrupamento dentro da função
        return obter_grafico_serie_temporal(
            df_filtrado,
            granularidade,
            metrica,
            rotulo_eixo_y,
            titulo_eixo_y,
            len(lojas_especificas) > 0 # Booleano para indicar se é para agrupar por loja ou tipo de loja
        )