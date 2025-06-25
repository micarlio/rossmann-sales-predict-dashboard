# dashboard/callbacks/callbacks_analise_3d.py
from dash import Input, Output, State, ctx
import dash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import numpy as np

from ..utils import criar_figura_vazia, filtrar_dataframe_para_3d, parse_json_to_df # Importar as funções utilitárias refatoradas
from ..config import VERMELHO_ROSSMANN, AZUL_ESCURO, CINZA_NEUTRO, PALETA_CORES_GRAFICO, MAPEAMENTO_DIAS_SEMANA, ORDEM_DIAS_SEMANA # Importar as novas constantes

def registrar_callbacks_analise_3d(aplicativo, dados):
    """
    Registra todos os callbacks relacionados à análise 3D no aplicativo Dash.
    
    Args:
        aplicativo: Instância do aplicativo Dash
        dados: DataFrame contendo os dados das lojas
    """
    df_principal = dados["df_principal"] # Usar o novo nome do DataFrame principal

    @aplicativo.callback( # Usar 'aplicativo'
        Output('armazenamento-dados-base-3d', 'data'), # Refatorar ID
        [
            Input('armazenamento-df-principal', 'data'), # Input do df principal dinâmico
            Input('conteudo-pagina-/analise-3d', 'style'), # Refatorar ID
            Input('filtro-data-3d', 'start_date'),
            Input('filtro-data-3d', 'end_date'),
            Input('filtro-feriado-estadual-3d', 'value'), # Refatorar ID
            Input('filtro-feriado-escolar-3d', 'value') # Refatorar ID
        ],
        [State('armazenamento-dados-base-3d', 'data')] # Refatorar ID
    )
    def atualizar_dados_base_3d(df_principal_json, estilo_pagina, data_inicio, data_fim, feriado_estadual, feriado_escolar, dados_existentes): # Lê df dinâmico e trata cache
        """
        Filtra e otimiza os dados para a página 3D, com lógica de cache.
        - Calcula na primeira visita (quando o estilo da página muda para 'block').
        - Recalcula quando um filtro é alterado.
        - Mantém os dados em cache em navegações subsequentes.
        """
        # Obter DataFrame principal (limpo ou amostrado) via cache/util
        df_principal = parse_json_to_df(df_principal_json)
        id_gatilho = ctx.triggered_id # Refatorar nome da variável

        # Se o gatilho foi a visibilidade da página e já temos dados, usamos o cache.
        if id_gatilho == 'conteudo-pagina-/analise-3d' and dados_existentes: # Refatorar ID
            return dash.no_update

        # Se a página não está visível, não faz nada.
        if estilo_pagina and estilo_pagina.get('display') == 'none': # Refatorar nome da variável
            return dash.no_update

        # Para o primeiro carregamento ou mudança de filtro, processa os dados.
        data_inicio = data_inicio or df_principal['Date'].min().date() # Usar df_principal
        data_fim = data_fim or df_principal['Date'].max().date() # Usar df_principal
        feriado_estadual = feriado_estadual or 'all'
        feriado_escolar = feriado_escolar or 'all'

        df_filtrado = filtrar_dataframe_para_3d(df_principal, data_inicio, data_fim, feriado_estadual, feriado_escolar) # Refatorar nome da variável e função

        if df_filtrado.empty:
            return pd.DataFrame().to_json(date_format='iso', orient='split')

        colunas_manter = [ # Refatorar nome da variável
            'Store', 'StoreType', 'DayOfWeek', 'Month', 'Sales', 'Customers',
            'SalesPerCustomer', 'Promo', 'CompetitionDistance'
        ]
        colunas_existentes = [col for col in colunas_manter if col in df_filtrado.columns] # Refatorar nome da variável
        df_otimizado = df_filtrado[colunas_existentes] # Refatorar nome da variável

        return df_otimizado.to_json(date_format='iso', orient='split') # Retornar novo nome de variável


    def obter_grafico_superficie_sazonalidade(df_filtrado):
        """
        Cria um gráfico de superfície 3D para visualizar padrões de sazonalidade nas vendas.
        A superfície mostra a interação entre dias da semana e meses do ano.
        
        Args:
            df_filtrado: DataFrame com os dados filtrados para análise
            
        Returns:
            tuple: (figura do gráfico, texto de análise)
        """
        df_para_plotar = df_filtrado.copy()

        if df_para_plotar.empty:
            return criar_figura_vazia(f"Sem dados para a seleção atual"), "Ajuste os filtros para visualizar dados."

        # Cria uma tabela pivô com médias de vendas por dia da semana e mês
        pivo_vendas = pd.pivot_table(
            df_para_plotar,
            values='Sales',
            index='DayOfWeek',
            columns='Month',
            aggfunc='mean'
        ).fillna(0)

        # Garante que a grade esteja completa para uma superfície contínua
        pivo_vendas = pivo_vendas.reindex(columns=range(1, 13), fill_value=0)
        pivo_vendas = pivo_vendas.rename(index=MAPEAMENTO_DIAS_SEMANA)
        pivo_vendas = pivo_vendas.reindex(ORDEM_DIAS_SEMANA, fill_value=0)

        dados_z = pivo_vendas.values
        dados_x = pivo_vendas.columns
        dados_y = pivo_vendas.index

        # Cria o gráfico de superfície 3D
        fig = go.Figure(data=[go.Surface(
            z=dados_z, x=dados_x, y=dados_y,
            colorscale='Reds',
            colorbar=dict(title='Vendas Médias')
        )])
        
        # Configura o layout do gráfico
        fig.update_layout(
            title_text=None,
            autosize=True,
            scene=dict(
                xaxis_title='Mês',
                yaxis_title='Dia da Semana',
                zaxis_title='Vendas Médias (€)',
                camera=dict(eye=dict(x=1.8, y=1.8, z=0.8))
            ),
            margin=dict(l=10, r=10, b=10, t=10)
        )
        
        texto_analise = "A superfície 3D revela a sazonalidade das vendas, combinando padrões semanais e mensais. Picos indicam os dias da semana e meses com maior volume de vendas, enquanto vales mostram períodos de menor atividade. Use o filtro para analisar o comportamento de cada tipo de loja."
        return fig, texto_analise

    def obter_grafico_dispersao_3d_dinamica_promocao(df_filtrado): # Refatorar nome da função e parâmetro
        """Cria um gráfico de dispersão 3D para analisar a dinâmica das promoções."""
        if df_filtrado.empty or 'SalesPerCustomer' not in df_filtrado.columns or df_filtrado['SalesPerCustomer'].isnull().all():
            return criar_figura_vazia("Dados insuficientes para análise"), "Ajuste os filtros." # Usar a função refatorada

        # Amostragem para performance em grandes datasets
        tamanho_amostra = min(len(df_filtrado), 15000) # Refatorar nome da variável
        df_amostra = df_filtrado.sample(n=tamanho_amostra, random_state=42).copy() # Refatorar nome da variável
        df_amostra.dropna(subset=['Customers', 'SalesPerCustomer'], inplace=True)

        if df_amostra.empty:
            return criar_figura_vazia("Dados insuficientes após limpeza."), "Ajuste os filtros." # Usar a função refatorada

        df_amostra['Promo_Label'] = df_amostra['Promo'].map({0: 'Sem Promoção', 1: 'Com Promoção'})
        df_amostra['DayName'] = df_amostra['DayOfWeek'].map(MAPEAMENTO_DIAS_SEMANA) # Usar a constante refatorada

        fig = px.scatter_3d(
            df_amostra,
            x='DayName',
            y='Customers',
            z='SalesPerCustomer',
            color='Promo_Label',
            hover_data=['Store'],
            title=None, # Título já está no Card
            labels={
                "DayName": "Dia da Semana",
                "Customers": "Nº de Clientes",
                "SalesPerCustomer": "Ticket Médio (€)"
            },
            category_orders={"DayName": ORDEM_DIAS_SEMANA, "Promo_Label": ["Com Promoção", "Sem Promoção"]}, # Usar a constante refatorada
            color_discrete_map={'Sem Promoção': CINZA_NEUTRO, 'Com Promoção': VERMELHO_ROSSMANN} # Usar constantes refatoradas
        )
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(aspectmode='cube'),
            legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="center", x=0.5),
            legend_title_text=None
        )
        fig.update_traces(marker=dict(size=3.5, opacity=0.7))
        texto_analise = "Este gráfico investiga o comportamento do consumidor em resposta a promoções. Ele ajuda a responder se as promoções majoritariamente atraem mais clientes (pontos mais altos no eixo 'Clientes') ou se levam os clientes existentes a gastarem mais (pontos mais altos no eixo 'Ticket Médio'). A separação por cor revela como esse comportamento varia entre dias com e sem promoção ativa." # Refatorar nome da variável
        return fig, texto_analise


    def obter_grafico_dispersao_3d_fatores_loja(df_filtrado): # Refatorar nome da função e parâmetro
        """Cria um gráfico de dispersão 3D para fatores da loja."""
        if df_filtrado.empty:
            return criar_figura_vazia("Sem dados para a dispersão 3D"), "Filtre por pelo menos um tipo de loja." # Usar a função refatorada

        agg_loja = df_filtrado.groupby('Store').agg( # Refatorar nome da variável
            MetricValue=('Sales', 'mean'), # Renomeado para mais genérico
            Customers=('Customers', 'mean'), # Adicionado Customers para usar no gráfico
            CompetitionDistance=('CompetitionDistance', 'first'),
            StoreType=('StoreType', 'first')
        ).reset_index().dropna()

        if agg_loja.empty:
            return criar_figura_vazia("Dados agregados de loja insuficientes"), "Nenhuma loja encontrada para os tipos selecionados." # Usar a função refatorada

        fig = px.scatter_3d(
            agg_loja,
            x='CompetitionDistance',
            y='Customers',
            z='MetricValue', # Usar MetricValue para o Z
            color='StoreType',
            symbol='StoreType',
            hover_data=['Store'],
            title=None, # Título já está no Card
            labels={
                "CompetitionDistance": "Distância do Concorrente (m)",
                "Customers": "Média de Clientes",
                "MetricValue": "Média de Vendas (€)", # Título do eixo Z
                "StoreType": "Tipo de Loja"
            }
        )
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene = dict(aspectmode='cube'),
            legend_title_text='Tipo de Loja'
        )
        texto_analise = "Este gráfico correlaciona três fatores-chave: distância do concorrente, média de clientes e média de vendas. Cada ponto representa uma loja. Permite identificar se lojas com concorrentes mais próximos (eixo X) têm desempenho diferente em termos de fluxo de clientes (eixo Y) e receita (eixo Z)." # Refatorar nome da variável
        return fig, texto_analise

    def preprocessar_dados_3d(dados_json, tipos_loja, lojas_especificas):
        """
        Função auxiliar para pré-processar dados para os gráficos 3D.
        Realiza validação, leitura de JSON e filtragem dos dados.
        
        Args:
            dados_json: String JSON com os dados
            tipos_loja: Lista de tipos de loja para filtrar
            lojas_especificas: Lista de IDs de lojas específicas
            
        Returns:
            tuple: (DataFrame filtrado, figura de erro, mensagem de erro, estilo)
        """
        estilo_visivel = {'height': '65vh', 'visibility': 'visible'}

        try:
            df_principal = pd.read_json(StringIO(dados_json), orient='split')
            if df_principal.empty:
                figura = criar_figura_vazia("Sem dados para os filtros gerais")
                mensagem = "Altere o período ou os filtros de feriado."
                return None, figura, mensagem, estilo_visivel

            # Aplica filtros de loja
            if lojas_especificas:
                df_filtrado = df_principal[df_principal['Store'].isin(lojas_especificas)]
            elif tipos_loja:
                df_filtrado = df_principal[df_principal['StoreType'].isin(tipos_loja)]
            else:
                df_filtrado = df_principal

            if df_filtrado.empty:
                figura = criar_figura_vazia("Nenhuma loja encontrada para os filtros selecionados")
                mensagem = "Ajuste os filtros de loja para visualizar dados."
                return None, figura, mensagem, estilo_visivel

            return df_filtrado, None, None, estilo_visivel

        except Exception as e:
            figura = criar_figura_vazia(f"Erro ao processar dados: {str(e)}")
            return None, figura, "Ocorreu um erro ao processar os dados.", estilo_visivel


    @aplicativo.callback( # Usar 'aplicativo'
        [Output('grafico-superficie-3d', 'figure'), # Refatorar ID
         Output('analise-superficie-3d', 'children'), # Refatorar ID
         Output('grafico-superficie-3d', 'style')], # Refatorar ID
        [Input('armazenamento-dados-base-3d', 'data'), # Refatorar ID
         Input('filtro-tipo-loja-superficie', 'value'), # Refatorar ID
         Input('filtro-loja-especifica-superficie', 'value')] # Refatorar ID
    )
    def atualizar_grafico_superficie_3d(dados_json, tipos_loja, lojas_especificas): # Refatorar nome da função e parâmetros
        """Atualiza o gráfico de superfície 3D, controlando sua visibilidade."""
        if not dados_json:
            return dash.no_update, dash.no_update, dash.no_update

        df_filtrado, figura_erro, mensagem_erro, estilo = preprocessar_dados_3d(dados_json, tipos_loja, lojas_especificas) # Refatorar nome das variáveis e função
        if figura_erro is not None:
            return figura_erro, mensagem_erro, estilo

        fig, texto_analise = obter_grafico_superficie_sazonalidade(df_filtrado) # Refatorar nome das variáveis e função
        return fig, texto_analise, estilo

    @aplicativo.callback( # Usar 'aplicativo'
        [Output('grafico-dispersao-3d', 'figure'), # Refatorar ID
         Output('analise-dispersao-3d', 'children'), # Refatorar ID
         Output('grafico-dispersao-3d', 'style')], # Refatorar ID
        [Input('armazenamento-dados-base-3d', 'data'), # Refatorar ID
         Input('filtro-tipo-loja-fatores', 'value'), # Refatorar ID
         Input('filtro-loja-especifica-fatores', 'value')] # Refatorar ID
    )
    def atualizar_grafico_fatores_3d(dados_json, tipos_loja, lojas_especificas): # Refatorar nome da função e parâmetros
        """Atualiza o gráfico de dispersão 3D de fatores da loja, controlando sua visibilidade."""
        if not dados_json:
            return dash.no_update, dash.no_update, dash.no_update

        df_filtrado, figura_erro, mensagem_erro, estilo = preprocessar_dados_3d(dados_json, tipos_loja, lojas_especificas) # Refatorar nome das variáveis e função
        if figura_erro is not None:
            return figura_erro, mensagem_erro, estilo

        fig, texto_analise = obter_grafico_dispersao_3d_fatores_loja(df_filtrado) # Refatorar nome das variáveis e função
        return fig, texto_analise, estilo

    @aplicativo.callback( # Usar 'aplicativo'
        [Output('grafico-dinamica-promocao-3d', 'figure'), # Refatorar ID
         Output('analise-dinamica-promocao-3d', 'children'), # Refatorar ID
         Output('grafico-dinamica-promocao-3d', 'style')], # Refatorar ID
        [Input('armazenamento-dados-base-3d', 'data'), # Refatorar ID
         Input('filtro-tipo-loja-promocao', 'value'), # Refatorar ID
         Input('filtro-loja-especifica-promocao', 'value')] # Refatorar ID
    )
    def atualizar_grafico_promocao_3d(dados_json, filtro_tipos_loja, filtro_lojas_especificas): # Refatorar nome da função e parâmetros
        """Atualiza o gráfico de dispersão 3D da dinâmica de promoções, controlando sua visibilidade."""
        if not dados_json:
            return dash.no_update, dash.no_update, dash.no_update

        df_filtrado, figura_erro, mensagem_erro, estilo = preprocessar_dados_3d(dados_json, filtro_tipos_loja, filtro_lojas_especificas) # Refatorar nome das variáveis e função
        if figura_erro is not None:
            return figura_erro, mensagem_erro, estilo

        fig, texto_analise = obter_grafico_dispersao_3d_dinamica_promocao(df_filtrado) # Refatorar nome das variáveis e função
        return fig, texto_analise, estilo

    @aplicativo.callback(
        [Output('grafico-correlacao-3d', 'figure'),
         Output('analise-correlacao-3d', 'children'),
         Output('grafico-correlacao-3d', 'style')],
        [Input('armazenamento-dados-base-3d', 'data'),
         Input('filtro-tipo-loja-correlacao', 'value'),
         Input('filtro-loja-especifica-correlacao', 'value')]
    )
    def atualizar_grafico_correlacao_3d(dados_armazenados, tipos_loja, lojas_especificas):
        """
        Callback para atualizar o gráfico de correlação 3D.
        Mostra as relações entre diferentes variáveis numéricas do dataset.
        
        Args:
            dados_armazenados: Dados JSON armazenados
            tipos_loja: Lista de tipos de loja selecionados
            lojas_especificas: Lista de lojas específicas selecionadas
            
        Returns:
            tuple: (figura do gráfico, texto de análise, estilo de visualização)
        """
        if not dados_armazenados:
            return dash.no_update, dash.no_update, dash.no_update

        df_filtrado, figura_erro, mensagem_erro, estilo = preprocessar_dados_3d(dados_armazenados, tipos_loja, lojas_especificas)

        if figura_erro is not None:
            return figura_erro, mensagem_erro, estilo

        try:
            if df_filtrado.select_dtypes(include=np.number).shape[1] < 3:
                fig = criar_figura_vazia("Dados insuficientes para gerar matriz de correlação.")
                texto_analise = "Filtro resultou em dados insuficientes."
                return fig, texto_analise, estilo

            # Calcula a matriz de correlação
            matriz_corr = df_filtrado.select_dtypes(include=np.number).corr()

            for col in ['Sales', 'Customers', 'Promo']:
                if col not in matriz_corr.columns:
                    matriz_corr[col] = np.nan

            df_corr_3d = matriz_corr.reset_index().rename(columns={'index': 'Variavel'})
            df_corr_3d['Abs_Corr_Sales'] = abs(df_corr_3d['Sales'])

            # Cria o gráfico de dispersão 3D das correlações
            fig = px.scatter_3d(
                df_corr_3d,
                x='Sales', y='Customers', z='Promo',
                text='Variavel', hover_name='Variavel',
                color='Abs_Corr_Sales', color_continuous_scale='Reds',
                title=None
            )
            fig.update_traces(textposition='top center', marker=dict(size=5))
            fig.update_layout(
                scene=dict(
                    xaxis_title='Corr. Vendas',
                    yaxis_title='Corr. Clientes',
                    zaxis_title='Corr. Promoções'
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )

            num_lojas = df_filtrado['Store'].nunique()
            texto_analise = f"Para as {num_lojas} loja(s) selecionada(s), este gráfico mapeia como as variáveis se correlacionam com os três principais impulsionadores do negócio: Vendas, Clientes e Promoções. A posição de cada ponto no espaço revela a natureza dessas inter-relações."

            return fig, texto_analise, estilo

        except Exception as e:
            fig = criar_figura_vazia(f"Erro ao gerar gráfico: {str(e)}")
            return fig, "Ocorreu um erro ao gerar o gráfico.", estilo