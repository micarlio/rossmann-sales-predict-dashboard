# dashboard/callbacks/callbacks_analise_preliminar.py
from dash import Input, Output, html
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import pandas as pd

from ..utils import criar_figura_vazia, parse_json_to_df # Importar as funções utilitárias refatoradas
from ..config import VERMELHO_ROSSMANN, CINZA_NEUTRO, AZUL_DESTAQUE # Importar as novas constantes
from ..data_loader import get_data_states

LAYOUT_GRAFICO_COMUM = { # Refatorar nome da constante
    'title_x': 0.5,
    'margin': dict(l=80, r=40, b=40, t=90)
}

def registrar_callbacks_analise_preliminar(aplicativo, dados): # Refatorar nome da função e parâmetro 'app' para 'aplicativo'
    # Os dados de análise preliminar serão obtidos dinamicamente via get_data_states(use_samples=False)

    # Callback para atualizar a matriz de correlação quando o dataset principal mudar
    @aplicativo.callback(
        Output('grafico-matriz-correlacao', 'figure'),
        Input('armazenamento-df-principal', 'data')
    )
    def atualizar_matriz_correlacao(df_principal_json):
        """Atualiza a matriz de correlação com base no dataset principal atual."""
        # Obter DataFrame principal (limpo ou amostrado)
        df_principal = parse_json_to_df(df_principal_json)
        
        matriz_corr = df_principal.select_dtypes(include=np.number).corr()
        fig_matriz_corr = px.imshow(
            matriz_corr,
            text_auto='.2f',
            aspect="auto",
            title="Matriz de Correlação de Variáveis Numéricas",
            color_continuous_scale='Reds'
        )
        # Ajusta tamanho do gráfico e margens
        fig_matriz_corr.update_layout(height=850, width=1150, margin=dict(l=160, r=40, t=80, b=80))
        
        # Rotaciona rótulos do eixo X para evitar corte de texto e habilita margens automáticas
        fig_matriz_corr.update_xaxes(tickangle=-45, automargin=True)
        fig_matriz_corr.update_yaxes(automargin=True)
        
        # Mantém o texto dos valores em branco sobre os quadrados
        fig_matriz_corr.update_traces(textfont=dict(color='white'))
        
        return fig_matriz_corr

    # --- Callback para Scatter Plot da Matriz de Correlação ---
    @aplicativo.callback( # Usar 'aplicativo'
        Output('grafico-dispersao-correlacao', 'figure'),
        [Input('armazenamento-df-principal', 'data'), # Input do df principal dinâmico
         Input('grafico-matriz-correlacao', 'clickData')] # Refatorar ID
    )
    def exibir_dados_clicados(df_principal_json, dados_clicados): # Lê df dinâmico e trata clique
        # Obter DataFrame principal (limpo ou amostrado)
        df_principal = parse_json_to_df(df_principal_json)
        if dados_clicados is None:
            return criar_figura_vazia("Clique em uma célula da matriz")

        try:
            ponto = dados_clicados['points'][0] # Refatorar nome da variável
            col_x = ponto['x'] # Refatorar nome da variável
            col_y = ponto['y'] # Refatorar nome da variável

            if col_x not in df_principal.select_dtypes(include=np.number).columns or col_y not in df_principal.select_dtypes(include=np.number).columns: # Usar df_principal
                return criar_figura_vazia(f"Não é possível plotar '{col_x}' vs '{col_y}'. Selecione colunas numéricas.") # Usar a função refatorada

            tamanho_amostra = min(len(df_principal), 5000) # Refatorar nome da variável, usar df_principal
            df_amostra = df_principal.sample(n=tamanho_amostra, random_state=42) # Refatorar nome da variável, usar df_principal

            fig = px.scatter(df_amostra, x=col_x, y=col_y, title=f'Dispersão: {col_x} vs {col_y}', color_discrete_sequence=[VERMELHO_ROSSMANN], render_mode='webgl') # Usar df_amostra e constante refatorada
            fig.update_layout(
                **LAYOUT_GRAFICO_COMUM, # Usar constante refatorada
                plot_bgcolor='white'
            )

            df_temp = df_amostra[[col_x, col_y]].dropna() # Refatorar nome da variável
            if not df_temp.empty:
                X = sm.add_constant(df_temp[col_x])
                model = sm.OLS(df_temp[col_y], X)
                results = model.fit()
                fig.add_trace(go.Scatter(x=df_temp[col_x], y=results.predict(X), mode='lines', name='Linha de Tendência', line=dict(color=AZUL_DESTAQUE, width=3))) # Usar constante refatorada

            return fig
        except Exception as e:
            return criar_figura_vazia(f"Erro ao gerar gráfico de dispersão: {e}") # Usar a função refatorada

    # --- Callbacks para Histograma Comparativo e Estatísticas Dinâmicas ---
    @aplicativo.callback(
        Output('histograma-vendas-comparativo', 'figure'),
        [Input('armazenamento-df-principal', 'data'),
         Input('dropdown-histograma-vendas', 'value')]
    )
    def atualizar_histograma_vendas(df_json, coluna):
        # Obter DataFrame selecionado (cache ou JSON)
        df_sel = parse_json_to_df(df_json)
        # Dados brutos para comparação
        states = get_data_states(use_samples=False)
        df_raw = states['antes']
        if not coluna or coluna not in df_raw or coluna not in df_sel:
            return criar_figura_vazia('Selecione uma variável válida')
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_raw[coluna], name='Raw', opacity=0.7, marker_color=CINZA_NEUTRO))
        fig.add_trace(go.Histogram(x=df_sel[coluna], name='Selecionado', opacity=0.7, marker_color=VERMELHO_ROSSMANN))
        fig.update_layout(**LAYOUT_GRAFICO_COMUM, barmode='overlay', title=f'Distribuição de {coluna}', xaxis_title=coluna)
        return fig

    @aplicativo.callback(
        Output('histograma-lojas', 'figure'),
        [Input('armazenamento-df-principal', 'data'),
         Input('dropdown-histograma-lojas', 'value')]
    )
    def atualizar_histograma_lojas(df_json, coluna):
        # Obter DataFrame selecionado (cache ou JSON)
        df_sel = parse_json_to_df(df_json)
        states = get_data_states(use_samples=False)
        df_raw = states['antes']
        if not coluna or coluna not in df_raw or coluna not in df_sel:
            return criar_figura_vazia('Selecione uma variável válida')
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_raw[coluna], name='Raw', opacity=0.7, marker_color=CINZA_NEUTRO))
        fig.add_trace(go.Histogram(x=df_sel[coluna], name='Selecionado', opacity=0.7, marker_color=VERMELHO_ROSSMANN))
        fig.update_layout(**LAYOUT_GRAFICO_COMUM, barmode='overlay', title=f'Distribuição de {coluna}', xaxis_title=coluna)
        return fig

    @aplicativo.callback(
        Output('grafico-estatisticas-vendas', 'figure'),
        [Input('armazenamento-df-principal', 'data'),
         Input('dropdown-histograma-vendas', 'value')]
    )
    def atualizar_grafico_estatisticas_vendas(df_json, coluna):
        # Obter DataFrame selecionado (cache ou JSON)
        df_sel = parse_json_to_df(df_json)
        states = get_data_states(use_samples=False)
        df_raw = states['antes']
        if not coluna or coluna not in df_raw or coluna not in df_sel:
            return criar_figura_vazia('Selecione uma variável válida')
        est_raw = df_raw[coluna].describe()
        est_sel = df_sel[coluna].describe()
        df_stats = pd.DataFrame({'Raw': est_raw, 'Selecionado': est_sel}).reset_index().rename(columns={'index':'Métrica'})
        df_stats['Métrica'] = df_stats['Métrica'].replace({'25%':'Q1','50%':'Q2','75%':'Q3'})
        df_long = df_stats.melt(id_vars='Métrica', var_name='Estado', value_name='Valor')
        fig = px.bar(df_long, x='Métrica', y='Valor', color='Estado', barmode='group',
                     title=f'Estatísticas de {coluna}', text_auto='.2s',
                     color_discrete_map={'Raw':CINZA_NEUTRO,'Selecionado':VERMELHO_ROSSMANN})
        fig.update_layout(**LAYOUT_GRAFICO_COMUM, yaxis_type='log', yaxis_title='Valor (Escala Log)')
        fig.update_xaxes(categoryorder='array', categoryarray=['count','mean','std','min','Q1','Q2','Q3','max'])
        return fig

    @aplicativo.callback(
        Output('grafico-estatisticas-lojas', 'figure'),
        [Input('armazenamento-df-principal', 'data'),
         Input('dropdown-histograma-lojas', 'value')]
    )
    def atualizar_grafico_estatisticas_lojas(df_json, coluna):
        # Obter DataFrame selecionado (cache ou JSON)
        df_sel = parse_json_to_df(df_json)
        states = get_data_states(use_samples=False)
        df_raw = states['antes']
        if not coluna or coluna not in df_raw or coluna not in df_sel:
            return criar_figura_vazia('Selecione uma variável válida')
        if df_raw[coluna].dtype == 'object':
            return criar_figura_vazia(f'Métrica não aplicável para \'{coluna}\'.')
        est_raw = df_raw[coluna].describe()
        est_sel = df_sel[coluna].describe()
        df_stats = pd.DataFrame({'Raw': est_raw, 'Selecionado': est_sel}).reset_index().rename(columns={'index':'Métrica'})
        df_stats['Métrica'] = df_stats['Métrica'].replace({'25%':'Q1','50%':'Q2','75%':'Q3'})
        df_long = df_stats.melt(id_vars='Métrica', var_name='Estado', value_name='Valor')
        fig = px.bar(df_long, x='Métrica', y='Valor', color='Estado', barmode='group',
                     title=f'Estatísticas de {coluna}', text_auto='.2s',
                     color_discrete_map={'Raw':CINZA_NEUTRO,'Selecionado':VERMELHO_ROSSMANN})
        fig.update_layout(**LAYOUT_GRAFICO_COMUM, yaxis_type='log', yaxis_title='Valor (Escala Log)')
        fig.update_xaxes(categoryorder='array', categoryarray=['count','mean','std','min','Q1','Q2','Q3','max'])
        return fig