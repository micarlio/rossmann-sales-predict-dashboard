from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from ..data_loader import get_data_states, N_AMOSTRAS_PADRAO
from ..config import CINZA_NEUTRO, VERMELHO_ROSSMANN


def registrar_callbacks_limpeza_dados(aplicativo, dados):
    """
    Registra callbacks para os controles e gráficos da página de Limpeza de Dados.
    """
    @aplicativo.callback(
        Output('container-input-amostras', 'style'),
        Input('seletor-modo-dados', 'value')
    )
    def toggle_input_amostras(modo):
        if modo == 'amostras':
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    @aplicativo.callback(
        Output('armazenamento-df-principal', 'data'),
        [Input('seletor-modo-dados', 'value'),
         Input('input-numero-amostras', 'value')]
    )
    def atualizar_dataset_automaticamente(modo, n_amostras):
        """
        Atualiza o dataset principal automaticamente quando o usuário muda o modo ou número de amostras.
        """
        # Validar número de amostras
        try:
            n = int(n_amostras)
            n_amostras_int = n if n > 0 else N_AMOSTRAS_PADRAO
        except:
            n_amostras_int = N_AMOSTRAS_PADRAO
        # Armazenar apenas a configuração de exibição para o store
        return {'modo': modo, 'n_amostras': n_amostras_int}

    @aplicativo.callback(
        Output('grafico-impacto-media', 'figure'),
        Output('grafico-impacto-contagem', 'figure'),
        Input('seletor-modo-dados', 'value'),
        Input('input-numero-amostras', 'value')
    )
    def update_graficos_limpeza(modo, n_amostras):
        use_samples = (modo == 'amostras')
        # Validar número de amostras
        try:
            n = int(n_amostras)
            n_amostras_int = n if n > 0 else N_AMOSTRAS_PADRAO
        except:
            n_amostras_int = N_AMOSTRAS_PADRAO

        # Obter os três estados dos dados
        states = get_data_states(use_samples=use_samples, n_amostras=n_amostras_int)
        df_antes = states['antes']
        df_depois = states['depois']
        df_amostrado = states['amostrado']

        # --- Gráfico de Média de Vendas ---
        media_antes = df_antes['Sales'].mean() if 'Sales' in df_antes else 0
        media_depois = df_depois['Sales'].mean() if 'Sales' in df_depois else 0
        media_amostra = df_amostrado['Sales'].mean() if use_samples and 'Sales' in df_amostrado else None
        fig_media = go.Figure()
        fig_media.add_trace(go.Bar(x=['Antes da Limpeza'], y=[media_antes], name='Antes da Limpeza', marker_color=CINZA_NEUTRO))
        fig_media.add_trace(go.Bar(x=['Após Limpeza'], y=[media_depois], name='Após Limpeza', marker_color=VERMELHO_ROSSMANN))
        if use_samples:
            fig_media.add_trace(go.Bar(x=[f'Amostra ({n_amostras_int} por loja)'], y=[media_amostra], name='Após Amostragem'))
        fig_media.update_layout(title='Impacto da Limpeza: Média de Vendas', yaxis_title='Média de Vendas', barmode='group')

        # --- Gráfico de Contagem de Registros ---
        cont_antes = len(df_antes)
        cont_depois = len(df_depois)
        cont_amostra = len(df_amostrado) if use_samples else None
        fig_contagem = go.Figure()
        fig_contagem.add_trace(go.Bar(x=['Antes da Limpeza'], y=[cont_antes], name='Antes da Limpeza', marker_color=CINZA_NEUTRO))
        fig_contagem.add_trace(go.Bar(x=['Após Limpeza'], y=[cont_depois], name='Após Limpeza', marker_color=VERMELHO_ROSSMANN))
        if use_samples:
            fig_contagem.add_trace(go.Bar(x=[f'Amostra ({n_amostras_int} por loja)'], y=[cont_amostra], name='Após Amostragem'))
        fig_contagem.update_layout(title='Impacto da Limpeza: Contagem de Registros', yaxis_title='Número de Registros', barmode='group')

        return fig_media, fig_contagem 