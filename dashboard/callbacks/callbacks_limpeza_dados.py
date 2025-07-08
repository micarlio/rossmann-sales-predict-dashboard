from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from ..core.config import CINZA_NEUTRO, VERMELHO_ROSSMANN
from ..data.data_loader import get_data_states, N_AMOSTRAS_PADRAO


def registrar_callbacks_limpeza_dados(aplicativo, dados):
    """
    Registra callbacks para os controles e gráficos da página de Limpeza de Dados.
    Agora sem lógica de amostragem.
    """

    # Callback principal – atualiza os dois gráficos quando o DataFrame principal for alterado
    @aplicativo.callback(
        Output('grafico-impacto-media', 'figure'),
        Output('grafico-impacto-contagem', 'figure'),
        Input('armazenamento-df-principal', 'data')
    )
    def update_graficos_limpeza(store_data):
        # Obter estados dos dados (antes e depois da limpeza)
        states = get_data_states(use_samples=False)
        df_antes = states['antes']
        df_depois = states['depois']

        # --- Gráfico de Média de Vendas ---
        media_antes = df_antes['Sales'].mean() if 'Sales' in df_antes else 0
        media_depois = df_depois['Sales'].mean() if 'Sales' in df_depois else 0
        fig_media = go.Figure()
        fig_media.add_trace(go.Bar(x=['Antes da Limpeza'], y=[media_antes], name='Antes da Limpeza', marker_color=CINZA_NEUTRO))
        fig_media.add_trace(go.Bar(x=['Após Limpeza'], y=[media_depois], name='Após Limpeza', marker_color=VERMELHO_ROSSMANN))
        fig_media.update_layout(title='Impacto da Limpeza: Média de Vendas', yaxis_title='Média de Vendas', barmode='group')

        # --- Gráfico de Contagem de Registros ---
        cont_antes = len(df_antes)
        cont_depois = len(df_depois)
        fig_contagem = go.Figure()
        fig_contagem.add_trace(go.Bar(x=['Antes da Limpeza'], y=[cont_antes], name='Antes da Limpeza', marker_color=CINZA_NEUTRO))
        fig_contagem.add_trace(go.Bar(x=['Após Limpeza'], y=[cont_depois], name='Após Limpeza', marker_color=VERMELHO_ROSSMANN))
        fig_contagem.update_layout(title='Impacto da Limpeza: Contagem de Registros', yaxis_title='Número de Registros', barmode='group')

        return fig_media, fig_contagem 