import pandas as pd
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc
from .config import CINZA_NEUTRO, ALTURA_GRAFICO
from ..data.data_loader import get_principal_dataset, N_AMOSTRAS_PADRAO


def criar_figura_vazia(texto_titulo="Sem dados para os filtros selecionados", altura=ALTURA_GRAFICO):
    fig = go.Figure()
    fig.update_layout(
        height=altura,
        annotations=[dict(
            text=texto_titulo,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=CINZA_NEUTRO)
        )]
    )
    return fig


def criar_icone_informacao(id_icone, texto_tooltip):
    return html.Span([
        html.I(id=id_icone, className="fas fa-info-circle info-icon ms-2"),
        dbc.Tooltip(texto_tooltip, target=id_icone, placement='top')
    ], className="d-inline-block")

# ======================================================================================
# Funções Auxiliares de Manipulação de Dados – utilizadas em diversos callbacks/layouts
# ======================================================================================

def filtrar_dataframe(
    df: pd.DataFrame,
    data_inicio: str | None,
    data_fim: str | None,
    tipos_loja: list[str] | None,
    lojas_especificas: list[int] | None,
    feriado_estadual: str | None,
    feriado_escolar: str | None,
):
    """Aplica uma série de filtros comuns ao *DataFrame* principal.

    Esta versão cobre apenas as necessidades do dashboard; se algum filtro
    estiver inativo/padrão, ele é ignorado para simplificar a lógica.
    """
    df_filtrado = df.copy()

    # Filtro de período
    if data_inicio:
        df_filtrado = df_filtrado[df_filtrado['Date'] >= pd.to_datetime(data_inicio)]
    if data_fim:
        df_filtrado = df_filtrado[df_filtrado['Date'] <= pd.to_datetime(data_fim)]

    # Filtro de loja
    if lojas_especificas:
        df_filtrado = df_filtrado[df_filtrado['Store'].isin(lojas_especificas)]
    elif tipos_loja:
        if 'todos' not in tipos_loja:
            df_filtrado = df_filtrado[df_filtrado['StoreType'].isin(tipos_loja)]

    # Filtro de feriados
    if feriado_estadual and feriado_estadual != 'all':
        df_filtrado = df_filtrado[df_filtrado['StateHoliday'].astype(str) == feriado_estadual]
    if feriado_escolar and feriado_escolar != 'all':
        df_filtrado = df_filtrado[df_filtrado['SchoolHoliday'] == int(feriado_escolar)]

    return df_filtrado


def parse_json_to_df(store_data):
    """Converte o conteúdo armazenado no *dcc.Store* em *DataFrame*.

    Se *store_data* for um dicionário com pista de carregamento, o *DataFrame*
    é obtido por ``get_principal_dataset``; caso contrário assume-se que seja
    uma string JSON no formato *orient='split'*."""
    if store_data is None:
        return pd.DataFrame()

    # Caso seja o dicionário especial para sinalizar uso de cache
    if isinstance(store_data, dict) and 'modo' in store_data:
        modo = store_data.get('modo', 'completo')
        n_amostras = store_data.get('n_amostras', N_AMOSTRAS_PADRAO)
        use_samples = (modo == 'amostras')
        return get_principal_dataset(use_samples=use_samples, n_amostras=n_amostras)

    # Caso contrário, assumimos string JSON
    try:
        df = pd.read_json(store_data, orient='split')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except ValueError:
        # Retorna vazio em caso de erro – callbacks tratarão
        return pd.DataFrame()


def filtrar_dataframe_para_3d(
    df: pd.DataFrame,
    data_inicio: str | None,
    data_fim: str | None,
    feriado_estadual: str | None,
    feriado_escolar: str | None,
):
    """Versão reduzida do filtro para página 3D (não considera tipos de loja)."""
    return filtrar_dataframe(
        df,
        data_inicio,
        data_fim,
        tipos_loja=None,
        lojas_especificas=None,
        feriado_estadual=feriado_estadual,
        feriado_escolar=feriado_escolar,
    )

# Restante do código (filtrar_dataframe, etc.) mantém-se idêntico — copiado do arquivo anterior
# ... existing code ... 