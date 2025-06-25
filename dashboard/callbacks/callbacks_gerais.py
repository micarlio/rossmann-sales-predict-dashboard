# dashboard/callbacks/callbacks_gerais.py
from dash import Input, Output, html, dcc, State, callback, ctx
import dash
import pandas as pd

from ..config import AZUL_DESTAQUE, VERDE_DESTAQUE, DESCRICOES_COLUNAS # Importar DESCRICOES_COLUNAS
from ..utils import filtrar_dataframe

def registrar_callbacks_gerais(aplicativo, dados):
    df_principal = dados["df_principal"]

    # --- Callback para o estado da Barra Lateral ---
    @aplicativo.callback(
        [
            Output('barra-lateral', 'className'),
            Output('conteudo-pagina', 'className'),
            Output('armazenamento-estado-barra-lateral', 'data'),
            Output('icone-alternar-barra-lateral', 'className'),
            Output('img-logo-barra-lateral', 'src')
        ],
        [
            Input('alternar-barra-lateral', 'n_clicks'),
        ],
        [
            State('armazenamento-estado-barra-lateral', 'data')
        ]
    )
    def alternar_sidebar(n_clicks, estado_atual):
        """
        Alterna a classe 'collapsed' para a barra lateral e para o contêiner de conteúdo,
        controlando a visibilidade da barra lateral. Também salva o estado e troca o logo.
        """

        if n_clicks is None:
            estado_aplicar = estado_atual or 'expanded'
        else:
            estado_aplicar = 'collapsed' if estado_atual == 'expanded' else 'expanded'

        if estado_aplicar == 'collapsed':
            classe_barra_lateral = 'sidebar d-flex flex-column collapsed'
            classe_conteudo = 'content collapsed'
            classe_icone = 'fas fa-angle-double-right'
            src_logo = '/assets/images/rossmann_logo1.png'
        else: # 'expanded'
            classe_barra_lateral = 'sidebar d-flex flex-column'
            classe_conteudo = 'content'
            classe_icone = 'fas fa-angle-double-left'
            src_logo = '/assets/images/rossmann_logo2.png'

        return classe_barra_lateral, classe_conteudo, estado_aplicar, classe_icone, src_logo

    # --- Callback Principal para Navegação entre Páginas (Modo Pré-carregado) ---
    PAGINAS = [
        "/",
        "/limpeza-dados",
        "/analise-preliminar",
        "/dashboard",
        "/analise-lojas",
        "/analise-3d",
        "/previsao-vendas",
    ]

    @aplicativo.callback(
        [Output(f'conteudo-pagina-{pagina}', 'style') for pagina in PAGINAS],
        [Input('url', 'pathname')]
    )
    def renderizar_conteudo_pagina(caminho_pagina):
        """
        Alterna a visibilidade dos contêineres de página com base na URL.
        Em vez de recriar o layout, apenas altera a propriedade CSS 'display'.
        """
        estilos = [{'display': 'none'}] * len(PAGINAS)

        try:
            idx = PAGINAS.index(caminho_pagina) if caminho_pagina in PAGINAS else 0
        except ValueError:
            idx = 0

        estilos[idx] = {'display': 'block'}

        return estilos

    # --- Callback para Tela Cheia (Clientside) ---
    aplicativo.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks) {
                if (!document.fullscreenElement) {
                    document.documentElement.requestFullscreen();
                } else {
                    if (document.exitFullscreen) {
                        document.exitFullscreen();
                    }
                }
            }
            return dash_clientside.no_update;
        }
        """,
        Output({'type': 'saida-tela-cheia', 'index': dash.dependencies.MATCH}, 'data'),
        Input({'type': 'botao-tela-cheia', 'index': dash.dependencies.MATCH}, 'n_clicks'),
        prevent_initial_call=True
    )

    # --- Callback Combinado para Validação de Datas ---
    @aplicativo.callback(
        Output('filtro-data', 'end_date', allow_duplicate=True),
        Input('filtro-data', 'start_date'),
        Input('filtro-data', 'end_date'),
        prevent_initial_call=True
    )
    def validar_datas(data_inicio, data_fim):
        contexto = ctx
        if not contexto.triggered:
            return dash.no_update

        id_gatilho = contexto.triggered[0]['prop_id'].split('.')[0]

        if id_gatilho == 'filtro-data':
            if data_inicio:
                data_inicio_dt = pd.to_datetime(data_inicio)
                if data_inicio_dt > pd.to_datetime(df_principal['Date'].max()):
                    return df_principal['Date'].max().date()
                if data_fim and data_inicio_dt > pd.to_datetime(data_fim):
                    return data_inicio
        return dash.no_update

    # --- Callback para Atualizar Opções de Lojas Específicas ---
    @aplicativo.callback(
        Output('filtro-loja-especifica', 'options', allow_duplicate=True),
        Output('filtro-loja-especifica', 'value', allow_duplicate=True), # Adicionado para limpar a seleção
        Input('filtro-tipo-loja', 'value'),
        prevent_initial_call=True
    )
    def atualizar_opcoes_lojas(tipos_loja_selecionados):
        if not tipos_loja_selecionados:
            return [], []

        lojas_filtradas = df_principal[df_principal['StoreType'].isin(tipos_loja_selecionados)]['Store'].unique()
        opcoes = [{'label': str(s), 'value': s} for s in sorted(lojas_filtradas)]
        return opcoes, [] # Limpa a seleção atual ao mudar os tipos de loja

    # --- Callback para Resetar Filtros ---
    @aplicativo.callback(
        [
            Output('dashboard-filtro-data', 'start_date'),
            Output('dashboard-filtro-data', 'end_date'),
            Output('dashboard-filtro-tipo-loja', 'value'),
            Output('dashboard-filtro-loja-especifica', 'value'),
            Output('filtro-granularidade', 'value'), # Permanece sem prefixo, controle de granularidade
            Output('dashboard-filtro-metrica-temporal', 'value'),
            Output('dashboard-filtro-feriado-estadual', 'value'),
            Output('dashboard-filtro-feriado-escolar', 'value'),
            # Adicionado para resetar os seletores de métrica dos gráficos de comportamento
            Output('seletor-metrica-comportamento-promocao', 'value'),
            Output('seletor-metrica-sortimento', 'value')
        ],
        Input('dashboard-botao-resetar-filtros', 'n_clicks'),
        prevent_initial_call=True
    )
    def resetar_filtros(n_clicks):
        if n_clicks:
            return (
                df_principal['Date'].min().date(),
                df_principal['Date'].max().date(),
                sorted(df_principal['StoreType'].unique()),
                [], # Resetar seleção de lojas específicas
                'M', # Permanece sem prefixo
                'Sales', # Valor padrão para métrica temporal
                'all', # Valor padrão para feriado estadual
                'all', # Valor padrão para feriado escolar
                'SalesPerCustomer', # Valor padrão para comportamento de promoção
                'SalesPerCustomer'  # Valor padrão para comportamento de sortimento
            )
        return dash.no_update

    # --- Callback para Download dos Dados Filtrados ---
    @aplicativo.callback(
        Output("download-df-parquet", "data"),
        Input("dashboard-botao-baixar-dados-filtrados", "n_clicks"),
        [
            dash.State('dashboard-filtro-data', 'start_date'),
            dash.State('dashboard-filtro-data', 'end_date'),
            dash.State('dashboard-filtro-tipo-loja', 'value'),
            dash.State('dashboard-filtro-loja-especifica', 'value'),
            dash.State('dashboard-filtro-feriado-estadual', 'value'),
            dash.State('dashboard-filtro-feriado-escolar', 'value')
        ],
        prevent_initial_call=True,
    )
    def baixar_dados_filtrados(n_clicks, data_inicio, data_fim, tipos_loja_selecionados, lojas_especificas_selecionadas, feriado_estadual_selecionado, feriado_escolar_selecionado):
        if n_clicks:
            df_download = filtrar_dataframe(df_principal, data_inicio, data_fim, tipos_loja_selecionados, lojas_especificas_selecionadas, feriado_estadual_selecionado, feriado_escolar_selecionado)
            return dcc.send_bytes(
                lambda buf: df_download.to_parquet(buf, index=False),
                f"rossmann_dados_filtrados_{data_inicio}_a_{data_fim}.parquet"
            )
        return dash.no_update

    @aplicativo.callback(
        Output('saida-descricao-coluna', 'children'),
        Input('dropdown-descricao-coluna', 'value')
    )
    def atualizar_descricao_coluna(coluna_selecionada):
        # A importação do DESCRICOES_COLUNAS foi movida para o topo deste arquivo
        # para evitar importações cíclicas se DESCRICOES_COLUNAS for usado em config.py
        # onde utils.py também é importado.
        
        if not coluna_selecionada:
            return html.P("Selecione uma coluna para ver sua descrição.", className="text-muted")

        descricao = DESCRICOES_COLUNAS.get(coluna_selecionada, "Descrição não disponível.")

        colunas_loja = ['StoreType', 'Assortment', 'CompetitionDistance',
                        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                        'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        eh_coluna_loja = coluna_selecionada in colunas_loja

        estilo_indicador = {
            'backgroundColor': VERDE_DESTAQUE if eh_coluna_loja else AZUL_DESTAQUE,
            'color': 'white',
            'padding': '2px 8px',
            'borderRadius': '4px',
            'fontSize': '0.8rem',
            'marginLeft': '10px',
            'display': 'inline-block'
        }

        return html.Div([
            html.Div([
                html.H6(
                    [
                        coluna_selecionada,
                        html.Span(
                            'store_df' if eh_coluna_loja else 'train_df',
                            style=estilo_indicador
                        )
                    ],
                    className="mb-2",
                    style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}
                ),
            ]),
            html.P(descricao, className="mb-0")
        ])