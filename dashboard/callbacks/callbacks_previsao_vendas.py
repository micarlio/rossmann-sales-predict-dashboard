# dashboard/callbacks/callbacks_previsao_vendas.py

from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from io import StringIO
from prophet import Prophet
# Ajuste de performance: cache para Prophet até horizonte fixo
MAX_HORIZON = 30
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import functools
from sklearn.ensemble import RandomForestRegressor
import dash
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ..data_loader import get_principal_dataset, filtrar_por_data
from ..utils import criar_figura_vazia, parse_json_to_df

# Cache para modelos treinados
# A chave será uma tupla de identificadores únicos para um conjunto de dados de treinamento
MODEL_CACHE = {}

def deserializar_df(store_data):
    """
    Desserializa o DataFrame principal a partir do dcc.Store.
    Lida com diferentes formatos de dados e adiciona logs para depuração.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not store_data:
        logger.warning("Store data está vazio")
        return None
    
    # Se store_data for um dicionário (que atua como um sinal de carregamento),
    # usa a função parse_json_to_df para obter o DataFrame
    if isinstance(store_data, dict):
        logger.info(f"Recebido dict como store_data: {store_data}")
        return parse_json_to_df(store_data)

    # Se for uma string, é o dataframe em JSON.
    try:
        df = pd.read_json(StringIO(store_data), orient='split')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        logger.info(f"DataFrame deserializado com sucesso: {len(df)} linhas")
        return df
    except Exception as e:
        logger.error(f"Erro ao deserializar DataFrame: {str(e)}")
        return None


@functools.lru_cache(maxsize=8)
def gerar_forecast_prophet(df_ts_json, freq):
    """
    Helper para gerar forecast com Prophet usando cache de fit para horizonte máximo.
    """
    df_prophet = pd.read_json(StringIO(df_ts_json), orient='split')
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    m.fit(df_prophet)
    # Gera forecast até MAX_HORIZON, para fatiar depois conforme slider
    future = m.make_future_dataframe(periods=MAX_HORIZON, freq=freq)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].to_json(date_format='iso', orient='split')


def filtrar_dados(df, lojas, tipos_loja, promocao, dias_semana):
    """Aplica filtros de loja, tipo de loja, promoção e dias da semana."""
    df_filtrado = df.copy()
    # Prioriza seleção específica de lojas; se vazio, usa filtro de tipos
    if lojas:
        df_filtrado = df_filtrado[df_filtrado['Store'].isin(lojas)]
    elif tipos_loja:
        df_filtrado = df_filtrado[df_filtrado['StoreType'].isin(tipos_loja)]
    else:
        return pd.DataFrame()  # Sem filtros de loja selecionados
    # Filtro de promoção
    if promocao != 'todos':
        df_filtrado = df_filtrado[df_filtrado['Promo'] == promocao]
    # Filtro de dias da semana
    if dias_semana and 'todos' not in dias_semana:
        df_filtrado = df_filtrado[df_filtrado['DayOfWeek'].isin(dias_semana)]
    return df_filtrado


def registrar_callbacks_previsao_vendas(aplicativo, dados):
    """
    Registra callbacks para a página de Previsão de Vendas.
    """
    # Toggle collapse do painel de filtros
    @aplicativo.callback(
        Output('collapse-filtros', 'is_open'),
        Input('btn-toggle-filtros', 'n_clicks'),
        State('collapse-filtros', 'is_open')
    )
    def toggle_collapse_filtros(n, is_open):
        if n:
            return not is_open
        return is_open

    # Callback para popular dropdown de lojas
    @aplicativo.callback(
        Output('dropdown-lojas-previsao', 'options'),
        Output('dropdown-lojas-previsao', 'value'),
        Input('armazenamento-df-principal', 'data'),
        Input('dropdown-tipo-loja', 'value')
    )
    def popula_lojas(store_data, tipos_loja):
        """Popula opções de lojas com base nos tipos selecionados e inicia sem seleção."""
        df = deserializar_df(store_data)
        if df is None or 'Store' not in df.columns:
            return [], []
        if tipos_loja:
            df = df[df['StoreType'].isin(tipos_loja)]
        else:
            return [], []
        lojas = sorted(df['Store'].unique())
        options = [{'label': f"Loja {loja}", 'value': loja} for loja in lojas]
        return options, []

    # Callback para tornar "Todos os dias" mutuamente exclusivo
    @aplicativo.callback(
        Output('checklist-dias-semana', 'value'),
        Input('checklist-dias-semana', 'value'),
        prevent_initial_call=True
    )
    def atualizar_dias_semana(dias_selecionados):
        # Pega o último item selecionado
        trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if not dias_selecionados:
            return ['todos']

        # Se 'todos' foi selecionado, desmarque os outros
        if 'todos' in dias_selecionados and len(dias_selecionados) > 1:
            # Se o último selecionado foi 'todos'
            if dash.callback_context.triggered[0]['value'][-1] == 'todos':
                return ['todos']
            # Se um dia específico foi selecionado enquanto 'todos' estava marcado
            else:
                return [d for d in dias_selecionados if d != 'todos']
        
        return dias_selecionados

    # Toggle de exibição dos parâmetros avançados conforme modelo
    @aplicativo.callback(
        Output('parametros-arima','style'),
        Output('parametros-xgboost','style'),
        Output('parametros-lightgbm','style'),
        Output('parametros-ensemble','style'),
        Input('dropdown-modelo-previsao', 'value')
    )
    def toggle_params_modelo(modelo):
        hidden = {'display':'none'}
        arima_style = hidden.copy()
        xgb_style = hidden.copy()
        lgbm_style = hidden.copy()
        ens_style = hidden.copy()
        if modelo == 'arima':
            arima_style = {'display':'block','marginBottom':'20px'}
        elif modelo == 'xgboost':
            xgb_style = {'display':'block','marginBottom':'20px'}
        elif modelo == 'lightgbm':
            lgbm_style = {'display':'block','marginBottom':'20px'}
        elif modelo == 'ensemble':
            ens_style = {'display':'block','marginBottom':'20px'}
        return arima_style, xgb_style, lgbm_style, ens_style

    # Callback principal de previsão (REVERTIDO E REFATORADO)
    @aplicativo.callback(
        Output('grafico-previsao', 'figure'),
        Output('tabela-previsao', 'children'),
        [
            Input('radio-metrica-previsao', 'value'),
            Input('slider-horizonte-previsao', 'value'),
            Input('dropdown-granularidade-previsao', 'value'),
            Input('dropdown-modelo-previsao', 'value'),
            Input('dropdown-tipo-loja', 'value'),
            Input('dropdown-lojas-previsao', 'value'),
            Input('dropdown-promocao', 'value'),
            Input('checklist-dias-semana', 'value'),
            Input('armazenamento-df-principal', 'data')
        ],
        [
            State('arima-p','value'), State('arima-d','value'), State('arima-q','value'),
            State('xgb-estimators','value'), State('xgb-lr','value'), State('lgbm-estimators','value'),
        ]
    )
    def gerar_previsao(target, horizonte, granularidade, modelo,
                       tipo_loja, lojas, promocao, dias_semana, store_data,
                       p, d, q, xgb_estimators, xgb_lr, lgbm_estimators):
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Callback de previsão iniciado. Modelo: {modelo}, Tipos: {tipo_loja}, Lojas: {lojas}")
        
        df = deserializar_df(store_data)
        if df is None:
            logger.warning("DataFrame é None após deserialização")
            return criar_figura_vazia("Carregando dados..."), []
        
        # Uma verificação inicial para não mostrar nada se nenhum filtro de loja estiver definido
        if not lojas and not tipo_loja:
            logger.warning("Nenhum filtro de loja selecionado")
            return criar_figura_vazia("Selecione um tipo de loja ou loja específica"), []

        df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, dias_semana)
        
        if df_filtrado.empty:
            logger.warning("DataFrame filtrado está vazio")
            return criar_figura_vazia("Sem dados após aplicação dos filtros"), []

        # Seleção de métrica conforme escolha do usuário
        if target == 'Sales':
            col = 'Sales'
            metrica_nome = 'Vendas'
        elif target == 'Customers':
            col = 'Customers'
            metrica_nome = 'Clientes'
        elif target == 'SalesPerCustomer':
            # CORREÇÃO: Evita divisão por zero
            df_filtrado['TicketMedio'] = (df_filtrado['Sales'] / df_filtrado['Customers'].replace(0, np.nan)).fillna(0)
            col = 'TicketMedio'
            metrica_nome = 'Ticket Médio'
        else:
            col = 'Sales'
            metrica_nome = 'Vendas'
            
        # Verificar se a coluna existe no dataframe
        if col not in df_filtrado.columns:
            logger.error(f"Coluna {col} não encontrada no DataFrame")
            return criar_figura_vazia(f"Erro: Coluna {col} não encontrada nos dados"), []
            
        # Preparar série temporal conforme granularidade
        df_agrupado = df_filtrado.set_index('Date')
        
        try:
            if granularidade == 'diaria':
                freq = 'D'
                if target == 'SalesPerCustomer':
                    ts = df_agrupado.resample(freq).mean(numeric_only=True)[[col]].reset_index()
                else:
                    ts = df_agrupado.resample(freq).sum(numeric_only=True)[[col]].reset_index()
            elif granularidade == 'semanal':
                freq = 'W'
                if target == 'SalesPerCustomer':
                    ts = df_agrupado.resample(freq).mean(numeric_only=True)[[col]].reset_index()
                else:
                    ts = df_agrupado.resample(freq).sum(numeric_only=True)[[col]].reset_index()
            else:  # mensal
                freq = 'MS' # Alterado de 'M' para 'MS' (Month Start)
                if target == 'SalesPerCustomer':
                    ts = df_agrupado.resample(freq).mean(numeric_only=True)[[col]].reset_index()
                else:
                    ts = df_agrupado.resample(freq).sum(numeric_only=True)[[col]].reset_index()

            ts = ts.dropna()
            ts.rename(columns={col: 'y', 'Date': 'ds'}, inplace=True)
            
            if ts.empty or len(ts) < 10:
                logger.warning(f"Dados insuficientes para o modelo {modelo}")
                return criar_figura_vazia(f"Dados insuficientes para o modelo {modelo}"), []
                
            logger.info(f"Série temporal preparada com sucesso: {len(ts)} pontos")
            
            # Geração do DataFrame futuro movida e corrigida
            last_date = ts['ds'].max()
            if freq == 'D':
                offset = pd.DateOffset(days=1)
            elif freq == 'W':
                offset = pd.DateOffset(weeks=1)
            else: # MS
                offset = pd.DateOffset(months=1)
            
            future_dates = pd.date_range(start=last_date + offset, periods=horizonte, freq=freq)
            df_future = pd.DataFrame({'ds': future_dates})

        except Exception as e:
            logger.error(f"Erro ao preparar série temporal: {str(e)}")
            return criar_figura_vazia(f"Erro ao preparar dados: {str(e)}"), []
        
        # --- LÓGICA DE MODELAGEM ---
        # Construção de DataFrame de feriados para Prophet (conforme notebook)
        state_hols = df_filtrado[df_filtrado['StateHoliday'].astype(str) != '0'][['Date']].drop_duplicates().rename(columns={'Date':'ds'})
        state_hols['holiday'] = 'state_holiday'
        school_hols = df_filtrado[df_filtrado['SchoolHoliday'] == 1][['Date']].drop_duplicates().rename(columns={'Date':'ds'})
        school_hols['holiday'] = 'school_holiday'
        holidays = pd.concat([state_hols, school_hols])

        forecast = None
        model_name_display = ""

        try:
            # Previsão com Prophet
            if modelo == 'prophet':
                model_name_display = "Prophet"
                logger.info("Iniciando modelagem com Prophet")
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=holidays)
                m.fit(ts)
                forecast = m.predict(df_future)
                forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            # Previsão com ARIMA
            elif modelo == 'arima':
                model_name_display = "ARIMA"
                logger.info(f"Iniciando modelagem com ARIMA(p={p}, d={d}, q={q})")
                try:
                    model_arima = ARIMA(ts['y'], order=(int(p), int(d), int(q)), index=ts['ds']).fit()
                    pred = model_arima.get_forecast(steps=horizonte)
                    forecast = pd.DataFrame({
                        'ds': future_dates,
                        'yhat': pred.predicted_mean.values,
                        'yhat_lower': pred.conf_int().iloc[:, 0].values,
                        'yhat_upper': pred.conf_int().iloc[:, 1].values
                    })
                except Exception as e:
                    logger.error(f"Erro específico do ARIMA: {str(e)}")
                    return criar_figura_vazia(f"Erro ARIMA: {e}"), []
            
            # Previsão com Modelos de ML
            elif modelo in ['random_forest', 'xgboost', 'lightgbm']:
                # Engenharia de Features
                def criar_features(df):
                    df_feat = df.copy()
                    df_feat['mes'] = df_feat['ds'].dt.month
                    df_feat['dia_semana'] = df_feat['ds'].dt.dayofweek
                    df_feat['dia_mes'] = df_feat['ds'].dt.day
                    df_feat['semana_ano'] = df_feat['ds'].dt.isocalendar().week.astype(int)
                    return df_feat

                logger.info(f"Iniciando modelagem com {modelo}")
                ts_featured = criar_features(ts)
                df_future_featured = criar_features(df_future)
                
                features = ['mes', 'dia_semana', 'dia_mes', 'semana_ano']
                X_train, y_train = ts_featured[features], ts_featured['y']
                X_future = df_future_featured[features]

                if modelo == 'random_forest':
                    model_name_display = "Random Forest"
                    model_ml = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                elif modelo == 'xgboost':
                    model_name_display = "XGBoost"
                    model_ml = XGBRegressor(n_estimators=xgb_estimators, learning_rate=xgb_lr, random_state=42, n_jobs=-1)
                elif modelo == 'lightgbm':
                    model_name_display = "LightGBM"
                    model_ml = LGBMRegressor(n_estimators=lgbm_estimators, random_state=42, n_jobs=-1)

                model_ml.fit(X_train, y_train)
                pred = model_ml.predict(X_future)
                # Para modelos de ML, o intervalo de confiança não é trivial, então usamos apenas a predição
                forecast = pd.DataFrame({'ds': future_dates, 'yhat': pred, 'yhat_lower': pred, 'yhat_upper': pred})

            # Previsão com Ensemble (Média Ponderada Prophet + LightGBM)
            elif modelo == 'ensemble':
                model_name_display = "Ensemble (Prophet + LGBM)"
                logger.info("Iniciando modelagem com Ensemble")
                m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, holidays=holidays).fit(ts)
                forecast_prophet = m_prophet.predict(df_future)
                
                # LightGBM
                def criar_features(df):
                    df_feat = df.copy()
                    df_feat['mes'] = df_feat['ds'].dt.month
                    df_feat['dia_semana'] = df_feat['ds'].dt.dayofweek
                    df_feat['dia_mes'] = df_feat['ds'].dt.day
                    df_feat['semana_ano'] = df_feat['ds'].dt.isocalendar().week.astype(int)
                    return df_feat
                
                ts_featured = criar_features(ts)
                df_future_featured = criar_features(df_future)
                features = ['mes', 'dia_semana', 'dia_mes', 'semana_ano']
                X_train, y_train = ts_featured[features], ts_featured['y']
                X_future = df_future_featured[features]
                
                model_lgbm = LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
                pred_lgbm = model_lgbm.predict(X_future)
                
                # Ensemble (média simples)
                pred_ensemble = (forecast_prophet['yhat'].values + pred_lgbm) / 2
                
                forecast = pd.DataFrame({'ds': future_dates, 'yhat': pred_ensemble})
                # Pegando o IC do prophet como referência
                forecast['yhat_lower'] = forecast_prophet['yhat_lower']
                forecast['yhat_upper'] = forecast_prophet['yhat_upper']

            if forecast is None:
                logger.error(f"Modelo {modelo} não implementado ou erro")
                return criar_figura_vazia(f"Modelo {modelo} não implementado ou erro"), []
                
            logger.info(f"Previsão gerada com sucesso: {len(forecast)} pontos")
                
        except Exception as e:
            logger.error(f"Erro ao gerar previsão: {str(e)}")
            return criar_figura_vazia(f"Erro ao gerar previsão: {str(e)}"), []
            
        # --- MONTAGEM DO GRÁFICO E TABELA ---
        try:
            fig = go.Figure()

            # Adiciona TODO o histórico e habilita range-slider para seleção dinâmica
            fig.add_trace(go.Scatter(x=ts['ds'], y=ts['y'], name='Histórico', mode='lines', line=dict(color='#1f77b4')))

            # Faixa padrão de visualização: últimos 2 meses antes da previsão
            data_inicio_visual = ts['ds'].max() - pd.DateOffset(months=2)
            data_fim_visual = forecast['ds'].max()

            # Ativa range-slider e define o intervalo inicial exibido
            fig.update_layout(xaxis_rangeslider=dict(visible=True))
            fig.update_xaxes(range=[data_inicio_visual, data_fim_visual])
            
            # Previsão
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão', mode='lines', line=dict(color='#ff7f0e')))
            # Intervalo de Confiança (se disponível)
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', 
                    line=dict(color='#ff7f0e', width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', 
                    fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='#ff7f0e', width=0),
                    name='Intervalo de Confiança'
                ))

            fig.update_layout(
                title=f'Previsão de {metrica_nome} com {model_name_display} ({granularidade.capitalize()})',
                xaxis_title='Data', 
                yaxis_title=metrica_nome,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # ---- Montagem da tabela com coluna de variação ----
            df_tab = pd.DataFrame({
                'Data': forecast['ds'],
                'Previsão': forecast['yhat']
            })
            # Formatação
            df_tab['Data_fmt'] = df_tab['Data'].dt.strftime('%d/%m/%Y')
            df_tab['Prev_fmt'] = df_tab['Previsão'].round(2).apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            # Cálculo da variação percentual dia-a-dia
            df_tab['Var_pct'] = df_tab['Previsão'].pct_change() * 100

            # Cabeçalho
            table_header = html.Thead(html.Tr([
                html.Th("Data", className="text-center"),
                html.Th("Previsão", className="text-center"),
                html.Th("Variação (%)", className="text-center")
            ]))

            # Linhas
            table_rows = []
            for _, r in df_tab.iterrows():
                # Variação formatada
                if pd.isna(r['Var_pct']):
                    var_cell = html.Span("-", className="text-muted")
                else:
                    sinal_classe = "text-success" if r['Var_pct'] >= 0 else "text-danger"
                    var_cell = html.Span(f"{r['Var_pct']:.2f}%", className=f"fw-bold {sinal_classe}")
                table_rows.append(html.Tr([
                    html.Td(r['Data_fmt']),
                    html.Td(r['Prev_fmt']),
                    html.Td(var_cell, className="text-center")
                ]))

            tabela_dash = dbc.Table(
                [table_header, html.Tbody(table_rows)],
                striped=True,
                bordered=False,
                hover=True,
                responsive=True,
                class_name="table-custom table-striped"
            )
            logger.info("Gráfico e tabela gerados com sucesso")
            return fig, tabela_dash
            
        except Exception as e:
            logger.error(f"Erro ao montar gráfico ou tabela: {str(e)}")
            return criar_figura_vazia(f"Erro ao montar visualização: {str(e)}"), []

    @aplicativo.callback(
        Output('informacoes-previsao', 'children'),
        Input('grafico-previsao', 'figure')
    )
    def atualizar_informacoes_previsao(fig):
        import pandas as pd
        import numpy as np
        if not fig or 'data' not in fig:
            return [html.Div("Sem dados de previsão para exibir.")]
        try:
            hist_y = np.array(fig['data'][0]['y'], dtype=float)
            fc = fig['data'][1]
            fc_dates = pd.to_datetime(fc['x'])
            fc_y = np.array(fc['y'], dtype=float)
            # Cálculos principais
            total_prev = fc_y.sum()
            media_prev = fc_y.mean()
            media_hist = hist_y.mean() if hist_y.size > 0 else np.nan
            var_perc = (media_prev / media_hist - 1) * 100 if media_hist else 0
            idx_max = int(np.nanargmax(fc_y))
            idx_min = int(np.nanargmin(fc_y))
            pico_date = fc_dates[idx_max].strftime('%d/%m/%Y')
            pico_val = fc_y[idx_max]
            vale_date = fc_dates[idx_min].strftime('%d/%m/%Y')
            vale_val = fc_y[idx_min]
            # Intervalo de confiança
            if len(fig['data']) >= 4:
                lower = np.array(fig['data'][2]['y'], dtype=float)
                upper = np.array(fig['data'][3]['y'], dtype=float)
                amp_ic = np.mean(upper - lower)
            else:
                amp_ic = np.nan
            # Variação acumulada e desvio padrão
            var_acum = (fc_y[-1] / fc_y[0] - 1) * 100 if fc_y[0] else 0
            std_val = np.std(fc_y)
            # Maior ganho percentual dia-a-dia
            pct_diff = (np.diff(fc_y) / fc_y[:-1] * 100) if fc_y.size > 1 else np.array([0])
            max_gain = pct_diff.max()
            # Dia da semana com maior previsão
            df_fc = pd.DataFrame({'ds': fc_dates, 'y': fc_y})
            df_fc['weekday'] = df_fc['ds'].dt.weekday
            dias_media = df_fc.groupby('weekday')['y'].mean()
            mapping = {0: 'Segunda-feira', 1: 'Terça-feira', 2: 'Quarta-feira', 3: 'Quinta-feira', 4: 'Sexta-feira', 5: 'Sábado', 6: 'Domingo'}
            top_weekday = mapping.get(int(dias_media.idxmax()), '') if not dias_media.empty else ''
            # Dias com promoção prevista (lista de datas)
            promo_dates = fc_dates.strftime('%d/%m/%Y')
            # Montar miniblocos (agora em linhas de 3 colunas para melhor simetria)
            items = [
                ("Total Previsto", f"€ {total_prev:,.0f}"),
                ("Média Prevista", f"€ {media_prev:,.0f}"),
                ("Variação vs. Hist. (%)", f"{var_perc:,.2f}%", "text-success" if var_perc>=0 else "text-danger"),
                ("Variação Acumulada (%)", f"{var_acum:,.2f}%", "text-success" if var_acum>=0 else "text-danger"),
                ("Dia da Semana Top", top_weekday),
                ("Dias com Promoção Prevista", html.Ul([html.Li(d) for d in promo_dates], style={'maxHeight':'3rem','overflowY':'auto'})),
            ]

            items_divs = []
            for titulo, valor, *color in items:
                estilo_valor = "fw-bold " + (color[0] if color else "")
                # Se o valor já é um componente (ex.: html.Ul), não o envolvemos em html.H5
                if isinstance(valor, (html.Ul, html.Ol, html.Div)):
                    valor_component = valor
                else:
                    valor_component = html.H5(valor, className=estilo_valor)

                items_divs.append(
                    html.Div([
                        html.P(titulo, className="text-muted mb-1 small"),
                        valor_component
                    ], className="info-item")
                )

            return items_divs
        except Exception as e:
            return [html.Div(f"Erro ao gerar informações: {e}")]
    
    # ==================================================================
    # NOVOS GRÁFICOS (gerados a partir do grafico-previsao já existente)
    # ==================================================================

    @aplicativo.callback(
        Output('grafico-empilhado', 'figure'),
        Output('heatmap-calendario', 'figure'),
        Input('grafico-previsao', 'figure')
    )
    def gerar_graficos_adicionais(fig):
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        import plotly.express as px

        # Se gráfico não está pronto ainda
        if not fig or 'data' not in fig or len(fig['data']) < 2:
            fig_placeholder = go.Figure()
            fig_placeholder.update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False,
                                          annotations=[dict(text="Aguardando geração da previsão", showarrow=False)])
            return [fig_placeholder]*2

        # --- Extrai dados ---
        hist_x = pd.to_datetime(fig['data'][0]['x'])
        hist_y = np.array(fig['data'][0]['y'], dtype=float)

        fc_x = pd.to_datetime(fig['data'][1]['x'])
        fc_y = np.array(fig['data'][1]['y'], dtype=float)

        # ======================= 1. Acumulado =========================
        # hist_cum = np.cumsum(hist_y)
        # fc_cum = hist_cum[-1] + np.cumsum(fc_y)
        #
        # fig_acum = go.Figure()
        # fig_acum.add_trace(go.Scatter(x=hist_x, y=hist_cum, mode='lines', name='Real Acumulado', line=dict(color='#1f77b4')))
        # fig_acum.add_trace(go.Scatter(x=fc_x, y=fc_cum, mode='lines', name='Previsto Acumulado', line=dict(color='#ff7f0e')))
        # fig_acum.update_layout(template='plotly_white', showlegend=True, xaxis_title='Data', yaxis_title='Vendas Acumuladas')

        # ======================= 2. Amplitude IC ======================
        # if len(fig['data']) >= 4:
        #     upper = np.array(fig['data'][2]['y'], dtype=float)
        #     lower = np.array(fig['data'][3]['y'], dtype=float)
        #     amplitude = upper - lower
        #     fig_ic = go.Figure(go.Bar(x=fc_x, y=amplitude, marker_color='#2ca02c'))
        #     fig_ic.update_layout(template='plotly_white', xaxis_title='Data', yaxis_title='Amplitude do IC')
        # else:
        #     fig_ic = go.Figure()
        #     fig_ic.update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False,
        #                          annotations=[dict(text="IC não disponível", showarrow=False)])

        # ======================= 3. Distribuição por Dia da Semana ====
        df_fc = pd.DataFrame({'ds': fc_x, 'y': fc_y})
        df_fc['weekday'] = df_fc['ds'].dt.weekday
        mapping = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 4: 'Sex', 5: 'Sáb', 6: 'Dom'}
        distrib = df_fc.groupby('weekday')['y'].sum().reset_index()
        distrib['weekday_name'] = distrib['weekday'].map(mapping)
        fig_dist = go.Figure(go.Bar(x=distrib['weekday_name'], y=distrib['y'], marker_color='#1f77b4'))
        fig_dist.update_layout(template='plotly_white', xaxis_title='Dia da Semana', yaxis_title='Total Previsto')

        # ======================= 4. Heatmap Calendário ===============
        df_fc['week'] = df_fc['ds'].dt.isocalendar().week.astype(int)
        pivot = df_fc.pivot_table(index='weekday', columns='week', values='y', aggfunc='sum')
        # Ordena dias da semana 0-6
        pivot = pivot.reindex(range(0,7))
        heatmap = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=[mapping.get(i,'') for i in pivot.index],
                                           colorscale='Blues'))
        heatmap.update_layout(template='plotly_white', xaxis_title='Semana do Ano', yaxis_title='Dia da Semana')

        # ======================= 5. Top Picos =========================
        # top_n = 10 if len(fc_y) > 10 else len(fc_y)
        # ind_top = np.argsort(fc_y)[-top_n:][::-1]
        # top_dates = fc_x[ind_top]
        # top_vals = fc_y[ind_top]
        # fig_top = go.Figure(go.Bar(x=top_dates.strftime('%d/%m/%Y'), y=top_vals, marker_color='#d62728'))
        # fig_top.update_layout(template='plotly_white', xaxis_title='Data', yaxis_title='Previsão',
        #                       xaxis_tickangle=-45)

        return fig_dist, heatmap

    # ==================================================================
    # SIMULADOR WHAT-IF
    # ==================================================================

    @aplicativo.callback(
        Output('grafico-whatif', 'figure'),
        Output('kpi-total-base', 'children'),
        Output('kpi-total-sim', 'children'),
        Input('btn-simular-whatif', 'n_clicks'),
        State('slider-whatif-preco', 'value'),
        State('slider-whatif-promo', 'value'),
        State('grafico-previsao', 'figure'),
        prevent_initial_call=True
    )
    def simular_whatif(n_clicks, delta_preco_pct, delta_promo_pp, fig_previsao):
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        # Parâmetros de elasticidade (hipotéticos, podem vir de arquivo JSON)
        ELASTIC_PRICE = -1.2   # cada -1% preço -> +1.2% vendas
        ELASTIC_PROMO = 0.8    # cada +1 p.p. promo -> +0.8% vendas

        if not fig_previsao or 'data' not in fig_previsao or len(fig_previsao['data']) < 2:
            fig_placeholder = go.Figure()
            fig_placeholder.update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False,
                                          annotations=[dict(text="Gere a previsão primeiro", showarrow=False)])
            return fig_placeholder, "-", "-"

        # Dados base
        fc_x = pd.to_datetime(fig_previsao['data'][1]['x'])
        fc_y = np.array(fig_previsao['data'][1]['y'], dtype=float)

        # Cálculo do fator de ajuste
        fator_preco = 1 + (ELASTIC_PRICE * (delta_preco_pct / 100))
        fator_promo = 1 + (ELASTIC_PROMO * (delta_promo_pp / 100))
        ajuste = fator_preco * fator_promo

        y_sim = fc_y * ajuste

        # Gráfico comparativo
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=fc_x, y=fc_y, name='Base', mode='lines', line=dict(color='#1f77b4')))
        fig_sim.add_trace(go.Scatter(x=fc_x, y=y_sim, name='Cenário', mode='lines', line=dict(color='#ff7f0e')))
        fig_sim.update_layout(template='plotly_white', xaxis_title='Data', yaxis_title='Vendas Previstas')

        total_base = fc_y.sum()
        total_sim = y_sim.sum()

        def format_currency(v):
            return f"€ {v:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')

        return fig_sim, format_currency(total_base), format_currency(total_sim) 

    # ==================================================================
    # GRÁFICO COMPARATIVO DE MÉTRICAS PREVISTAS (Sales / Customers / Ticket)
    # ==================================================================
    
    @aplicativo.callback(
        Output('grafico-comparativo', 'figure'),
        [
            Input('slider-horizonte-previsao', 'value'),
            Input('dropdown-granularidade-previsao', 'value'),
            Input('dropdown-modelo-previsao', 'value'),
            Input('dropdown-tipo-loja', 'value'),
            Input('dropdown-lojas-previsao', 'value'),
            Input('dropdown-promocao', 'value'),
            Input('checklist-dias-semana', 'value'),
            Input('armazenamento-df-principal', 'data')
        ]
    )
    def gerar_comparativo_metricas(horizonte, granularidade, modelo, tipo_loja, lojas, promocao, dias_semana, store_data):
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from prophet import Prophet
        from statsmodels.tsa.arima.model import ARIMA
        # Preparação de dados compartilhada
        df = deserializar_df(store_data)
        if df is None or df.empty:
            return go.Figure()
        df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, dias_semana)
        if df_filtrado.empty:
            return go.Figure()

        fig = go.Figure()
        metrica_info = {
            'Sales': {'col':'Sales','nome':'Vendas','cor':'#1f77b4'},
            'Customers': {'col':'Customers','nome':'Clientes','cor':'#2ca02c'},
            'SalesPerCustomer': {'col':'TicketMedio','nome':'Ticket Médio','cor':'#ff7f0e'}
        }

        for key, info in metrica_info.items():
            df_work = df_filtrado.copy()
            if key == 'SalesPerCustomer':
                df_work['TicketMedio'] = (df_work['Sales'] / df_work['Customers'].replace(0, np.nan)).fillna(0)

            col = info['col']
            df_agr = df_work.set_index('Date')
            freq_map = {'diaria':'D','semanal':'W','mensal':'MS'}
            freq = freq_map.get(granularidade, 'D')
            if key == 'SalesPerCustomer':
                ts = df_agr.resample(freq).mean(numeric_only=True)[[col]].reset_index()
            else:
                ts = df_agr.resample(freq).sum(numeric_only=True)[[col]].reset_index()
            ts = ts.dropna()
            ts.rename(columns={col:'y','Date':'ds'}, inplace=True)
            if ts.empty or len(ts)<10:
                continue

            # Forecast simples com Prophet (para tempo)
            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                m.fit(ts)
                last_date = ts['ds'].max()
                offset = pd.DateOffset(days=1) if freq=='D' else (pd.DateOffset(weeks=1) if freq=='W' else pd.DateOffset(months=1))
                future_dates = pd.date_range(start=last_date+offset, periods=horizonte, freq=freq)
                forecast = m.predict(pd.DataFrame({'ds':future_dates}))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name=info['nome'], line=dict(color=info['cor'])))
            except Exception:
                continue

        fig.update_layout(template='plotly_white', xaxis_title='Data', yaxis_title='Valor Previsto')
        return fig 

    # ==================================================================
    # PREVISÃO POR TIPO DE LOJA
    # ==================================================================

    @aplicativo.callback(
        Output('grafico-tipoloja', 'figure'),
        Output('tabela-tipoloja', 'children'),
        [
            Input('radio-metrica-previsao', 'value'),
            Input('slider-horizonte-previsao', 'value'),
            Input('dropdown-granularidade-previsao', 'value'),
            Input('dropdown-tipo-loja', 'value'),
            Input('dropdown-lojas-previsao', 'value'),
            Input('dropdown-promocao', 'value'),
            Input('checklist-dias-semana', 'value'),
            Input('armazenamento-df-principal', 'data')
        ]
    )
    def previsao_por_tipoloja(target, horizonte, granularidade, tipos_loja, lojas, promocao, dias_semana, store_data):
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from prophet import Prophet

        df = deserializar_df(store_data)
        if df is None or df.empty or not tipos_loja:
            return go.Figure(), []

        freq_map = {'diaria':'D','semanal':'W','mensal':'MS'}
        freq = freq_map.get(granularidade, 'D')

        fig = go.Figure()
        resumo = []

        # Determine column for metric
        metrica_nome = ''
        if target == 'Sales':
            col_base = 'Sales'; metrica_nome='Vendas'
        elif target == 'Customers':
            col_base = 'Customers'; metrica_nome='Clientes'
        else:
            col_base = 'TicketMedio'; metrica_nome='Ticket Médio'

        for tipo in tipos_loja:
            df_tipo = df[df['StoreType'] == tipo]
            df_tipo = filtrar_dados(df_tipo, lojas, [tipo], promocao, dias_semana)
            if df_tipo.empty: continue
            if target == 'SalesPerCustomer':
                df_tipo['TicketMedio'] = (df_tipo['Sales'] / df_tipo['Customers'].replace(0, np.nan)).fillna(0)

            df_agr = df_tipo.set_index('Date')
            if target == 'SalesPerCustomer':
                ts = df_agr.resample(freq).mean(numeric_only=True)[[col_base]].reset_index()
            else:
                ts = df_agr.resample(freq).sum(numeric_only=True)[[col_base]].reset_index()
            ts = ts.dropna()
            ts.rename(columns={'Date':'ds', col_base:'y'}, inplace=True)
            if ts.empty or len(ts)<10: continue

            try:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                m.fit(ts)
                last_date = ts['ds'].max()
                offset = pd.DateOffset(days=1) if freq=='D' else (pd.DateOffset(weeks=1) if freq=='W' else pd.DateOffset(months=1))
                future_dates = pd.date_range(start=last_date+offset, periods=horizonte, freq=freq)
                forecast = m.predict(pd.DataFrame({'ds':future_dates}))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name=f"Tipo {tipo.upper()}"))

                total_prev = forecast['yhat'].sum()
                resumo.append({'Tipo': tipo.upper(), 'Total Previsto': total_prev})
            except Exception:
                continue

        # Tabela resumo
        if resumo:
            df_resumo = pd.DataFrame(resumo)
            total_geral = df_resumo['Total Previsto'].sum()
            df_resumo['% Part'] = (df_resumo['Total Previsto']/total_geral*100).round(2)
            df_resumo['Total Previsto'] = df_resumo['Total Previsto'].round(2).apply(lambda x: f"{x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
            tabela = dbc.Table.from_dataframe(df_resumo, striped=True, bordered=False, hover=True, class_name="table-custom")
        else:
            tabela = html.P("Sem dados para exibir")

        fig.update_layout(template='plotly_white', xaxis_title='Data', yaxis_title=metrica_nome)
        return fig, tabela 