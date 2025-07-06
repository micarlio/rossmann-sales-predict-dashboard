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

            # Filtra o histórico para mostrar apenas os últimos 2 meses antes da previsão
            data_inicio_historico = forecast['ds'].min() - pd.DateOffset(months=2)
            ts_display = ts[ts['ds'] >= data_inicio_historico]
            
            # Histórico - agora usando o dataframe 'ts_display' filtrado
            fig.add_trace(go.Scatter(x=ts_display['ds'], y=ts_display['y'], name='Histórico (2 Meses)', mode='lines', line=dict(color='#1f77b4')))
            
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
            
            # Formatar tabela
            df_table = pd.DataFrame({'Data': forecast['ds'], 'Previsão': forecast['yhat']})
            df_table['Data'] = df_table['Data'].dt.strftime('%d/%m/%Y')
            df_table['Previsão'] = df_table['Previsão'].round(2).apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            
            tabela_dash = dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True, responsive=True)
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
            # Montar miniblocos (3 por linha, 12 KPIs)
            items = [
                dbc.Col(html.Div([html.P("Total Previsto", className="text-muted"), html.H5(f"€ {total_prev:,.2f}", className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P("Média Prevista", className="text-muted"), html.H5(f"€ {media_prev:,.2f}", className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P("Variação vs. Hist. (%)", className="text-muted"), html.H5(f"{var_perc:,.2f}%", className=("fw-bold text-success" if var_perc>=0 else "fw-bold text-danger"))]), md=4),
                dbc.Col(html.Div([html.P(f"Pico ({pico_date})", className="text-muted"), html.H5(f"€ {pico_val:,.2f}", className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P(f"Vale ({vale_date})", className="text-muted"), html.H5(f"€ {vale_val:,.2f}", className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P("Ampl. IC Média", className="text-muted"), html.H5(f"€ {amp_ic:,.2f}", className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P("Variação Acumulada (%)", className="text-muted"), html.H5(f"{var_acum:,.2f}%", className=("fw-bold text-success" if var_acum>=0 else "fw-bold text-danger"))]), md=4),
                dbc.Col(html.Div([html.P("Períodos Previstos", className="text-muted"), html.H5(len(fc_y), className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P("Dias com Promoção Prevista", className="text-muted"), html.Div(html.Ul([html.Li(d) for d in promo_dates]), style={'maxHeight':'8rem','overflowY':'auto'})]), md=4),
                dbc.Col(html.Div([html.P("Maior Ganho (%)", className="text-muted"), html.H5(f"{max_gain:,.2f}%", className=("fw-bold text-success" if max_gain>=0 else "fw-bold text-danger"))]), md=4),
                dbc.Col(html.Div([html.P("Dia da Semana Top", className="text-muted"), html.H5(top_weekday, className="fw-bold")]), md=4),
                dbc.Col(html.Div([html.P("Desvio Padrão da Previsão", className="text-muted"), html.H5(f"€ {std_val:,.2f}", className="fw-bold")]), md=4),
            ]
            return [dbc.Row(items)]
        except Exception as e:
            return [html.Div(f"Erro ao gerar informações: {e}")]
    
    @aplicativo.callback(
        Output('otimizacao-estoque-conteudo', 'children'),
        [
            Input('dropdown-tipo-loja', 'value'),
            Input('dropdown-lojas-previsao', 'value'),
            Input('dropdown-promocao', 'value'),
            Input('checklist-dias-semana', 'value'),
            Input('armazenamento-df-principal', 'data')
        ],
        prevent_initial_call=True
    )
    def atualizar_otimizacao_estoque(tipo_loja, lojas, promocao, dias_semana, store_data):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Atualizando otimização de estoque")
        
        try:
            df = deserializar_df(store_data)
            if df is None: 
                logger.warning("DataFrame é None após deserialização")
                return html.P("Carregando dados...")

            # Usa a função de filtro centralizada
            df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, dias_semana)
            
            if df_filtrado.empty:
                logger.warning("DataFrame filtrado está vazio")
                return html.P("Sem dados para a otimização de estoque.")
                
            # Verificar se a coluna necessária existe
            if 'Sales' not in df_filtrado.columns:
                logger.error("Coluna Sales não encontrada no DataFrame")
                return html.P("Erro: Coluna Sales não encontrada nos dados")

            # Lógica simplificada de otimização de estoque
            venda_media_diaria = df_filtrado['Sales'].mean()
            estoque_seguranca = venda_media_diaria * 1.5
            ponto_ressuprimento = estoque_seguranca + (venda_media_diaria * 3)
            
            logger.info(f"Otimização calculada: Venda média={venda_media_diaria}, Estoque segurança={estoque_seguranca}")

            return [
                html.H5("Análise de Estoque Simplificada"),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardBody([
                        html.H6("Venda Média Diária", className="card-title"),
                        html.P(f"{venda_media_diaria:,.2f} unidades", className="card-text")
                    ])]), width=4),
                    dbc.Col(dbc.Card([dbc.CardBody([
                        html.H6("Estoque de Segurança", className="card-title"),
                        html.P(f"{estoque_seguranca:,.2f} unidades", className="card-text")
                    ])]), width=4),
                    dbc.Col(dbc.Card([dbc.CardBody([
                        html.H6("Ponto de Ressuprimento", className="card-title"),
                        html.P(f"{ponto_ressuprimento:,.2f} unidades", className="card-text")
                    ])]), width=4),
                ])
            ]
        except Exception as e:
            logger.error(f"Erro ao atualizar otimização de estoque: {str(e)}")
            return html.P(f"Erro ao calcular otimização de estoque: {str(e)}")

    @aplicativo.callback(
        Output('roi-marketing-output','children'),
        Input('btn-simular-campanha','n_clicks'),
        [
            State('dropdown-tipo-loja', 'value'),
            State('dropdown-lojas-previsao', 'value'),
            State('dropdown-promocao', 'value'),
            State('checklist-dias-semana', 'value'),
            State('input-custo-campanha','value'),
            State('input-periodo-campanha','value'),
            State('armazenamento-df-principal','data')
        ],
        prevent_initial_call=True
    )
    def simular_roi_marketing(n_clicks, tipo_loja, lojas, promocao, dias_semana, custo, periodo, store_data):
        import logging
        logger = logging.getLogger(__name__)
        
        if n_clicks == 0: 
            return ""
            
        logger.info(f"Simulando ROI de marketing. Custo: {custo}, Período: {periodo}")
        
        try:
            df = deserializar_df(store_data)
            if df is None: 
                logger.warning("DataFrame é None após deserialização")
                return dbc.Alert("Dados não carregados.", color="warning")

            # Usa a função de filtro centralizada
            df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, dias_semana)
            
            if df_filtrado.empty:
                logger.warning("DataFrame filtrado está vazio")
                return dbc.Alert("Não há dados de vendas para o período e filtros selecionados.", color="warning")
                
            # Verificar se a coluna necessária existe
            if 'Sales' not in df_filtrado.columns:
                logger.error("Coluna Sales não encontrada no DataFrame")
                return dbc.Alert("Erro: Coluna Sales não encontrada nos dados", color="danger")
                
            # Verificar se custo e período são válidos
            if custo is None or custo <= 0:
                logger.warning(f"Custo inválido: {custo}")
                return dbc.Alert("Por favor, insira um custo de campanha válido (maior que zero).", color="warning")
                
            if periodo is None or periodo <= 0:
                logger.warning(f"Período inválido: {periodo}")
                return dbc.Alert("Por favor, insira um período válido (maior que zero).", color="warning")

            vendas_base = df_filtrado['Sales'].sum()
            vendas_projetadas = vendas_base * 1.15
            lucro_incremental = (vendas_projetadas - vendas_base) * 0.1
            roi = ((lucro_incremental - custo) / custo) * 100 if custo and custo > 0 else 0
            
            logger.info(f"ROI calculado: {roi:.2f}%")

            return html.Div([
                html.H5("Resultado da Simulação de ROI"),
                html.P(f"Lucro Incremental Estimado: R$ {lucro_incremental:,.2f}"),
                html.P(f"ROI Estimado: {roi:.2f}%")
            ])
        except Exception as e:
            logger.error(f"Erro ao simular ROI de marketing: {str(e)}")
            return dbc.Alert(f"Erro ao calcular ROI: {str(e)}", color="danger")

    @aplicativo.callback(
        Output('modal-elasticidade','is_open'),
        Input('btn-open-modal-elasticidade','n_clicks'),
        Input('btn-close-modal-elasticidade','n_clicks'),
        State('modal-elasticidade','is_open')
    )
    def toggle_modal_elasticidade(n_open, n_close, is_open):
        if n_open or n_close:
            return not is_open
        return is_open

    @aplicativo.callback(
        Output('grafico-elasticidade-vendas','figure'),
        Output('grafico-elasticidade-receita','figure'),
        Input('btn-simular-preco','n_clicks'),
        [
            State('dropdown-tipo-loja', 'value'),
            State('dropdown-lojas-previsao', 'value'),
            State('dropdown-promocao', 'value'),
            State('checklist-dias-semana', 'value'),
            State('input-alteracao-preco','value'),
            State('armazenamento-df-principal','data')
        ],
        prevent_initial_call=True
    )
    def simular_elasticidade(n_clicks, tipo_loja, lojas, promocao, dias_semana, pct_change, store_data):
        import logging
        logger = logging.getLogger(__name__)
        
        if n_clicks == 0 or pct_change is None:
            return criar_figura_vazia("Aguardando simulação"), criar_figura_vazia("")
            
        logger.info(f"Simulando elasticidade de preço. Alteração: {pct_change}%")
        
        try:
            df = deserializar_df(store_data)
            if df is None:
                logger.warning("DataFrame é None após deserialização")
                return criar_figura_vazia("Carregando dados..."), criar_figura_vazia("")
            
            # Usa a função de filtro centralizada
            df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, dias_semana)
            
            if df_filtrado.empty:
                logger.warning("DataFrame filtrado está vazio")
                return criar_figura_vazia("Sem dados para simular"), criar_figura_vazia("")
                
            # Verificar se a coluna necessária existe
            if 'Sales' not in df_filtrado.columns:
                logger.error("Coluna Sales não encontrada no DataFrame")
                return criar_figura_vazia("Erro: Coluna Sales não encontrada"), criar_figura_vazia("")

            elasticidade_preco = -1.5
            variacao_vendas = elasticidade_preco * pct_change
            vendas_base = df_filtrado['Sales'].sum()
            vendas_simulada = vendas_base * (1 + variacao_vendas / 100)
            
            receita_base = df_filtrado['Sales'].sum() 
            receita_simulada = vendas_simulada * (1 + pct_change / 100)
            
            logger.info(f"Elasticidade calculada: Variação vendas={variacao_vendas:.2f}%, Receita simulada={receita_simulada:.2f}")

            fig_vendas = go.Figure(go.Indicator(
                mode = "number+delta", 
                value = vendas_simulada,
                delta = {'reference': vendas_base, 'relative': True, 'valueformat': '.1%'},
                title = {"text": "Vendas Totais"}
            ))
                
            fig_receita = go.Figure(go.Indicator(
                mode = "number+delta", 
                value = receita_simulada,
                delta = {'reference': receita_base, 'relative': True, 'valueformat': '.1%'},
                title = {"text": "Receita Total"}
            ))
                
            # Melhorar a aparência dos gráficos
            for fig in [fig_vendas, fig_receita]:
                fig.update_layout(
                    height=300,
                    margin=dict(l=30, r=30, t=50, b=30),
                    font=dict(size=14)
                )

            return fig_vendas, fig_receita
        except Exception as e:
            logger.error(f"Erro ao simular elasticidade: {str(e)}")
            return criar_figura_vazia(f"Erro: {str(e)}"), criar_figura_vazia("") 