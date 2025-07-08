# dashboard/callbacks/callbacks_previsao_vendas.py

from dash.dependencies import Input, Output, State, ALL
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import pickle
import os
from io import StringIO
from prophet import Prophet
# Ajuste de performance: cache para Prophet até horizonte fixo
MAX_HORIZON = 90
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import functools
from sklearn.ensemble import RandomForestRegressor
import dash
import plotly.io as pio
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import logging
import json
from datetime import datetime
import glob
from plotly.subplots import make_subplots

from ..data.data_loader import carregar_dados, get_principal_dataset, filtrar_por_data
from ..core.utils import criar_figura_vazia, parse_json_to_df
from ..core.forecast_utils import train_and_forecast

# Cache para modelos treinados
# A chave será uma tupla de identificadores únicos para um conjunto de dados de treinamento
MODEL_CACHE = {}

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_currency(v, metrica_selecionada=None):
    if metrica_selecionada == 'SalesPerCustomer':
        return f"€ {v:,.2f}".replace('.', 'X').replace(',', '.').replace('X', ',')
    else:
        return f"€ {v:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.')

# Parâmetros de elasticidade (hipotéticos, podem vir de arquivo JSON)
ELASTIC_PRICE = -1.2    # cada -1% preço -> +1.2% vendas
ELASTIC_PROMO = 0.8     # cada +1 p.p. promo -> +0.8% vendas
ELASTIC_COMP_DIST = 0.3 # cada +1km distância -> +0.3% vendas
ELASTIC_COMP_PROMO = -0.4 # cada +1% promo comp -> -0.4% vendas

def deserializar_df(store_data):
    """
    Desserializa o DataFrame principal a partir do dcc.Store.
    Lida com diferentes formatos de dados e adiciona logs para depuração.
    """
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
        if 'todos' not in tipos_loja:
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
        if tipos_loja and 'todos' not in tipos_loja:
            df = df[df['StoreType'].isin(tipos_loja)]
        lojas = sorted(df['Store'].unique())
        options = [{'label': f"Loja {loja}", 'value': loja} for loja in lojas]
        return options, []

    # Callback para tornar "Todos" mutuamente exclusivo no tipo de loja
    @aplicativo.callback(
        Output('dropdown-tipo-loja', 'value'),
        Input('dropdown-tipo-loja', 'value'),
        prevent_initial_call=True
    )
    def atualizar_tipos_loja(tipos_selecionados):
        if not tipos_selecionados:
            return ['todos']

        # Se 'todos' foi selecionado, desmarque os outros
        if 'todos' in tipos_selecionados and len(tipos_selecionados) > 1:
            # Se o último selecionado foi 'todos'
            if dash.callback_context.triggered[0]['value'][-1] == 'todos':
                return ['todos']
            # Se um tipo específico foi selecionado enquanto 'todos' estava marcado
            else:
                return [t for t in tipos_selecionados if t != 'todos']
        
        return tipos_selecionados

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
        Output('metricas-previsao', 'children'),
        Output('armazenamento-metrica-selecionada', 'data'),
        Output('armazenamento-forecast-diario', 'data'), # NOVO OUTPUT para o forecast diário
        Output('armazenamento-hist-diario', 'data'),
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
            State('xgb-estimators','value'), State('xgb-lr','value'), State('lgbm-estimators','value'),
        ]
    )
    def gerar_previsao(target, horizonte, granularidade, modelo,
                       tipo_loja, lojas, promocao, dias_semana, store_data,
                       xgb_estimators, xgb_lr, lgbm_estimators):
        logger.info(f"Callback de previsão iniciado. Modelo: {modelo}, Tipos: {tipo_loja}, Lojas: {lojas}")
        
        df = deserializar_df(store_data)
        if df is None:
            logger.warning("DataFrame é None após deserialização")
            return criar_figura_vazia("Carregando dados..."), [], html.Div(), None, None, None
        
        # Uma verificação inicial para não mostrar nada se nenhum filtro de loja estiver definido
        if not lojas and not tipo_loja:
            logger.warning("Nenhum filtro de loja selecionado")
            return criar_figura_vazia("Selecione um tipo de loja ou loja específica"), [], html.Div(), None, None, None

        # MODIFICAÇÃO: Filtra tudo, EXCETO dias da semana. O filtro de dias será aplicado DEPOIS da previsão.
        df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, ['todos'])
        
        if df_filtrado.empty:
            logger.warning("DataFrame filtrado está vazio")
            return criar_figura_vazia("Sem dados após aplicação dos filtros"), [], html.Div(), None, None, None

        # Seleção de métrica conforme escolha do usuário
        if target == 'Sales':
            col = 'Sales'
            metrica_nome = 'Vendas'
        elif target == 'Customers':
            col = 'Customers'
            metrica_nome = 'Clientes'
        elif target == 'SalesPerCustomer':
            col = 'TicketMedio' # A coluna será criada abaixo
            metrica_nome = 'Ticket Médio'
        else: # Default para 'Sales'
            col = 'Sales'
            metrica_nome = 'Vendas'
            
        # Preparar série temporal conforme granularidade
        df_agrupado = df_filtrado.set_index('Date')
        
        try:
            # =============================
            # 1) Sempre criar série DIÁRIA
            # =============================
            freq = 'D'
            if target == 'SalesPerCustomer':
                # Agrega Sales e Customers primeiro, depois calcula TicketMedio
                ts_sales = df_agrupado.resample(freq).sum(numeric_only=True)[['Sales']].reset_index()
                ts_customers = df_agrupado.resample(freq).sum(numeric_only=True)[['Customers']].reset_index()
                # REMOVER DOMINGOS
                ts_sales = ts_sales[ts_sales['Date'].dt.dayofweek != 6]
                ts_customers = ts_customers[ts_customers['Date'].dt.dayofweek != 6]
                
                ts = pd.merge(ts_sales, ts_customers, on='Date', how='inner')

                # Remove dias sem clientes para evitar ticket = 0
                ts = ts[(ts['Customers'] > 0) & (ts['Sales'] > 0)]

                ts['TicketMedio'] = ts['Sales'] / ts['Customers']
                ts.rename(columns={'TicketMedio': 'y', 'Date': 'ds'}, inplace=True)
                ts = ts[['ds', 'y']]  # Manter apenas as colunas necessárias
            else:
                ts_raw = df_agrupado.resample(freq).sum(numeric_only=True)[[col]].reset_index()
                # REMOVER DOMINGOS
                ts_raw = ts_raw[ts_raw['Date'].dt.dayofweek != 6]
                # Remove dias com vendas nulas ou zeradas (loja fechada) em vez de interpolar
                ts_raw = ts_raw[ts_raw[col] > 0]
                ts_raw.rename(columns={col: 'y', 'Date': 'ds'}, inplace=True)
                ts = ts_raw

            if ts.empty or len(ts) < 10:
                logger.warning(f"Dados insuficientes para o modelo {modelo}")
                return criar_figura_vazia(f"Dados insuficientes para o modelo {modelo}"), [], html.Div(), None, None, None
                
            logger.info(f"Série temporal preparada com sucesso: {len(ts)} pontos")
            
            # Geração do DataFrame futuro movida e corrigida
            last_date = ts['ds'].max()
            
            # Horizonte sempre em semanas → converter para dias
            periodos = 7 * horizonte
            offset = pd.DateOffset(days=1)
            
            future_dates = pd.date_range(start=last_date + offset, periods=periodos, freq=freq)
            # REMOVER DOMINGOS DO FUTURO
            if freq == 'D':
                future_dates = future_dates[future_dates.dayofweek != 6]
            df_future = pd.DataFrame({'ds': future_dates})

            # =============================
            # 2) Treinamento / Geração do Forecast (refatorado)
            # =============================
            # Constrói DataFrame de feriados (estaduais e escolares)
            state_hols = df_filtrado[df_filtrado['StateHoliday'].astype(str) != '0'][['Date']].drop_duplicates().rename(columns={'Date': 'ds'})
            state_hols['holiday'] = 'state_holiday'
            school_hols = df_filtrado[df_filtrado['SchoolHoliday'] == 1][['Date']].drop_duplicates().rename(columns={'Date': 'ds'})
            school_hols['holiday'] = 'school_holiday'
            holidays = pd.concat([state_hols, school_hols])

            # --------------------------------------
            #  Variáveis EXÓGENAS (Promo, feriados)
            # --------------------------------------
            exog_daily = (df_filtrado
                           .groupby('Date')
                           .agg(Promo=('Promo', 'max'),
                                SchoolHoliday=('SchoolHoliday', 'max'),
                                StateHoliday_flag=('StateHoliday', lambda x: int(any(x.astype(str) != '0'))))
                           .reset_index())
            # Normalização de tipos
            exog_daily['Promo'] = exog_daily['Promo'].astype(int)
            exog_daily['SchoolHoliday'] = exog_daily['SchoolHoliday'].astype(int)

            # Junta às series
            ts = ts.merge(exog_daily, left_on='ds', right_on='Date', how='left').drop(columns=['Date'])
            ts.fillna({'Promo': 0, 'SchoolHoliday': 0, 'StateHoliday_flag': 0}, inplace=True)

            # ---------- CACHE DE MODELOS ----------
            flag_excluir_domingo = True  # sempre excluímos domingo na granularidade diária

            # MODIFICAÇÃO: A chave do cache não inclui mais dias da semana, pois o filtro será aplicado após a previsão.
            cache_key = (
                tuple(sorted(lojas)) if lojas else tuple(sorted(tipo_loja)) if tipo_loja else 'ALL',
                target,
                modelo,
                horizonte,
                ts['ds'].max(),
                'no_sunday' if flag_excluir_domingo else 'with_sunday'
            )
            if cache_key in MODEL_CACHE:
                logger.info("Reutilizando modelo do cache para chave %s", cache_key)
                forecast, pred_train, deviation_count, model_name_display = MODEL_CACHE[cache_key]
            else:
                try:
                    # MODIFICAÇÃO: a função train_and_forecast não recebe mais dias_semana
                    forecast, pred_train, deviation_count, model_name_display = train_and_forecast(
                        ts,
                        future_dates,
                        modelo,
                        holidays=holidays,
                        xgb_params={'n_estimators': xgb_estimators, 'learning_rate': xgb_lr},
                        lgbm_params={'n_estimators': lgbm_estimators}
                    )
                    # Armazena a previsão completa no cache
                    MODEL_CACHE[cache_key] = (forecast, pred_train, deviation_count, model_name_display)
                except Exception as e:
                    logger.error(f"Erro ao gerar previsão: {str(e)}")
                    return criar_figura_vazia(f"Erro ao gerar previsão: {str(e)}"), [], html.Div(), None, None, None

            # MODIFICAÇÃO: Filtra a previsão *APÓS* obtê-la (do cache ou do cálculo)
            if dias_semana and 'todos' not in dias_semana:
                forecast['DayOfWeek'] = forecast['ds'].dt.dayofweek + 1
                forecast = forecast[forecast['DayOfWeek'].isin(dias_semana)].drop(columns=['DayOfWeek'])

            # MODIFICAÇÃO EXTRA: remover domingos do forecast (segurança)
            if granularidade == 'diaria':
                forecast = forecast[forecast['ds'].dt.dayofweek != 6]

            logger.info(f"Previsão gerada com sucesso: {len(forecast)} pontos")

            # =============================
            # 3) Agrega dados para exibição
            # =============================
            if granularidade == 'diaria':
                display_freq = 'D'
            elif granularidade == 'semanal':
                # âncora no dia da semana do último ponto histórico
                anchor = ts['ds'].max().strftime('%a').upper()[:3]  # ex: 'FRI'
                display_freq = f'W-{anchor}'
            else:
                display_freq = 'MS'

            agg_func = 'mean' if target == 'SalesPerCustomer' else 'sum'

            def _agg(df, cols):
                mapping = {c: agg_func for c in cols}
                res = (df.set_index('ds')
                         .resample(display_freq, label='left', closed='left')
                         .agg(mapping))
                # Conta quantos registros originais caíram em cada bucket
                counts = (df.set_index('ds')
                            .resample(display_freq, label='left', closed='left')
                            .size())
                max_count = counts.max() if len(counts) else 0
                # Mantém apenas buckets completos
                res = res[counts == max_count]
                return res.reset_index()

            # Cria uma cópia do histórico para plotagem e a filtra se necessário
            ts_para_plot = ts.copy()
            if dias_semana and 'todos' not in dias_semana:
                ts_para_plot['DayOfWeek'] = ts_para_plot['ds'].dt.dayofweek + 1
                ts_para_plot = ts_para_plot[ts_para_plot['DayOfWeek'].isin(dias_semana)].drop(columns=['DayOfWeek'])

            # NOVO: se não houver dados após o filtro, retornar figura/tabela vazias
            if ts_para_plot.empty:
                logger.warning("Sem dados após aplicar filtro de dia da semana (%s)", dias_semana)
                return criar_figura_vazia("Sem dados para o(s) dia(s) da semana selecionado(s)."), [], html.Div(), None, None, None

            if display_freq != 'D':
                ts_plot = _agg(ts_para_plot, ['y'])
                if {'yhat_lower','yhat_upper'}.issubset(forecast.columns):
                    forecast_plot = _agg(forecast, ['yhat','yhat_lower','yhat_upper'])
                else:
                    forecast_plot = _agg(forecast, ['yhat'])
            else:
                ts_plot = ts_para_plot
                forecast_plot = forecast

            # =============================
            # 4) Constrói Figura
            # =============================
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=ts_plot['ds'], y=ts_plot['y'], name='Histórico', mode='lines', line=dict(color='#1f77b4')))

            data_inicio_visual = ts_plot['ds'].max() - pd.DateOffset(months=2)
            data_fim_visual = forecast_plot['ds'].max()

            fig.update_layout(xaxis_rangeslider=dict(visible=True))
            fig.update_xaxes(range=[data_inicio_visual, data_fim_visual])

            # Previsão
            fig.add_trace(go.Scatter(x=forecast_plot['ds'], y=forecast_plot['yhat'], name='Previsão', mode='lines', line=dict(color='#ff7f0e')))

            if 'yhat_lower' in forecast_plot.columns and 'yhat_upper' in forecast_plot.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_plot['ds'], y=forecast_plot['yhat_upper'], fill=None, mode='lines', 
                    line=dict(color='#ff7f0e', width=0), showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_plot['ds'], y=forecast_plot['yhat_lower'], fill='tonexty', mode='lines', 
                    fillcolor='rgba(255, 127, 14, 0.2)', line=dict(color='#ff7f0e', width=0),
                    name='Intervalo de Confiança'
                ))

            fig.update_layout(
                title=f'Previsão de {metrica_nome} com {model_name_display} ({granularidade.capitalize()})',
                xaxis_title='Data', 
                yaxis_title=metrica_nome,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # =============================
            # 5) Montagem da tabela com coluna de variação
            # =============================
            df_tab = pd.DataFrame({
                'Data': forecast_plot['ds'],
                'Previsão': forecast_plot['yhat']
            })
            # Formatação
            df_tab['Data_fmt'] = df_tab['Data'].dt.strftime('%d/%m/%Y')
            df_tab['Prev_fmt'] = df_tab['Previsão'].round(2).apply(lambda x: f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
            # Cálculo da variação percentual dia-a-dia (ignora divisão por ~0)
            df_tab['Var_pct'] = df_tab['Previsão'].pct_change() * 100
            df_tab.loc[df_tab['Previsão'].shift(1).abs() < 1, 'Var_pct'] = np.nan

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

            # ========================= CALCULA MÉTRICAS (refatorado) =========================
            try:
                if pred_train is not None:
                    y_true = ts['y'].values
                    mae = mean_absolute_error(y_true, pred_train)
                    rmse = np.sqrt(mean_squared_error(y_true, pred_train))
                    # Evita divisão por zero no MAPE
                    mask = y_true != 0
                    mape = np.mean(np.abs((y_true[mask] - pred_train[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
                else:
                    mae = rmse = mape = np.nan
            except Exception as e:
                logger.error(f"Erro ao calcular métricas: {e}")
                mae = rmse = mape = np.nan

            # --------- COMPONENTE DE MÉTRICAS ---------
            metric_items = [
                html.Span(f"MAE: {mae:,.2f}", className="me-3"),
                html.Span(f"RMSE: {rmse:,.2f}", className="me-3"),
                html.Span(f"MAPE: {mape:.2f}%", className="me-3")
            ]
            metricas_div = html.Div(metric_items)

            logger.info("Gráfico, tabela e métricas gerados com sucesso")
            return fig, tabela_dash, metricas_div, target, forecast.to_json(date_format='iso', orient='split'), ts.to_json(date_format='iso', orient='split')
            
        except Exception as e:
            logger.error(f"Erro ao montar gráfico ou tabela: {str(e)}")
            return criar_figura_vazia(f"Erro ao montar visualização: {str(e)}"), [], html.Div(), None, None, None

    @aplicativo.callback(
        Output('informacoes-previsao', 'children'),
        Input('grafico-previsao', 'figure'),
        Input('armazenamento-metrica-selecionada', 'data'),
        Input('armazenamento-forecast-diario', 'data'), 
        Input('armazenamento-hist-diario', 'data')
    )
    def atualizar_informacoes_previsao(fig, metrica_selecionada, forecast_diario_json, hist_diario_json): 
        if not fig or 'data' not in fig or not forecast_diario_json or not hist_diario_json:
            return [html.Div("Sem dados de previsão para exibir.")]
        try:
            # Carrega dados históricos e de previsão DIÁRIOS
            df_hist_daily = pd.read_json(StringIO(hist_diario_json), orient='split')
            df_fc_daily = pd.read_json(StringIO(forecast_diario_json), orient='split')

            # Garante que as colunas de data sejam do tipo datetime
            df_hist_daily['ds'] = pd.to_datetime(df_hist_daily['ds'])
            df_fc_daily['ds'] = pd.to_datetime(df_fc_daily['ds'])

            # Extrai o 'y' do histórico e 'yhat' da previsão
            hist_y = df_hist_daily['y']
            fc_y_daily = df_fc_daily['yhat']
            
            # Extrair informações do título do gráfico para obter a granularidade
            titulo = fig['layout']['title']['text'] if 'title' in fig['layout'] and 'text' in fig['layout']['title'] else ""
            partes = titulo.split(' - ') if titulo else []
            granularidade = partes[1] if len(partes) > 1 else "Semanal"
            
            # Calcular horizonte em semanas
            horizonte_semanas = len(df_fc_daily) // 7
            if len(df_fc_daily) % 7 > 0:
                horizonte_semanas += 1
            
            # Cálculos principais
            if metrica_selecionada == 'SalesPerCustomer':
                total_prev = fc_y_daily.mean()
                total_prev_label = "Média Prevista"
            else:
                total_prev = fc_y_daily.sum()
                total_prev_label = "Total Previsto"

            media_prev = fc_y_daily.mean()
            media_hist = hist_y.mean() if not hist_y.empty else np.nan
            var_perc = (media_prev / media_hist - 1) * 100 if not np.isnan(media_hist) and media_hist != 0 else 0
            
            idx_max = int(np.nanargmax(fc_y_daily))
            idx_min = int(np.nanargmin(fc_y_daily))
            pico_date = df_fc_daily['ds'].iloc[idx_max].strftime('%d/%m/%Y')
            pico_val = fc_y_daily.iloc[idx_max]
            vale_date = df_fc_daily['ds'].iloc[idx_min].strftime('%d/%m/%Y')
            vale_val = fc_y_daily.iloc[idx_min]
            
            # Intervalo de confiança (ainda do gráfico, pois é para exibição)
            if len(fig['data']) >= 4:
                lower = np.array(fig['data'][2]['y'], dtype=float)
                upper = np.array(fig['data'][3]['y'], dtype=float)
                amp_ic = np.mean(upper - lower)
            else:
                amp_ic = np.nan
                
            # Variação acumulada e desvio padrão (usando forecast diário)
            var_acum = (fc_y_daily.iloc[-1] / fc_y_daily.iloc[0] - 1) * 100 if fc_y_daily.iloc[0] else 0
            std_val = np.std(fc_y_daily)
            
            # Maior ganho percentual dia-a-dia (usando forecast diário)
            pct_diff = (np.diff(fc_y_daily) / fc_y_daily[:-1] * 100) if fc_y_daily.size > 1 else np.array([0])
            max_gain = pct_diff.max()
            
            # Dia da semana com maior previsão (usando forecast diário)
            df_fc_daily['weekday'] = df_fc_daily['ds'].dt.weekday
            dias_media = df_fc_daily.groupby('weekday')['yhat'].mean()
            mapping = {0: 'Segunda-feira', 1: 'Terça-feira', 2: 'Quarta-feira', 3: 'Quinta-feira', 4: 'Sexta-feira', 5: 'Sábado', 6: 'Domingo'}
            top_weekday = mapping.get(int(dias_media.idxmax()), '') if not dias_media.empty else ''
            
            # Adicionar informação do horizonte em semanas
            items = [
                (total_prev_label, format_currency(total_prev, metrica_selecionada)),
                ("Variação vs. Hist. (%)", f"{var_perc:,.2f}%", "text-success" if var_perc>=0 else "text-danger"),
                ("Variação Acumulada (%)", f"{var_acum:,.2f}%", "text-success" if var_acum>=0 else "text-danger"),
                ("Melhor Dia da Semana", top_weekday),
            ]
            # Adiciona Média Prevista apenas se não for Ticket Médio
            if metrica_selecionada != 'SalesPerCustomer':
                items.insert(1, ("Média Prevista", format_currency(media_prev, metrica_selecionada)))

            items_divs = []
            for item in items:
                titulo = item[0]
                valor = item[1]
                estilo_valor = "fw-bold " + (item[2] if len(item) > 2 else "")
                
                items_divs.append(
                    html.Div([
                        html.P(titulo, className="text-muted mb-1 small"),
                        html.H5(valor, className=estilo_valor)
                    ], className="info-item")
                )

            return html.Div(items_divs, className="info-panel-grid")
        except Exception as e:
            return [html.Div(f"Erro ao gerar informações: {e}")]
    
    # ==================================================================
    # NOVOS GRÁFICOS (gerados a partir do grafico-previsao já existente)
    # ==================================================================

    # Callback para gráficos adicionais substituído por uma nova versão abaixo

    # ==================================================================
    # SIMULADOR WHAT-IF
    # ==================================================================

    # Callback principal do What-If
    @aplicativo.callback(
        Output('grafico-whatif', 'figure'),
        Output('kpi-total-base', 'children'),
        Output('kpi-total-sim', 'children'),
        Output('kpi-variacao-sim', 'children'),
        Output('insights-whatif', 'children'),
        Output('resultados-whatif', 'style'),
        Input('btn-simular-whatif', 'n_clicks'),
        [
            State('slider-whatif-preco', 'value'),
            State('slider-whatif-promo', 'value'),
            State('slider-whatif-comp-dist', 'value'),
            State('slider-whatif-comp-promo', 'value'),
            State('check-whatif-feriados', 'value'),
            State('select-whatif-eventos', 'value'),
            State('grafico-previsao', 'figure')
        ],
        prevent_initial_call=True
    )
    def simular_whatif(n_clicks, delta_preco_pct, delta_promo_pp, 
                      comp_dist, comp_promo, feriados, eventos, fig_previsao):
        if not fig_previsao or 'data' not in fig_previsao or len(fig_previsao['data']) < 2:
            fig_placeholder = go.Figure()
            fig_placeholder.update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False,
                                          annotations=[dict(text="Gere a previsão primeiro", showarrow=False)])
            return fig_placeholder, "-", "-", "-", "", {"display": "none"}

        # Dados base
        fc_x = pd.to_datetime(fig_previsao['data'][1]['x'])
        fc_y = np.array(fig_previsao['data'][1]['y'], dtype=float)

        # Fatores de ajuste por variável
        fator_preco = 1 + (ELASTIC_PRICE * (delta_preco_pct / 100))
        fator_promo = 1 + (ELASTIC_PROMO * (delta_promo_pp / 100))
        fator_comp_dist = 1 + (ELASTIC_COMP_DIST * ((comp_dist - 5) / 10))  # 5km é o baseline
        fator_comp_promo = 1 + (ELASTIC_COMP_PROMO * (comp_promo / 100))

        # Ajuste por eventos especiais
        fator_eventos = {
            "none": 1.0,
            "back_to_school": 1.15,
            "christmas": 1.3,
            "easter": 1.2
        }.get(eventos, 1.0)

        # Ajuste por feriados
        fator_feriados = 1.1 if feriados else 1.0

        # Ajuste final combinado
        ajuste = (fator_preco * fator_promo * fator_comp_dist * fator_comp_promo * 
                 fator_eventos * fator_feriados)

        y_sim = fc_y * ajuste

        # Gráfico comparativo
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=fc_x, y=fc_y, name='Base', mode='lines', 
                                   line=dict(color='#1f77b4')))
        fig_sim.add_trace(go.Scatter(x=fc_x, y=y_sim, name='Cenário', mode='lines', 
                                   line=dict(color='#ff7f0e')))
        
        # Área entre as curvas
        fig_sim.add_trace(go.Scatter(
            x=fc_x.tolist() + fc_x.tolist()[::-1],
            y=y_sim.tolist() + fc_y.tolist()[::-1],
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_sim.update_layout(
            template='plotly_white',
            xaxis_title='Data',
            yaxis_title='Vendas Previstas',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Cálculos dos KPIs
        total_base = fc_y.sum()
        total_sim = y_sim.sum()
        var_pct = ((total_sim / total_base) - 1) * 100

        

        # Gerar insights automáticos
        insights = []
        
        # Insight de variação geral
        if var_pct > 0:
            insights.append(html.Div([
                html.I(className="fas fa-arrow-up text-success me-2"),
                f"Aumento projetado de {var_pct:.1f}% nas vendas"
            ], className="mb-2"))
        else:
            insights.append(html.Div([
                html.I(className="fas fa-arrow-down text-danger me-2"),
                f"Redução projetada de {abs(var_pct):.1f}% nas vendas"
            ], className="mb-2"))

        # Análise de sensibilidade
        impactos = {
            'Preço': ELASTIC_PRICE * delta_preco_pct,
            'Promoção': ELASTIC_PROMO * delta_promo_pp,
            'Dist. Competidor': ELASTIC_COMP_DIST * (comp_dist - 5),
            'Promo Competidor': ELASTIC_COMP_PROMO * comp_promo
        }
        maior_impacto = max(impactos.items(), key=lambda x: abs(x[1]))
        insights.append(html.Div([
            html.I(className="fas fa-chart-line me-2"),
            f"Maior impacto: {maior_impacto[0]} ({maior_impacto[1]:.1f}%)"
        ], className="mb-2"))

        # Alertas de risco
        if abs(var_pct) > 50:
            insights.append(html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                "Atenção: Variação muito expressiva, revise os parâmetros"
            ], className="mb-2"))

        # Recomendações
        if delta_preco_pct > 0 and delta_promo_pp < 0:
            insights.append(html.Div([
                html.I(className="fas fa-lightbulb text-primary me-2"),
                "Considere aumentar promoções para compensar aumento de preço"
            ], className="mb-2"))

        insights_div = html.Div([
            html.H6("Insights", className="mb-3"),
            html.Div(insights)
        ])

        return (
            fig_sim,
            format_currency(total_base),
            format_currency(total_sim),
            f"{var_pct:+.1f}%" if var_pct != 0 else "0%",
            insights_div,
            {"display": "block"}
        )

    # Callback para abrir o modal de salvar cenário
    @aplicativo.callback(
        Output("modal-salvar-cenario", "is_open"),
        [
            Input("btn-salvar-whatif", "n_clicks"),
            Input("btn-cancelar-salvar-cenario", "n_clicks"),
            Input("btn-confirmar-salvar-cenario", "n_clicks")
        ],
        [State("modal-salvar-cenario", "is_open")]
    )
    def toggle_modal_salvar_cenario(n1, n2, n3, is_open):
        if n1 or n2 or n3:
            return not is_open
        return is_open

    # Callback para salvar o cenário
    @aplicativo.callback(
        [
            Output("toast-feedback-cenario", "is_open"),
            Output("toast-feedback-cenario", "children"),
            Output("toast-feedback-cenario", "header"),
            Output("input-nome-cenario", "value")
        ],
        Input("btn-confirmar-salvar-cenario", "n_clicks"),
        [
            State("input-nome-cenario", "value"),
            State("slider-whatif-preco", "value"),
            State("slider-whatif-promo", "value"),
            State("slider-whatif-comp-dist", "value"),
            State("slider-whatif-comp-promo", "value"),
            State("check-whatif-feriados", "value"),
            State("select-whatif-eventos", "value")
        ],
        prevent_initial_call=True
    )
    def salvar_cenario(n_clicks, nome_cenario, delta_preco_pct, delta_promo_pp, 
                       comp_dist, comp_promo, feriados, eventos):
        if not n_clicks:
            return False, "", "", ""
            
        if not nome_cenario:
            return True, "Por favor, forneça um nome para o cenário", "Erro", ""
            
        # Prepara os dados do cenário
        dados_cenario = {
            "descricao": nome_cenario,
            "data_criacao": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "parametros": {
                "delta_preco_pct": delta_preco_pct,
                "delta_promo_pp": delta_promo_pp,
                "comp_dist": comp_dist,
                "comp_promo": comp_promo,
                "feriados": bool(feriados),
                "eventos": eventos if eventos else "none"
            }
        }
        
        # Garante que o diretório existe
        os.makedirs("cenarios", exist_ok=True)
        
        # Gera nome do arquivo
        nome_arquivo = f"cenarios/cenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Salva o cenário
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                json.dump(dados_cenario, f, ensure_ascii=False, indent=4)
                
            return True, f"Cenário '{nome_cenario}' salvo com sucesso!", "Sucesso", ""
        except Exception as e:
            return True, f"Erro ao salvar cenário: {str(e)}", "Erro", nome_cenario

    # Callback para exibir a confirmação de limpeza de cenários
    @aplicativo.callback(
        Output('confirm-limpar-cenarios', 'displayed'),
        Input('btn-limpar-cenarios', 'n_clicks'),
        prevent_initial_call=True
    )
    def exibir_confirmacao_limpeza(n_clicks):
        if n_clicks:
            return True
        return False

    # Callback para efetivamente limpar os cenários
    @aplicativo.callback(
        [Output('store-cenarios-update-trigger', 'data'),
         Output('toast-feedback-cenario', 'is_open', allow_duplicate=True),
         Output('toast-feedback-cenario', 'children', allow_duplicate=True),
         Output('toast-feedback-cenario', 'header', allow_duplicate=True)],
        Input('confirm-limpar-cenarios', 'submit_n_clicks'),
        prevent_initial_call=True
    )
    def limpar_todos_cenarios(submit_n_clicks):
        if not submit_n_clicks:
            return no_update, no_update, no_update, no_update
        
        try:
            arquivos = glob.glob('cenarios/*.json')
            for arquivo in arquivos:
                os.remove(arquivo)
            
            return datetime.now().isoformat(), True, "Todos os cenários foram apagados.", "Sucesso"
        except Exception as e:
            return no_update, True, f"Erro ao apagar cenários: {str(e)}", "Erro"

    # Callback para atualizar a lista de cenários disponíveis
    @aplicativo.callback(
        Output('dropdown-cenarios', 'options'),
        [
            Input('btn-confirmar-salvar-cenario', 'n_clicks'),
            Input('store-cenarios-update-trigger', 'data'), # Alterado
            Input('dropdown-cenarios', 'id')  # Para carga inicial
        ]
    )
    def atualizar_lista_cenarios(n_clicks_salvar, update_trigger, _):
        import glob
        import json
        from datetime import datetime
        
        # Lista todos os arquivos de cenário
        arquivos = glob.glob('cenarios/*.json')
        opcoes = []
        
        for arquivo in arquivos:
            try:
                with open(arquivo, 'r') as f:
                    dados = json.load(f)
                    # Formata o nome do cenário com data e descrição
                    data = datetime.strptime(dados.get('data_criacao', ''), '%Y%m%d_%H%M%S')
                    nome = f"{data.strftime('%d/%m/%Y %H:%M')} - {dados.get('descricao', 'Sem descrição')}"
                    opcoes.append({
                        'label': nome,
                        'value': arquivo
                    })
            except:
                continue
                
        # Ordena por data (mais recente primeiro)
        opcoes.sort(key=lambda x: x['label'], reverse=True)
        return opcoes

    # Callback para comparar cenários
    @aplicativo.callback(
        Output('resultados-comparacao', 'children'),
        Output('resultados-comparacao', 'style'),
        Input('btn-comparar-cenarios', 'n_clicks'),
        State('dropdown-cenarios', 'value'),
        State('grafico-previsao', 'figure'),
        prevent_initial_call=True
    )
    def comparar_cenarios(n_clicks, cenarios_selecionados, fig_previsao):
        import json
        import pandas as pd
        import plotly.graph_objects as go
        from dash import dash_table
        
        if not n_clicks or not cenarios_selecionados or len(cenarios_selecionados) == 0:
            return "", {"display": "none"}
            
        # Se não houver dados de previsão base, retorna erro
        if not fig_previsao or 'data' not in fig_previsao or len(fig_previsao['data']) < 2:
            return html.Div([
                html.I(className="fas fa-exclamation-circle text-danger me-2"),
                "Gere a previsão base primeiro"
            ], className="alert alert-danger"), {"display": "block"}
            
        # Dados base
        fc_x = pd.to_datetime(fig_previsao['data'][1]['x'])
        fc_y = np.array(fig_previsao['data'][1]['y'], dtype=float)
        total_base = fc_y.sum()
        
        # Carrega dados dos cenários
        dados_cenarios = []
        for arquivo in cenarios_selecionados:
            try:
                with open(arquivo, 'r') as f:
                    cenario = json.load(f)
                    dados_cenarios.append(cenario)
            except:
                continue
                
        if not dados_cenarios:
            return html.Div([
                html.I(className="fas fa-exclamation-circle text-warning me-2"),
                "Nenhum cenário válido selecionado"
            ], className="alert alert-warning"), {"display": "block"}
            
        # Cria gráfico comparativo
        fig_comp = go.Figure()
        
        # Adiciona linha base
        fig_comp.add_trace(go.Scatter(
            x=fc_x, 
            y=fc_y, 
            name='Base',
            mode='lines',
            line=dict(color='#1f77b4')
        ))
        
        # Cores para os cenários
        cores = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Dados para a tabela comparativa
        dados_tabela = [{
            'cenario': 'Base',
            'total_vendas': f"€ {total_base:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
            'variacao': '0%'
        }]
        
        # Adiciona cada cenário
        for i, cenario in enumerate(dados_cenarios):
            # Recalcula o ajuste usando os parâmetros salvos
            params = cenario['parametros']
            
            # Fatores de ajuste por variável
            fator_preco = 1 + (ELASTIC_PRICE * (params['delta_preco_pct'] / 100))
            fator_promo = 1 + (ELASTIC_PROMO * (params['delta_promo_pp'] / 100))
            fator_comp_dist = 1 + (ELASTIC_COMP_DIST * ((params['comp_dist'] - 5) / 10))
            fator_comp_promo = 1 + (ELASTIC_COMP_PROMO * (params['comp_promo'] / 100))
            
            # Ajuste por eventos especiais
            fator_eventos = {
                "none": 1.0,
                "back_to_school": 1.15,
                "christmas": 1.3,
                "easter": 1.2
            }.get(params.get('eventos', 'none'), 1.0)
            
            # Ajuste por feriados
            fator_feriados = 1.1 if params.get('feriados', False) else 1.0
            
            # Ajuste final
            ajuste = (fator_preco * fator_promo * fator_comp_dist * 
                     fator_comp_promo * fator_eventos * fator_feriados)
            
            y_sim = fc_y * ajuste
            total_sim = y_sim.sum()
            var_pct = ((total_sim / total_base) - 1) * 100
            
            # Adiciona ao gráfico
            cor = cores[i % len(cores)]
            fig_comp.add_trace(go.Scatter(
                x=fc_x,
                y=y_sim,
                name=cenario['descricao'],
                mode='lines',
                line=dict(color=cor)
            ))
            
            # Adiciona à tabela
            dados_tabela.append({
                'cenario': cenario['descricao'],
                'total_vendas': f"€ {total_sim:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
                'variacao': f"{var_pct:+.1f}%"
            })
            
        # Atualiza layout do gráfico
        fig_comp.update_layout(
            template='plotly_white',
            xaxis_title='Data',
            yaxis_title='Vendas Previstas',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        
        # Cria tabela comparativa
        tabela = dash_table.DataTable(
            data=dados_tabela,
            columns=[
                {'name': 'Cenário', 'id': 'cenario'},
                {'name': 'Total de Vendas', 'id': 'total_vendas'},
                {'name': 'Variação', 'id': 'variacao'}
            ],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'variacao', 'filter_query': '{variacao} contains "+"'},
                    'color': 'green'
                },
                {
                    'if': {'column_id': 'variacao', 'filter_query': '{variacao} contains "-"'},
                    'color': 'red'
                }
            ]
        )
        
        return html.Div([
            dcc.Graph(figure=fig_comp, className="mb-4"),
            html.H6("Comparativo de Resultados", className="mb-3"),
            tabela
        ]), {"display": "block"}

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
        from prophet import Prophet
        df = deserializar_df(store_data)
        if df is None or df.empty:
            return go.Figure()
        df_filtrado = filtrar_dados(df, lojas, tipo_loja, promocao, dias_semana)
        if df_filtrado.empty:
            return go.Figure()

        # Prepara série de SALES (principal) --------------------------------
        df_work = df_filtrado.copy()
        df_work['TicketMedio'] = (df_work['Sales'] / df_work['Customers'].replace(0, np.nan)).fillna(0)
        freq_map = {'diaria':'D','semanal':'W','mensal':'MS'}
        freq = freq_map.get(granularidade, 'D')
        df_agr_sales = df_work.set_index('Date').resample(freq).sum(numeric_only=True)[['Sales']].dropna().reset_index()
        df_agr_sales.rename(columns={'Sales':'y','Date':'ds'}, inplace=True)
        if df_agr_sales.empty or len(df_agr_sales)<10:
            return go.Figure()

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df_agr_sales)
        last_date = df_agr_sales['ds'].max()
        offset = pd.DateOffset(days=1) if freq=='D' else (pd.DateOffset(weeks=1) if freq=='W' else pd.DateOffset(months=1))
        future_dates = pd.date_range(start=last_date+offset, periods=horizonte, freq=freq)
        fc_sales = m.predict(pd.DataFrame({'ds':future_dates}))

        # YoY growth (%): comparar com histórico 1 ano antes
        one_year = pd.DateOffset(years=1)
        past_dates = fc_sales['ds'] - one_year
        hist_lookup = df_agr_sales.set_index('ds')['y']
        hist_values = hist_lookup.reindex(past_dates).values
        yoy = (fc_sales['yhat'].values - hist_values) / hist_values * 100

        # --- Construir figura com dois eixos
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=fc_sales['ds'], y=fc_sales['yhat'], mode='lines', name='Vendas Previstas', line=dict(color='#1f77b4', width=3)), secondary_y=False)
        # Banda de confiança
        fig.add_trace(go.Scatter(x=pd.concat([fc_sales['ds'], fc_sales['ds'][::-1]]),
                                 y=pd.concat([fc_sales['yhat_upper'], fc_sales['yhat_lower'][::-1]]),
                                 fill='toself', name='Intervalo Conf.', line=dict(color='rgba(31,119,180,0)'),
                                 fillcolor='rgba(31,119,180,0.2)'), secondary_y=False)

        fig.add_trace(go.Scatter(x=fc_sales['ds'], y=yoy, mode='lines', name='% YoY', line=dict(color='#ff7f0e', dash='dash')), secondary_y=True)

        fig.update_yaxes(title_text='Vendas Previstas (€)', secondary_y=False)
        fig.update_yaxes(title_text='% YoY', secondary_y=True, tickformat='.1f')
        fig.update_layout(template='plotly_white', xaxis_title='Data', legend_title='', height=400)
        return fig 

    @aplicativo.callback(
        Output('grafico-media-global', 'figure'),
        Output('grafico-media-tipo-loja', 'figure'),
        Input('grafico-previsao', 'figure'),
        Input('radio-metrica-previsao', 'value'),
        Input('armazenamento-df-principal', 'data')
    )
    def gerar_graficos_adicionais(fig_previsao, metrica_selecionada, store_data):
        """Cria dois gráficos auxiliares, evitando barras:
        1) Linha acumulada semanal previstos × histórico × meta.
        2) Heatmap de vendas previstas por Semana × Tipo de Loja.
        """
        import pandas as pd
        import numpy as np
        from plotly.subplots import make_subplots
        if not fig_previsao or 'data' not in fig_previsao or len(fig_previsao['data']) < 2:
            placeholder = go.Figure()
            placeholder.update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False,
                                      annotations=[dict(text="Aguardando geração da previsão", showarrow=False)])
            return placeholder, placeholder

        # --- Dados da previsão principal ---
        dates_fc = pd.to_datetime(fig_previsao['data'][1]['x'])
        values_fc = np.array(fig_previsao['data'][1]['y'], dtype=float)
        df_fc = pd.DataFrame({'Date': dates_fc, 'Valor': values_fc})
        df_fc['Week'] = df_fc['Date'].dt.isocalendar().week
        df_fc['Year'] = df_fc['Date'].dt.year

        # --- Carregar histórico ---
        df_hist_full = deserializar_df(store_data)
        if df_hist_full is None or df_hist_full.empty:
            df_hist_full = pd.DataFrame(columns=['Date','Sales','Customers'])
        # define coluna a usar
        if metrica_selecionada == 'Sales':
            col_hist = 'Sales'
        elif metrica_selecionada == 'Customers':
            col_hist = 'Customers'
        else:  # Ticket
            df_hist_full['TicketMedio'] = (df_hist_full['Sales'] / df_hist_full['Customers'].replace(0, np.nan)).fillna(0)
            col_hist = 'TicketMedio'

        # Histórico referente ao mesmo número de semanas, ano anterior
        last_hist_date = df_hist_full['Date'].max() if not df_hist_full.empty else None
        df_hist = pd.DataFrame()
        if last_hist_date is not None:
            one_year = pd.DateOffset(years=1)
            past_start = dates_fc.min() - one_year
            past_end = dates_fc.max() - one_year
            df_hist = df_hist_full[(df_hist_full['Date']>=past_start)&(df_hist_full['Date']<=past_end)].copy()
            df_hist['ValorHist'] = df_hist[col_hist]
            df_hist['Week'] = df_hist['Date'].dt.isocalendar().week
            df_hist_group = df_hist.groupby('Week')['ValorHist'].mean().reset_index()
        else:
            df_hist_group = pd.DataFrame({'Week':[], 'ValorHist':[]})

        # --- META ---
        # Meta simples = média histórica +5% (exemplo). Caso não haja histórico, meta = previsão *0.95
        if not df_hist_group.empty:
            meta_series = df_hist_group.copy()
            meta_series['Meta'] = meta_series['ValorHist']*1.05
        else:
            meta_series = df_fc.groupby('Week')['Valor'].mean().reset_index()
            meta_series['Meta'] = meta_series['Valor']*0.95

        # --- Acumulados ---
        acum_prev = df_fc.groupby('Week')['Valor'].mean().cumsum()
        acum_meta = meta_series['Meta'].cumsum()
        acum_hist = df_hist_group['ValorHist'].cumsum() if not df_hist_group.empty else None

        fig_global = go.Figure()
        fig_global.add_trace(go.Scatter(x=acum_prev.index, y=acum_prev.values,
                                         mode='lines', name='Previsto', line=dict(color='#1f77b4', width=3)))
        fig_global.add_trace(go.Scatter(x=acum_meta.index, y=acum_meta.values,
                                         mode='lines', name='Meta', line=dict(color='#ff7f0e', dash='dash')))
        if acum_hist is not None:
            fig_global.add_trace(go.Scatter(x=acum_hist.index, y=acum_hist.values,
                                             mode='lines', name='Ano Anterior', line=dict(color='#2ca02c', dash='dot')))
        fig_global.update_layout(template='plotly_white', xaxis_title='Semana',
                                yaxis_title=f'Acumulado de {metrica_selecionada}',
                                legend_title='', height=300, margin=dict(l=40,r=40,t=30,b=40))

        # --- Heatmap Semana × Tipo de Loja ---
        # Obter StoreType por Store
        df_hist_full = df_hist_full.copy()
        mapa_tipo = dict(zip(df_hist_full['Store'], df_hist_full['StoreType']))
        df_fc['StoreType'] = df_fc['Date'].apply(lambda x: 'Todos')  # placeholder
        # Se existirem lojas filtradas, extrair da string do título do gráfico principal (hack)
        try:
            lojas_str = fig_previsao['layout']['title']['text'].split(' - ')[-1]
            lojas_selec = [] if lojas_str=='Todas' else [int(s) for s in lojas_str.split(', ') if s.isdigit()]
        except Exception:
            lojas_selec = []
        if lojas_selec:
            tipos = [mapa_tipo.get(loja,'Desconhecido') for loja in lojas_selec]
            tipo_dominante = tipos[0] if tipos else 'Desconhecido'
            df_fc['StoreType'] = tipo_dominante
        else:
            # Usar participação média de cada tipo em histórico para distribuir valor previsto
            distrib_tipo = df_hist_full.groupby('StoreType')[col_hist].sum()
            distrib_tipo = distrib_tipo / distrib_tipo.sum()
            rows = []
            for stype, prop in distrib_tipo.items():
                temp = df_fc.copy()
                temp['StoreType'] = stype
                temp['Valor'] = temp['Valor']*prop
                rows.append(temp)
            df_fc_tipos = pd.concat(rows, ignore_index=True)
            df_fc = df_fc_tipos

        heat = df_fc.groupby(['StoreType','Week'])['Valor'].mean().reset_index()
        pivot = heat.pivot(index='StoreType', columns='Week', values='Valor')
        pivot = pivot.sort_index()
        fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                             colorscale='Viridis'))
        fig_heat.update_layout(template='plotly_white', xaxis_title='Semana', yaxis_title='Tipo de Loja',
                               height=300, margin=dict(l=40,r=40,t=30,b=40))

        return fig_global, fig_heat 