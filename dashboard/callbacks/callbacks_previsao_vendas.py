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
from ..utils import criar_figura_vazia


def deserializar_df(store_data):
    """
    Desserializa o DataFrame principal a partir do dcc.Store.
    """
    # Uso de cache no servidor quando store_data é dict
    if isinstance(store_data, dict) and 'modo' in store_data:
        modo = store_data.get('modo', 'amostras')
        n_amostras = store_data.get('n_amostras', 50)
        use_samples = (modo == 'amostras')
        df = get_principal_dataset(use_samples=use_samples, n_amostras=n_amostras)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    # Caso de JSON string
    if not store_data:
        return None
    df = pd.read_json(StringIO(store_data), orient='split')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


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


def registrar_callbacks_previsao_vendas(aplicativo, dados):
    """
    Registra callbacks para a página de Previsão de Vendas.
    """
    # Callback para popular dropdown de lojas
    @aplicativo.callback(
        Output('dropdown-lojas-previsao', 'options'),
        Input('armazenamento-df-principal', 'data')
    )
    def popula_lojas(store_data):
        df = deserializar_df(store_data)
        if df is None or 'Store' not in df.columns:
            return []
        lojas = sorted(df['Store'].unique())
        return [{'label': loja, 'value': loja} for loja in lojas]

    # Toggle de exibição dos parâmetros avançados conforme modelo
    @aplicativo.callback(
        Output('parametros-arima','style'),
        Output('parametros-xgboost','style'),
        Output('parametros-lightgbm','style'),
        Output('parametros-ensemble','style'),
        Input('dropdown-modelo-previsao','value')
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

    # Callback principal de previsão
    @aplicativo.callback(
        Output('grafico-previsao', 'figure'),
        Output('cards-metricas', 'children'),
        Output('tabela-previsao', 'children'),
        Input('radio-metrica-previsao', 'value'),
        Input('slider-horizonte-previsao', 'value'),
        Input('dropdown-granularidade-previsao', 'value'),
        Input('dropdown-lojas-previsao', 'value'),
        Input('date-picker-historico-previsao', 'start_date'),
        Input('date-picker-historico-previsao', 'end_date'),
        Input('dropdown-modelo-previsao', 'value'),
        State('arima-p','value'),
        State('arima-d','value'),
        State('arima-q','value'),
        State('xgb-estimators','value'),
        State('xgb-lr','value'),
        State('lgbm-estimators','value'),
        State('armazenamento-df-principal', 'data')
    )
    def gerar_previsao(target, horizonte, granularidade, lojas, data_inicio, data_fim, modelo, p, d, q, xgb_estimators, xgb_lr, lgbm_estimators, store_data):
        # Deserializar DataFrame principal
        df = deserializar_df(store_data)
        if df is None or df.empty:
            return criar_figura_vazia("Sem dados"), [], []

        # Filtrar por data e lojas
        if data_inicio or data_fim:
            df = filtrar_por_data(df, data_inicio, data_fim)
        if lojas:
            df = df[df['Store'].isin(lojas)]
        if df.empty:
            return criar_figura_vazia("Sem dados para os filtros selecionados"), [], []

        # Seleção de métrica conforme escolha do usuário
        col = 'Sales' if target == 'Sales' else 'Customers'
        # Preparar série temporal conforme granularidade
        ts = df[['Date', col]].copy()
        ts.rename(columns={col: 'Sales'}, inplace=True)
        if granularidade == 'diaria':
            ts = ts.groupby('Date').sum().reset_index()
            freq = 'D'
        elif granularidade == 'semanal':
            ts = ts.set_index('Date').resample('W').sum().reset_index()
            freq = 'W'
        else:  # mensal
            ts = ts.set_index('Date').resample('M').sum().reset_index()
            freq = 'M'

        # Previsão com ARIMA
        if modelo == 'arima':
            try:
                model = ARIMA(ts['Sales'], order=(int(p), int(d), int(q))).fit()
            except Exception as e:
                return criar_figura_vazia(f"Erro ARIMA: {e}"), [], []
            # Previsão futura e PACF
            future = pd.date_range(start=ts['Date'].max(), periods=horizonte+1, freq=freq)[1:]
            fc = model.get_forecast(steps=horizonte)
            pred = fc.predicted_mean
            conf = fc.conf_int()
            # Gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts['Date'], y=ts['Sales'], name='Histórico', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=future, y=pred, name='ARIMA', line=dict(color='purple')))
            fig.add_trace(go.Scatter(x=future, y=conf.iloc[:,1], fill=None, line=dict(color='lavender'), showlegend=False))
            fig.add_trace(go.Scatter(x=future, y=conf.iloc[:,0], fill='tonexty', line=dict(color='lavender'), name='ARIMA IC'))
            fig.update_layout(title='Previsão de Vendas (ARIMA)', xaxis_title='Data', yaxis_title='Vendas')
            # Métricas in-sample
            y_true = ts['Sales'].values
            y_pred_fit = model.predict(start=0, end=len(y_true)-1)
            mae = mean_absolute_error(y_true, y_pred_fit)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_fit))
            mape = np.mean(np.abs((y_true - y_pred_fit)/ y_true))*100
            cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAE'), html.P(f"{mae:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('RMSE'), html.P(f"{rmse:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAPE'), html.P(f"{mape:.2f}%")])) , md=4),
            ], className='g-4 mt-3')
            df_table = pd.DataFrame({'Date':future, 'Forecast':pred.values})
            table = dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)
            return fig, cards, table
        # Previsão com Prophet usando cache
        if modelo == 'prophet':
            df_ts = ts.rename(columns={'Date': 'ds', 'Sales': 'y'})
            df_ts_json = df_ts.to_json(date_format='iso', orient='split')
            # Chama helper memoizado (fit apenas uma vez)
            forecast_json = gerar_forecast_prophet(df_ts_json, freq)
            forecast = pd.read_json(StringIO(forecast_json), orient='split')
            # Converter ds para datetime para comparações e gráfico
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            # Construir gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts['Date'], y=ts['Sales'], name='Histórico'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, line=dict(color='lightgrey'), name='Upper'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', line=dict(color='lightgrey'), name='Lower'))
            fig.update_layout(title='Previsão de Vendas (Prophet)', xaxis_title='Data', yaxis_title='Vendas')
            # Cálculo de métricas no histórico
            y_true = df_ts['y'].values
            y_pred = forecast.loc[forecast['ds'] <= df_ts['ds'].max(), 'yhat'].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            # Cards de métricas
            cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAE'), html.P(f"{mae:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('RMSE'), html.P(f"{rmse:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAPE'), html.P(f"{mape:.2f}%")])) , md=4),
            ], className='g-4 mt-3')
            # Tabela de previsões
            df_table = forecast[['ds', 'yhat']].tail(horizonte).rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
            table = dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)
            return fig, cards, table
        # Previsão com Random Forest usando lags e features de data
        if modelo == 'random_forest':
            # Extrair lags e features de data
            ts['DayOfWeek'] = ts['Date'].dt.dayofweek
            ts['Month'] = ts['Date'].dt.month
            # Definir lags conforme granularidade
            lags = [1, 7] if granularidade == 'diaria' else [1]
            y = ts['Sales'].values
            n_lags = max(lags)
            if len(y) <= n_lags:
                return criar_figura_vazia('Dados insuficientes para Random Forest'), [], []
            # Construir dataset supervisionado
            X_sup, y_sup = [], []
            for i in range(n_lags, len(ts)):
                row = [y[i - lag] for lag in lags]
                row += [ts.at[i, 'DayOfWeek'], ts.at[i, 'Month']]
                X_sup.append(row)
                y_sup.append(y[i])
            X_sup, y_sup = np.array(X_sup), np.array(y_sup)
            # Treinar modelo
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_sup, y_sup)
            # Previsão histórica para métricas
            y_pred_hist = rf.predict(X_sup)
            # Gerar previsões recursivas
            preds, history = [], list(y)
            future_dates = pd.date_range(start=ts['Date'].max(), periods=horizonte + 1, freq=freq)[1:]
            for dt in future_dates:
                # calcular features de lags e data
                base = [history[-lag] for lag in lags]
                base += [dt.dayofweek, dt.month]
                yhat = rf.predict(np.array(base).reshape(1, -1))[0]
                preds.append(yhat)
                history.append(yhat)
            # Construir gráfico
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts['Date'], y=ts['Sales'], name='Histórico'))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name='Previsão'))
            fig.update_layout(title='Previsão de Vendas (Random Forest)', xaxis_title='Data', yaxis_title='Vendas')
            # Calcular métricas
            mae = mean_absolute_error(y_sup, y_pred_hist)
            rmse = np.sqrt(mean_squared_error(y_sup, y_pred_hist))
            mape = np.mean(np.abs((y_sup - y_pred_hist) / y_sup)) * 100
            # Cards de métricas
            cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAE'), html.P(f"{mae:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('RMSE'), html.P(f"{rmse:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAPE'), html.P(f"{mape:.2f}%")])) , md=4),
            ], className='g-4 mt-3')
            # Tabela de previsões
            df_table = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
            table = dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)
            return fig, cards, table
        # Previsão com XGBoost usando lags e features de data
        if modelo == 'xgboost':
            ts['DayOfWeek'] = ts['Date'].dt.dayofweek
            ts['Month'] = ts['Date'].dt.month
            lags = [1, 7] if granularidade == 'diaria' else [1]
            y = ts['Sales'].values
            n_lags = max(lags)
            if len(y) <= n_lags:
                return criar_figura_vazia('Dados insuficientes para XGBoost'), [], []
            X_sup, y_sup = [], []
            for i in range(n_lags, len(ts)):
                row = [y[i - lag] for lag in lags]
                row += [ts.at[i, 'DayOfWeek'], ts.at[i, 'Month']]
                X_sup.append(row)
                y_sup.append(y[i])
            X_sup, y_sup = np.array(X_sup), np.array(y_sup)
            try:
                xgb = XGBRegressor(n_estimators=int(xgb_estimators), learning_rate=float(xgb_lr), objective='reg:squarederror', random_state=42)
                xgb.fit(X_sup, y_sup)
            except Exception as e:
                return criar_figura_vazia(f"Erro XGBoost: {e}"), [], []
            y_pred_hist = xgb.predict(X_sup)
            preds, history = [], list(y)
            future_dates = pd.date_range(start=ts['Date'].max(), periods=horizonte + 1, freq=freq)[1:]
            for dt in future_dates:
                base = [history[-lag] for lag in lags] + [dt.dayofweek, dt.month]
                yhat = xgb.predict(np.array(base).reshape(1, -1))[0]
                preds.append(yhat)
                history.append(yhat)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts['Date'], y=ts['Sales'], name='Histórico'))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name='XGBoost'))
            fig.update_layout(title='Previsão de Vendas (XGBoost)', xaxis_title='Data', yaxis_title='Vendas')
            mae = mean_absolute_error(y_sup, y_pred_hist)
            rmse = np.sqrt(mean_squared_error(y_sup, y_pred_hist))
            mape = np.mean(np.abs((y_sup - y_pred_hist) / y_sup)) * 100
            cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAE'), html.P(f"{mae:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('RMSE'), html.P(f"{rmse:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAPE'), html.P(f"{mape:.2f}%")])) , md=4),
            ], className='g-4 mt-3')
            df_table = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
            table = dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)
            return fig, cards, table
        # Previsão com LightGBM usando lags e features de data
        if modelo == 'lightgbm':
            ts['DayOfWeek'] = ts['Date'].dt.dayofweek
            ts['Month'] = ts['Date'].dt.month
            lags = [1, 7] if granularidade == 'diaria' else [1]
            y = ts['Sales'].values
            n_lags = max(lags)
            if len(y) <= n_lags:
                return criar_figura_vazia('Dados insuficientes para LightGBM'), [], []
            X_sup, y_sup = [], []
            for i in range(n_lags, len(ts)):
                row = [y[i - lag] for lag in lags]
                row += [ts.at[i, 'DayOfWeek'], ts.at[i, 'Month']]
                X_sup.append(row)
                y_sup.append(y[i])
            X_sup, y_sup = np.array(X_sup), np.array(y_sup)
            try:
                lgbm = LGBMRegressor(n_estimators=int(lgbm_estimators), random_state=42)
                lgbm.fit(X_sup, y_sup)
            except Exception as e:
                return criar_figura_vazia(f"Erro LightGBM: {e}"), [], []
            y_pred_hist = lgbm.predict(X_sup)
            preds, history = [], list(y)
            future_dates = pd.date_range(start=ts['Date'].max(), periods=horizonte + 1, freq=freq)[1:]
            for dt in future_dates:
                base = [history[-lag] for lag in lags] + [dt.dayofweek, dt.month]
                yhat = lgbm.predict(np.array(base).reshape(1, -1))[0]
                preds.append(yhat)
                history.append(yhat)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts['Date'], y=ts['Sales'], name='Histórico'))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name='LightGBM'))
            fig.update_layout(title='Previsão de Vendas (LightGBM)', xaxis_title='Data', yaxis_title='Vendas')
            mae = mean_absolute_error(y_sup, y_pred_hist)
            rmse = np.sqrt(mean_squared_error(y_sup, y_pred_hist))
            mape = np.mean(np.abs((y_sup - y_pred_hist) / y_sup)) * 100
            cards = dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAE'), html.P(f"{mae:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('RMSE'), html.P(f"{rmse:.2f}")])) , md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H5('MAPE'), html.P(f"{mape:.2f}%")])) , md=4),
            ], className='g-4 mt-3')
            df_table = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
            table = dbc.Table.from_dataframe(df_table, striped=True, bordered=True, hover=True)
            return fig, cards, table
        # Modelo não implementado ou inválido
        fig = criar_figura_vazia(f'Modelo {modelo} não implementado')
        return fig, [], []

    # Download CSV de previsões
    @aplicativo.callback(
        Output('download-dataframe','data'),
        Input('btn-download-csv','n_clicks'),
        State('slider-horizonte-previsao','value'),
        State('dropdown-granularidade-previsao','value'),
        State('dropdown-lojas-previsao','value'),
        State('date-picker-historico-previsao','start_date'),
        State('date-picker-historico-previsao','end_date'),
        State('dropdown-modelo-previsao','value'),
        State('armazenamento-df-principal','data'),
        prevent_initial_call=True
    )
    def download_csv(n_clicks, horizonte, granularidade, lojas, data_inicio, data_fim, modelo, store_data):
        df = deserializar_df(store_data)
        if df is None or df.empty:
            return dash.no_update
        if data_inicio or data_fim:
            df = filtrar_por_data(df, data_inicio, data_fim)
        if lojas:
            df = df[df['Store'].isin(lojas)]
        ts = df[['Date','Sales']].copy()
        if granularidade == 'diaria':
            ts = ts.groupby('Date').sum().reset_index(); freq='D'
        elif granularidade == 'semanal':
            ts = ts.set_index('Date').resample('W').sum().reset_index(); freq='W'
        else:
            ts = ts.set_index('Date').resample('M').sum().reset_index(); freq='M'
        if modelo == 'prophet':
            df_ts = ts.rename(columns={'Date':'ds','Sales':'y'})
            forecast_json = gerar_forecast_prophet(df_ts.to_json(date_format='iso',orient='split'), freq)
            forecast = pd.read_json(StringIO(forecast_json),orient='split')
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            df_table = forecast[['ds','yhat']].tail(horizonte).rename(columns={'ds':'Date','yhat':'Forecast'})
        else:
            # RF supervisionado com lags
            # Reusar lógica interna conforme callback principal
            # Construir lags e dataset
            if granularidade == 'diaria':
                lags=[1,7]
            else:
                lags=[1]
            y=ts['Sales'].values; n_lags=max(lags)
            X_sup=[ [y[i-l] for l in lags] for i in range(n_lags,len(y)) ]
            y_sup=[y[i] for i in range(n_lags,len(y))]
            rf=RandomForestRegressor(n_estimators=100,random_state=42)
            rf.fit(np.array(X_sup),np.array(y_sup))
            history=list(y)
            future_dates=pd.date_range(start=ts['Date'].max(),periods=horizonte+1,freq=freq)[1:]
            preds=[]
            for _ in range(horizonte):
                Xp=np.array([history[-l] for l in lags]).reshape(1,-1)
                p=rf.predict(Xp)[0]; preds.append(p); history.append(p)
            df_table=pd.DataFrame({'Date':future_dates,'Forecast':preds})
        return dcc.send_data_frame(df_table.to_csv, filename='previsoes.csv', index=False)

    # Download PNG do gráfico
    @aplicativo.callback(
        Output('download-graph','data'),
        Input('btn-download-png','n_clicks'),
        State('grafico-previsao','figure'),
        prevent_initial_call=True
    )
    def download_graph(n_clicks, figure):
        fig = go.Figure(figure)
        try:
            img_bytes = pio.to_image(fig, format='png')
            return dcc.send_bytes(img_bytes, filename='grafico_previsao.png')
        except ValueError:
            # Kaleido não instalado ou erro na exportação de imagem
            print("Erro ao exportar imagem: instale kaleido com 'pip install kaleido'")
            return dash.no_update 

    # Fim Random Forest
    # Callback para comparar modelos e mostrar feature importance
    @aplicativo.callback(
        Output('grafico-previsao-agregado', 'figure'),
        Output('grafico-feature-importance', 'figure'),
        Input('checklist-modelos', 'value'),
        Input('slider-horizonte-previsao', 'value'),
        Input('dropdown-granularidade-previsao', 'value'),
        Input('dropdown-lojas-previsao', 'value'),
        Input('date-picker-historico-previsao', 'start_date'),
        Input('date-picker-historico-previsao', 'end_date'),
        State('armazenamento-df-principal', 'data')
    )
    def atualizar_comparativo(modelos, horizonte, granularidade, lojas, data_inicio, data_fim, store_data):
        df = deserializar_df(store_data)
        if df is None or df.empty:
            return criar_figura_vazia('Sem dados'), criar_figura_vazia('')
        # Filtrar e preparar série
        if data_inicio or data_fim:
            df = filtrar_por_data(df, data_inicio, data_fim)
        if lojas:
            df = df[df['Store'].isin(lojas)]
        ts = df[['Date','Sales']].copy()
        ts['Date'] = pd.to_datetime(ts['Date'])
        if granularidade == 'diaria':
            ts = ts.groupby('Date').sum().reset_index(); freq='D'
        elif granularidade == 'semanal':
            ts = ts.set_index('Date').resample('W').sum().reset_index(); freq='W'
        else:
            ts = ts.set_index('Date').resample('M').sum().reset_index(); freq='M'
        # Iniciar figura agregada
        fig_agg = go.Figure()
        fig_agg.add_trace(go.Scatter(x=ts['Date'], y=ts['Sales'], name='Histórico', line=dict(color='black')))
        # Prophet
        if 'prophet' in modelos:
            df_p = ts.rename(columns={'Date':'ds','Sales':'y'})
            forecast_json = gerar_forecast_prophet(df_p.to_json(date_format='iso',orient='split'), freq)
            fore_p = pd.read_json(StringIO(forecast_json), orient='split')
            fore_p['ds'] = pd.to_datetime(fore_p['ds'])
            # Fatiar apenas os períodos futuros conforme slider
            hist_end = ts['Date'].max()
            fore_p = fore_p[fore_p['ds'] > hist_end].iloc[:horizonte]
            # bandas
            fig_agg.add_trace(go.Scatter(x=fore_p['ds'], y=fore_p['yhat'], name='Prophet', line=dict(color='blue')))
            fig_agg.add_trace(go.Scatter(x=fore_p['ds'], y=fore_p['yhat_upper'], fill=None, line=dict(color='lightblue'), showlegend=False))
            fig_agg.add_trace(go.Scatter(x=fore_p['ds'], y=fore_p['yhat_lower'], fill='tonexty', line=dict(color='lightblue'), name='Prophet IC'))
        # Random Forest
        fig_imp = criar_figura_vazia('')
        if 'random_forest' in modelos:
            # preparar features
            rfts = ts.copy()
            rfts['DayOfWeek'] = rfts['Date'].dt.dayofweek
            rfts['Month'] = rfts['Date'].dt.month
            lags = [1,7] if granularidade=='diaria' else [1]
            y = rfts['Sales'].values; n_lags=max(lags)
            X, Y = [], []
            for i in range(n_lags, len(rfts)):
                row = [y[i-l] for l in lags] + [rfts.at[i,'DayOfWeek'], rfts.at[i,'Month']]
                X.append(row); Y.append(y[i])
            X, Y = np.array(X), np.array(Y)
            rf = RandomForestRegressor(n_estimators=100,random_state=42)
            rf.fit(X, Y)
            # feature importance
            feats = [f'lag_{l}' for l in lags] + ['DayOfWeek','Month']
            fig_imp = go.Figure([go.Bar(x=feats, y=rf.feature_importances_, marker_color='green')])
            fig_imp.update_layout(title='Feature Importance (RF)')
            # previsões medianas e bandas
            future = pd.date_range(start=ts['Date'].max(), periods=horizonte+1, freq=freq)[1:]
            preds = []
            lowers, uppers = [], []
            hist = list(rfts['Sales'])
            for dt in future:
                base = [hist[-l] for l in lags] + [dt.dayofweek, dt.month]
                arr = [est.predict(np.array(base).reshape(1,-1))[0] for est in rf.estimators_]
                med = np.median(arr); lo=np.percentile(arr,2.5); hi=np.percentile(arr,97.5)
                preds.append(med); lowers.append(lo); uppers.append(hi); hist.append(med)
            fig_agg.add_trace(go.Scatter(x=future, y=preds, name='RF', line=dict(color='red')))
            fig_agg.add_trace(go.Scatter(x=future, y=uppers, fill=None, line=dict(color='pink'), showlegend=False))
            fig_agg.add_trace(go.Scatter(x=future, y=lowers, fill='tonexty', line=dict(color='pink'), name='RF IC'))
        fig_agg.update_layout(title='Comparativo de Modelos', xaxis_title='Data', yaxis_title='Vendas')
        return fig_agg, fig_imp 