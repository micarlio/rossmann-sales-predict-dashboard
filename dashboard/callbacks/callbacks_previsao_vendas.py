
# dashboard/callbacks/callbacks_previsao_vendas.py

from dash.dependencies import Input, Output, State
from dash import dcc, html, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import os
from io import StringIO
from prophet import Prophet
# Ajuste de performance: cache para Prophet até horizonte fixo
MAX_HORIZON = 365  # 52 semanas (1 ano)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import functools
import dash
import logging
import json
from datetime import datetime
import glob
from plotly.subplots import make_subplots

from ..core.utils import criar_figura_vazia, parse_json_to_df, criar_icone_informacao
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
        Output('icone-toggle-filtros', 'className'),
        Input('btn-toggle-filtros', 'n_clicks'),
        State('collapse-filtros', 'is_open')
    )
    def toggle_collapse_filtros(n, is_open):
        if n:
            novo_estado = not is_open
            novo_icone = "fas fa-chevron-down" if novo_estado else "fas fa-chevron-up"
            return novo_estado, novo_icone
        return is_open, "fas fa-chevron-up"

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

        if 'todos' in tipos_selecionados and len(tipos_selecionados) > 1:
            ultimo = dash.callback_context.triggered[0]['value'][-1]
            return ['todos'] if ultimo == 'todos' else [t for t in tipos_selecionados if t != 'todos']

        return tipos_selecionados

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
            Input('dropdown-tipo-loja', 'value'),
            Input('dropdown-lojas-previsao', 'value'),
            Input('dropdown-promocao', 'value'),
            Input('checklist-dias-semana', 'value'),
            Input('armazenamento-df-principal', 'data')
        ]
    )
    def gerar_previsao(target, horizonte, granularidade, 
                       tipo_loja, lojas, promocao, dias_semana, store_data):
        logger.info(f"Callback de previsão iniciado. Modelo: prophet, Tipos: {tipo_loja}, Lojas: {lojas}")
        
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

        col, metrica_nome = {
            'Sales': ('Sales', 'Vendas'),
            'Customers': ('Customers', 'Clientes'),
            'SalesPerCustomer': ('TicketMedio', 'Ticket Médio'),
        }.get(target, ('Sales', 'Vendas'))
            
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
                logger.warning(f"Dados insuficientes para o modelo prophet")
                return criar_figura_vazia(f"Dados insuficientes para o modelo prophet"), [], html.Div(), None, None, None
                
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
                'prophet',
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
                        'prophet',
                        holidays=holidays
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

            fig.update_layout(
                height=400,
                xaxis_rangeslider_visible=True,
                xaxis_rangeslider_thickness=0.1
            )
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
                title=f'Previsão de {metrica_nome}',
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
            desc_map = {
                "Total Previsto": "Soma das previsões de vendas para todo o horizonte selecionado.",
                "Média Prevista": "Média diária das vendas previstas no período.",
                "Variação vs. Hist. (%)": "Diferença percentual entre a média prevista e a média histórica do mesmo período.",
                "Variação Acumulada (%)": "Crescimento percentual da primeira até a última previsão no horizonte.",
                "Melhor Dia da Semana": "Dia da semana cuja média de vendas previstas é a mais alta no horizonte.",
            }

            items = [
                (total_prev_label, format_currency(total_prev, metrica_selecionada), None, desc_map.get(total_prev_label, "")),
                ("Variação vs. Hist. (%)", f"{var_perc:,.2f}%", "text-success" if var_perc>=0 else "text-danger", desc_map["Variação vs. Hist. (%)"]),
                ("Variação Acumulada (%)", f"{var_acum:,.2f}%", "text-success" if var_acum>=0 else "text-danger", desc_map["Variação Acumulada (%)"]),
                ("Melhor Dia da Semana", top_weekday, None, desc_map["Melhor Dia da Semana"]),
            ]
            if metrica_selecionada != 'SalesPerCustomer':
                items.insert(1, ("Média Prevista", format_currency(media_prev, metrica_selecionada), None, desc_map["Média Prevista"]))

            items_divs = []
            for idx, item in enumerate(items):
                titulo, valor, classe_extra, tooltip_text = item
                estilo_valor = "fw-bold " + (classe_extra if classe_extra else "")

                tooltip_id = f"kpi-tooltip-{idx}"

                items_divs.append(
                    html.Div([
                        # Ícone posicionado no canto superior esquerdo
                        html.Span(
                            criar_icone_informacao(tooltip_id, tooltip_text),
                            className="position-absolute top-0 start-0 ms-2 mt-2"
                        ),
                        # Conteúdo centralizado
                        html.Div([
                            html.P(titulo, className="text-muted mb-1 small"),
                            html.H5(valor, className=estilo_valor)
                        ], className="d-flex flex-column align-items-center justify-content-center h-100")
                    ], className="info-item position-relative", style={"min-height": "90px"})
                )

            return html.Div(items_divs, className="info-panel-grid")
        except Exception as e:
            return [html.Div(f"Erro ao gerar informações: {e}")]
    
    # ==================================================================
    # NOVOS GRÁFICOS (gerados a partir do grafico-previsao já existente)
    # ==================================================================

    # Callback para gráficos adicionais substituído por uma nova versão abaixo

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
        Input('armazenamento-df-principal', 'data'),
        Input('armazenamento-forecast-diario', 'data'),
        Input('armazenamento-hist-diario', 'data')
    )
    def gerar_graficos_adicionais(fig_previsao, metrica_selecionada, store_data, forecast_diario_json, hist_diario_json):
        """Cria dois gráficos auxiliares:
        1) Média da métrica selecionada por semana (histórico e previsão em um gráfico contínuo)
        2) Heatmap de vendas previstas por Semana × Tipo de Loja.
        """
        # importações duplicadas removidas (já declaradas no topo)
        if not fig_previsao or 'data' not in fig_previsao or len(fig_previsao['data']) < 2:
            placeholder = go.Figure()
            placeholder.update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False,
                                      annotations=[dict(text="Aguardando geração da previsão", showarrow=False)])
            return placeholder, placeholder

        # --- Dados da previsão principal ---
        dates_fc = pd.to_datetime(fig_previsao['data'][1]['x'])
        values_fc = np.array(fig_previsao['data'][1]['y'], dtype=float)
        df_fc = pd.DataFrame({'Date': dates_fc, 'Valor': values_fc})
        
        # --- Carregar histórico e previsão diária ---
        df_hist_full = deserializar_df(store_data)
        
        # Se temos os dados diários completos, vamos usá-los
        has_daily_data = hist_diario_json is not None and forecast_diario_json is not None
        
        if has_daily_data:
            try:
                # Carrega dados históricos e de previsão DIÁRIOS
                df_hist_daily = pd.read_json(StringIO(hist_diario_json), orient='split')
                df_fc_daily = pd.read_json(StringIO(forecast_diario_json), orient='split')

                # Garante que as colunas de data sejam do tipo datetime
                df_hist_daily['ds'] = pd.to_datetime(df_hist_daily['ds'])
                df_fc_daily['ds'] = pd.to_datetime(df_fc_daily['ds'])
                
                # Renomeia colunas para padronização
                df_hist_daily = df_hist_daily.rename(columns={'ds': 'Date', 'y': 'Valor'})
                df_fc_daily = df_fc_daily.rename(columns={'ds': 'Date', 'yhat': 'Valor'})
                
                # Marca origem dos dados
                df_hist_daily['Origem'] = 'Histórico'
                df_fc_daily['Origem'] = 'Previsão'
                
                # Combina histórico e previsão
                df_combined = pd.concat([df_hist_daily, df_fc_daily], ignore_index=True)
            except Exception as e:
                logger.error(f"Erro ao processar dados diários: {e}")
                has_daily_data = False
        
        if not has_daily_data or df_hist_full is None or df_hist_full.empty:
            df_hist_full = pd.DataFrame(columns=['Date','Sales','Customers'])
            df_combined = pd.DataFrame(columns=['Date','Valor','Origem'])
        
        # Define coluna a usar e nome da métrica
        if metrica_selecionada == 'Sales':
            col_hist = 'Sales'
            metrica_nome = 'Vendas'
        elif metrica_selecionada == 'Customers':
            col_hist = 'Customers'
            metrica_nome = 'Clientes'
        else:  # Ticket
            df_hist_full['TicketMedio'] = (df_hist_full['Sales'] / df_hist_full['Customers'].replace(0, np.nan)).fillna(0)
            col_hist = 'TicketMedio'
            metrica_nome = 'Ticket Médio'

        # --- Preparar dados para gráfico de média semanal ---
        if has_daily_data and not df_combined.empty:
            # Adicionar semana e ano
            df_combined['Week'] = df_combined['Date'].dt.isocalendar().week
            df_combined['Year'] = df_combined['Date'].dt.year
            
            # Calcular médias semanais
            weekly_data = df_combined.groupby(['Year', 'Week', 'Origem'])['Valor'].mean().reset_index()
            
            # Criar índice de data para ordenação correta (primeiro dia da semana)
            weekly_data['WeekStart'] = weekly_data.apply(
                lambda row: pd.Timestamp(f"{row['Year']}-01-01") + 
                           pd.DateOffset(weeks=int(row['Week'])-1),
                axis=1
            )
            
            # Separar histórico e previsão
            hist_weekly = weekly_data[weekly_data['Origem'] == 'Histórico']
            fc_weekly = weekly_data[weekly_data['Origem'] == 'Previsão']
        else:
            # Criar DataFrames vazios para evitar erros
            hist_weekly = pd.DataFrame(columns=['WeekStart', 'Valor'])
            fc_weekly = pd.DataFrame(columns=['WeekStart', 'Valor'])

        # Criar figura para média semanal
        fig_global = go.Figure()
        
        # Adicionar linha de histórico
        if not hist_weekly.empty:
            # Ordenar por data para garantir a continuidade da linha
            hist_weekly = hist_weekly.sort_values('WeekStart')
            
            # Aplicar suavização usando média móvel
            if len(hist_weekly) > 5:
                hist_weekly['Valor_Suavizado'] = hist_weekly['Valor'].rolling(window=3, center=True).mean().fillna(hist_weekly['Valor'])
            else:
                hist_weekly['Valor_Suavizado'] = hist_weekly['Valor']
                
            fig_global.add_trace(go.Scatter(
                x=hist_weekly['WeekStart'], 
                y=hist_weekly['Valor_Suavizado'],
                mode='lines', 
                name='Histórico', 
                line=dict(color='#1f77b4', width=2.5, shape='spline', smoothing=1.3)
            ))
        
        # Adicionar linha de previsão
        if not fc_weekly.empty:
            # Ordenar por data para garantir a continuidade da linha
            fc_weekly = fc_weekly.sort_values('WeekStart')
            
            # Aplicar suavização usando média móvel
            if len(fc_weekly) > 5:
                fc_weekly['Valor_Suavizado'] = fc_weekly['Valor'].rolling(window=3, center=True).mean().fillna(fc_weekly['Valor'])
            else:
                fc_weekly['Valor_Suavizado'] = fc_weekly['Valor']
                
            fig_global.add_trace(go.Scatter(
                x=fc_weekly['WeekStart'], 
                y=fc_weekly['Valor_Suavizado'],
                mode='lines', 
                name='Previsão', 
                line=dict(color='#ff7f0e', width=3, shape='spline', smoothing=1.3)
            ))
            
            # Adicionar área sombreada entre as curvas se ambas existirem
            if not hist_weekly.empty:
                # Encontrar o ponto de junção entre histórico e previsão
                junction_date = fc_weekly['WeekStart'].min()
                
                # Obter o último ponto do histórico próximo à junção
                last_hist_point = hist_weekly[hist_weekly['WeekStart'] <= junction_date]
                if not last_hist_point.empty:
                    last_hist_point = last_hist_point.iloc[-1]
                    
                    # Obter o primeiro ponto da previsão
                    first_fc_point = fc_weekly[fc_weekly['WeekStart'] >= junction_date]
                    if not first_fc_point.empty:
                        first_fc_point = first_fc_point.iloc[0]
                        
                        # Adicionar um ponto de conexão suave
                        connection_x = [last_hist_point['WeekStart'], first_fc_point['WeekStart']]
                        connection_y = [last_hist_point['Valor_Suavizado'], first_fc_point['Valor_Suavizado']]
                        
                        fig_global.add_trace(go.Scatter(
                            x=connection_x,
                            y=connection_y,
                            mode='lines',
                            line=dict(color='#9467bd', width=2, dash='dot'),
                            name='Conexão',
                            showlegend=False
                        ))
        
        # Configurar layout
        fig_global.update_layout(
            template='plotly_white', 
            xaxis_title='Data',
            yaxis_title=f'Média Semanal de {metrica_nome}',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300, 
            margin=dict(l=40, r=40, t=30, b=40),
            hovermode='x unified',
            xaxis_rangeslider=dict(visible=True, thickness=0.05)
        )
        
        # Definir intervalo de visualização para mostrar parte do histórico e toda a previsão
        if not hist_weekly.empty and not fc_weekly.empty:
            # Mostrar os últimos 3 meses de histórico e toda a previsão
            data_inicio_visual = fc_weekly['WeekStart'].min() - pd.DateOffset(months=3)
            data_fim_visual = fc_weekly['WeekStart'].max()
            fig_global.update_xaxes(range=[data_inicio_visual, data_fim_visual])

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

        # Adicionar semana e ano para o heatmap
        df_fc['Week'] = df_fc['Date'].dt.isocalendar().week
        df_fc['Year'] = df_fc['Date'].dt.year
        
        heat = df_fc.groupby(['StoreType','Week'])['Valor'].mean().reset_index()
        pivot = heat.pivot(index='StoreType', columns='Week', values='Valor')
        pivot = pivot.sort_index()
        fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index,
                                             colorscale='Viridis'))
        fig_heat.update_layout(template='plotly_white', xaxis_title='Semana', yaxis_title='Tipo de Loja',
                               height=300, margin=dict(l=40,r=40,t=30,b=40))

        return fig_global, fig_heat 