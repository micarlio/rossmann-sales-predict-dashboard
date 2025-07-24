#Funções auxiliares de previsão usadas em toda a aplicação.

from __future__ import annotations
import logging # Para logs
from typing import Optional, Tuple # Para tipos

# Bibliotecas
import numpy as np
import pandas as pd
from prophet import Prophet
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

__all__ = [
    "create_date_features",
    "train_and_forecast",
    "create_lag_features",
    "preprocess_prophet_ts",
]

# ------------------------------------------------------------------
# FUNÇÃO DE PRÉ-PROCESSAMENTO PARA O PROPHET
# ------------------------------------------------------------------

def preprocess_prophet_ts(ts: pd.DataFrame) -> pd.DataFrame:
    """Limpa a série antes de treinar o Prophet.

    1. Converte valores <= 0 em ``NaN`` (loja fechada ou erro).
    2. Interpola valores ausentes linearmente.
    3. Remove linhas ainda faltantes.
    """
    ts_clean = ts.copy()
    # Mantém valores iguais a 0 (ex.: loja fechada), mas continua removendo negativos
    ts_clean.loc[ts_clean["y"] < 0, "y"] = np.nan
    # Interpolação linear e preenchimento inicial/final com método de borda
    ts_clean["y"] = ts_clean["y"].interpolate(method="linear").ffill().bfill()
    return ts_clean.dropna()


def create_date_features(df: pd.DataFrame, date_col: str = "ds") -> pd.DataFrame:
    """Adiciona colunas temporais clássicas e *flags* úteis.

    Colunas adicionadas
    -------------------
    mes, dia_semana, dia_mes, semana_ano,
    is_weekend, is_month_start, is_month_end, trimestre
    """
    df_feat = df.copy()
    dt = df_feat[date_col]
    df_feat["mes"] = dt.dt.month
    df_feat["dia_semana"] = dt.dt.dayofweek
    df_feat["dia_mes"] = dt.dt.day
    df_feat["semana_ano"] = dt.dt.isocalendar().week.astype(int)
    df_feat["trimestre"] = dt.dt.quarter
    df_feat["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df_feat["is_month_start"] = dt.dt.is_month_start.astype(int)
    df_feat["is_month_end"] = dt.dt.is_month_end.astype(int)
    return df_feat


def create_lag_features(
    df: pd.DataFrame,
    *,
    lags: list[int] | None = None,
    roll_windows: list[int] | None = None,
) -> pd.DataFrame:
    """Adiciona colunas de *lags* e *rolling means* da série ``y``.

    Parâmetros
    ----------
    df : DataFrame com coluna ``y``
    lags : lista de lags (dias)
    roll_windows : tamanhos de janelas para médias móveis
    """
    lags = lags or [1, 3, 7, 14, 28]
    roll_windows = roll_windows or [7, 28]

    df_out = df.copy()
    for lag in lags:
        df_out[f"lag_{lag}"] = df_out["y"].shift(lag)

    # Rolling means usando valores anteriores (shift 1) para evitar look-ahead
    for w in roll_windows:
        df_out[f"roll_mean_{w}"] = df_out["y"].shift(1).rolling(window=w).mean()
    return df_out


def preprocess_prophet_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pré-processamento básico para Prophet:
    - Remove valores nulos em 'ds' ou 'y'.
    - Ordena por data.
    """
    df = df.copy()
    df = df.dropna(subset=["ds", "y"])
    df = df.sort_values("ds")
    return df


def forecast_with_prophet(
    df: pd.DataFrame,
    horizon_weeks: int = 52,
    freq: str = "W",
    holidays: Optional[pd.DataFrame] = None,
    extra_regressors: Optional[dict] = None,
    prophet_params: Optional[dict] = None,
) -> tuple[pd.DataFrame, Prophet]:
    """
    Treina e faz previsão com Prophet de forma simples e clara.

    Parâmetros:
    - df: DataFrame com colunas 'ds' (datas) e 'y' (valores)
    - horizon_weeks: horizonte de previsão em semanas
    - freq: frequência da previsão ('W' para semanal)
    - holidays: DataFrame de feriados (opcional)
    - extra_regressors: dicionário {nome: série} de regressores externos (opcional)
    - prophet_params: parâmetros extras para o Prophet (opcional)

    Retorna:
    - forecast: DataFrame com previsões futuras
    - model: objeto Prophet treinado
    """
    df = preprocess_prophet_data(df)
    prophet_params = prophet_params or {}
    m = Prophet(holidays=holidays, **prophet_params)

    # Adiciona regressores externos, se houver
    if extra_regressors:
        for reg_name, reg_values in extra_regressors.items():
            m.add_regressor(reg_name)
            df[reg_name] = reg_values

    m.fit(df)

    # Cria datas futuras
    last_date = df["ds"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=horizon_weeks, freq=freq)
    future = pd.DataFrame({"ds": future_dates})

    # Adiciona regressores externos futuros, se houver
    if extra_regressors:
        for reg_name in extra_regressors.keys():
            # Preenche com o último valor conhecido
            last_val = df[reg_name].iloc[-1]
            future[reg_name] = last_val

    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], m


def train_and_forecast(
    ts: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
    modelo: str,
    *,
    holidays: Optional[pd.DataFrame] = None,
    xgb_params: Optional[dict] = None,
    lgbm_params: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], int, str]:
    """Treina o modelo escolhido e devolve o *forecast*.

    Parâmetros
    ----------
    ts: DataFrame com colunas ``ds`` e ``y``
    future_dates: datas futuras para previsão
    modelo: 'prophet', 'random_forest', 'xgboost', 'lightgbm' ou 'ensemble'

    Retorna
    -------
    forecast, pred_train, deviation_count, model_name_display
    """
    modelo = (modelo or "").lower()
    xgb_params = xgb_params or {}
    lgbm_params = lgbm_params or {}

    pred_train: Optional[np.ndarray] = None
    deviation_count = 0

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------
    if modelo == "random_forest":
        model_name_display = "Random Forest"
        
        # Pré-processamento
        ts_clean = ts.copy()
        # Mantém zeros para que o modelo aprenda quedas provocadas por feriados
        ts_clean.loc[ts_clean["y"] < 0, "y"] = np.nan
        ts_clean["y"] = ts_clean["y"].interpolate(method="linear").ffill().bfill()
        ts_clean = ts_clean.dropna()
        
        if ts_clean.empty:
            raise ValueError("Série vazia após pré-processamento.")
            
        # Verificar se estamos lidando com uma loja individual ou um grupo
        is_single_store = len(ts_clean) < 500  # Heurística para identificar loja individual
        
        # Transformação logarítmica para estabilizar a variância
        log_transform = True
        if log_transform:
            epsilon = 1e-3
            ts_clean['y_original'] = ts_clean['y'].copy()
            ts_clean['y'] = np.log1p(ts_clean['y'])
            
        # Features de data e lag
        ts_with_features = create_date_features(ts_clean)
        
        if is_single_store:
            ts_with_features = create_lag_features(
                ts_with_features,
                lags=[1, 2, 3, 7],
                roll_windows=[7, 14]
            )
            ts_with_features = ts_with_features.dropna()
            
        # Preparar features para treinamento
        feature_columns = [
            'mes', 'dia_semana', 'dia_mes', 'semana_ano',
            'is_weekend', 'is_month_start', 'is_month_end', 'trimestre',
            # Variáveis exógenas relevantes
            'Promo', 'SchoolHoliday', 'StateHoliday_flag'
        ]
        
        if is_single_store:
            feature_columns.extend([f'lag_{i}' for i in [1, 2, 3, 7]])
            feature_columns.extend([f'roll_mean_{i}' for i in [7, 14]])
            
        X_train = ts_with_features[feature_columns]
        y_train = ts_with_features['y']
        
        # Treinar modelo
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # Preparar dados futuros para previsão
        future_df = pd.DataFrame({'ds': future_dates})
        future_df = create_date_features(future_df)
        # Define valores padrão (zero) para variáveis exógenas futuras
        future_df['Promo'] = 0
        future_df['SchoolHoliday'] = 0
        future_df['StateHoliday_flag'] = 0
        
        if is_single_store:
            # Propagar últimos valores conhecidos
            last_values = ts_with_features.iloc[-1].to_dict()
            for lag in [1, 2, 3, 7]:
                if f'lag_{lag}' in last_values:
                    future_df[f'lag_{lag}'] = last_values[f'lag_{lag}']
            for w in [7, 14]:
                if f'roll_mean_{w}' in last_values:
                    future_df[f'roll_mean_{w}'] = last_values[f'roll_mean_{w}']
                    
        X_future = future_df[feature_columns]
        
        # Fazer previsões
        y_pred = rf.predict(X_future)
        
        # Calcular intervalos de confiança usando a variância das árvores
        predictions = []
        n_trees = rf.n_estimators
        for estimator in rf.estimators_:
            predictions.append(estimator.predict(X_future))
        predictions = np.array(predictions)
        
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        # Criar DataFrame de forecast
        forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': y_pred,
            'yhat_lower': lower_bound,
            'yhat_upper': upper_bound
        })
        
        # Reverter transformação logarítmica
        if log_transform:
            forecast['yhat'] = np.expm1(forecast['yhat'])
            forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
            
        # Garantir valores não negativos
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        # Calcular previsões in-sample
        pred_train = rf.predict(X_train)
        if log_transform:
            pred_train = np.expm1(pred_train)
            
        # Calcular desvios
        y_true = ts_clean['y_original'].values if 'y_original' in ts_clean.columns else ts_clean['y'].values
        
        # Calcular previsões in-sample com intervalos de confiança
        train_predictions = []
        for estimator in rf.estimators_:
            train_predictions.append(estimator.predict(X_train))
        train_predictions = np.array(train_predictions)
        
        train_lower = np.percentile(train_predictions, 2.5, axis=0)
        train_upper = np.percentile(train_predictions, 97.5, axis=0)
        
        if log_transform:
            train_lower = np.expm1(train_lower)
            train_upper = np.expm1(train_upper)
            
        deviation_mask = (y_true > train_upper) | (y_true < train_lower)
        deviation_count = int(deviation_mask.sum())
        
        logger.info("Previsão gerada (%s) – %d pontos futuros", model_name_display, len(forecast))
        return forecast, pred_train, deviation_count, model_name_display

    # ------------------------------------------------------------------
    # Prophet
    # ------------------------------------------------------------------
    if modelo == "prophet":
        model_name_display = "Prophet"

    # ---------------------------
    # Pré-processamento específico
    # ---------------------------
    ts_clean = preprocess_prophet_ts(ts)
    if ts_clean.empty:
        raise ValueError("Série vazia após pré-processamento Prophet.")

    # Verificar se estamos lidando com uma loja individual ou um grupo
    is_single_store = len(ts_clean) < 500  # Heurística para identificar loja individual
    
    # Aplicar transformação logarítmica para estabilizar a variância
    # Isso ajuda especialmente com lojas individuais que têm maior volatilidade
    log_transform = True
    if log_transform:
        # Adicionar um pequeno valor para evitar log(0)
        epsilon = 1e-3
        ts_clean['y_original'] = ts_clean['y'].copy()
        ts_clean['y'] = np.log1p(ts_clean['y'])

    # ---------------------------
    # Adicionar features de data e lag para melhorar a previsão
    # ---------------------------
    # Adicionar features de data
    ts_with_features = create_date_features(ts_clean)
    
    # Para lojas individuais, adicionar lag features para capturar padrões recentes
    if is_single_store:
        # Usar lags mais curtos para lojas individuais
        ts_with_features = create_lag_features(
            ts_with_features, 
            lags=[1, 2, 3, 7], 
            roll_windows=[7, 14]
        )
        # Remover linhas iniciais com NaN devido aos lags
        ts_with_features = ts_with_features.dropna()

    # ---------------------------
    # Construção do modelo Prophet customizado
    # ---------------------------
    extra_regressors = [c for c in ts_with_features.columns if c not in ("ds", "y", "y_original")]

    # Ajustar configurações do Prophet com base no tipo de loja
    if is_single_store:
        # Para lojas individuais: mais flexibilidade na tendência, menos nas sazonalidades
        m = Prophet(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.2,  # aumenta flexibilidade para quedas bruscas
            seasonality_prior_scale=3,
            holidays=holidays,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )
    else:
        # Para grupos de lojas: configuração padrão
        m = Prophet(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.1,
            seasonality_prior_scale=8,
            holidays=holidays,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
        )

    # Sazonalidades customizadas
    m.add_seasonality(name="weekly", period=7, fourier_order=8)
    m.add_seasonality(name="monthly", period=30.5, fourier_order=6)
    m.add_seasonality(name="yearly", period=365.25, fourier_order=10)

    # Regressoras externas
    for reg in extra_regressors:
        m.add_regressor(reg)

    # Treinamento
    m.fit(ts_with_features)

    # ---------------------------
    # Previsão futura
    # ---------------------------
    # Criar DataFrame futuro com as mesmas features
    future_df = pd.DataFrame({"ds": future_dates})
    future_df = create_date_features(future_df)
    
    # Para lojas individuais, propagar os últimos valores de lag
    if is_single_store:
        # Obter os últimos valores conhecidos para propagar
        last_values = ts_with_features.iloc[-1].to_dict()
        
        # Propagar lag features
        for lag in [1, 2, 3, 7]:
            if f"lag_{lag}" in last_values:
                future_df[f"lag_{lag}"] = last_values[f"lag_{lag}"]
        
        # Propagar rolling means
        for w in [7, 14]:
            if f"roll_mean_{w}" in last_values:
                future_df[f"roll_mean_{w}"] = last_values[f"roll_mean_{w}"]
    
    # Preencher regressores externos com valores padrão
    for reg in extra_regressors:
        if reg not in future_df.columns:
            future_df[reg] = 0  # assume ausência de promo/feriado se não informado
    
    # Fazer a previsão
    forecast_df = m.predict(future_df)
    forecast = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    
    # Reverter transformação logarítmica se aplicada
    if log_transform:
        forecast['yhat'] = np.expm1(forecast['yhat'])
        if 'yhat_lower' in forecast.columns:
            forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
        if 'yhat_upper' in forecast.columns:
            forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # Garantir suavidade na transição entre histórico e previsão
    # Ajustar o primeiro ponto de previsão para reduzir saltos
    if not ts_clean.empty and not forecast.empty:
        last_actual = ts_clean['y_original'].iloc[-1] if 'y_original' in ts_clean.columns else ts_clean['y'].iloc[-1]
        first_pred = forecast['yhat'].iloc[0]
        
        # Se a diferença for muito grande (>20%), suavizar
        if abs(first_pred / last_actual - 1) > 0.2:
            # Aplicar um fator de correção que diminui gradualmente
            correction_factor = last_actual / first_pred
            decay_rate = 0.8  # Taxa de decaimento do fator de correção
            
            for i in range(min(14, len(forecast))):  # Primeiros 14 dias ou menos
                decay = decay_rate ** i
                adjustment = 1 + (correction_factor - 1) * decay
                forecast.loc[i, 'yhat'] *= adjustment
                
                if 'yhat_lower' in forecast.columns:
                    forecast.loc[i, 'yhat_lower'] *= adjustment
                if 'yhat_upper' in forecast.columns:
                    forecast.loc[i, 'yhat_upper'] *= adjustment

    # Garante que o forecast não seja negativo
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    if 'yhat_lower' in forecast.columns:
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    if 'yhat_upper' in forecast.columns:
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

    # in-sample
    pred_train_df = m.predict(ts_with_features)
    pred_train = pred_train_df["yhat"].values
    
    # Reverter transformação logarítmica para pred_train se aplicada
    if log_transform:
        pred_train = np.expm1(pred_train)
        
    # Calcular desvios
    y_true = ts_clean['y_original'].values if 'y_original' in ts_clean.columns else ts_clean['y'].values
    deviation_mask = (y_true > np.expm1(pred_train_df["yhat_upper"]) if log_transform else y_true > pred_train_df["yhat_upper"]) | (
        y_true < np.expm1(pred_train_df["yhat_lower"]) if log_transform else y_true < pred_train_df["yhat_lower"]
    )
    deviation_count = int(deviation_mask.sum())

    # Não há outras opções além de Prophet e Random Forest neste momento.
    if modelo not in ["prophet", "random_forest"]:
        raise ValueError(
            "Modelo não implementado. Use 'prophet' ou 'random_forest'."
        )

    logger.info("Previsão gerada (%s) – %d pontos futuros", model_name_display, len(forecast))
    return forecast, pred_train, deviation_count, model_name_display 