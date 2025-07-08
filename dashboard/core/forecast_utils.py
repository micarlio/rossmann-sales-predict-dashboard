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
    ts_clean.loc[ts_clean["y"] <= 0, "y"] = np.nan
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

    # Por enquanto suportamos apenas o Prophet — demais modelos serão implementados futuramente
    if modelo != "prophet":
        raise ValueError(
            "Modelo não implementado."
        )

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

        # ---------------------------
        # Construção do modelo Prophet customizado
        # ---------------------------
        extra_regressors = [c for c in ts_clean.columns if c not in ("ds", "y")]

        m = Prophet(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            holidays=holidays,
            daily_seasonality=False,  # vamos adicionar manualmente se necessário
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
        m.fit(ts_clean)

        # ---------------------------
        # Previsão futura
        # ---------------------------
        future_df = future_dates.to_frame(name="ds")
        for reg in extra_regressors:
            future_df[reg] = 0  # assume ausência de promo/feriado se não informado
        forecast_df = m.predict(future_df)
        forecast = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        # Garante que o forecast não seja negativo
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        if 'yhat_lower' in forecast.columns:
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        if 'yhat_upper' in forecast.columns:
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

        # in-sample
        pred_train_df = m.predict(ts_clean)
        pred_train = pred_train_df["yhat"].values
        deviation_mask = (ts_clean["y"] > pred_train_df["yhat_upper"]) | (
            ts_clean["y"] < pred_train_df["yhat_lower"]
        )
        deviation_count = int(deviation_mask.sum())

    # Não há outras opções além de Prophet neste momento.

    logger.info("Previsão gerada (%s) – %d pontos futuros", model_name_display, len(forecast))
    return forecast, pred_train, deviation_count, model_name_display 