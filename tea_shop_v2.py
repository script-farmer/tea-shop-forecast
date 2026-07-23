#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
茶饮店销量预测 + 智能备货系统 V2
=================================
V2 升级：
  - 泊松回归（GLM）替代朴素基线模型，整合周末/节假日/天气/促销特征
  - 可作为 Python 模块导入（供 Streamlit 仪表板使用），也可 CLI 独立运行

用法：
  python tea_shop_v2.py --demo
  python tea_shop_v2.py --input sales.csv --city 北京
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    backtest_days: int = 7
    ma_window: int = 7
    ewma_alpha: float = 0.3
    seasonal_period: int = 7
    min_history_days: int = 14  # GLM 比基线模型需要更多数据
    weather_cache_dir: str = "weather_cache"


# 中国主要城市经纬度（用于 open-meteo API）
CITY_COORDS: dict[str, tuple[float | None, float | None]] = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1291, 113.2644),
    "深圳": (22.5431, 114.0579),
    "成都": (30.5728, 104.0668),
    "杭州": (30.2741, 120.1551),
    "武汉": (30.5928, 114.3055),
    "重庆": (29.4316, 106.9123),
    "西安": (34.3416, 108.9398),
    "昆明": (25.0389, 102.7183),
    "苏州": (31.2990, 120.5853),
    "南京": (32.0603, 118.7969),
    "长沙": (28.2282, 112.9388),
    "郑州": (34.7466, 113.6253),
    "天津": (39.1252, 117.1905),
    "无天气数据": (None, None),
}

# open-meteo 免费 API 端点
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"


# ═══════════════════════════════════════════════════════════
#  第一部分：数据加载与清洗
# ═══════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """读取交易 CSV 并标准化列名和类型。"""
    df = pd.read_csv(path)

    col_map = {
        "交易编号": "txn_id", "商品名称": "item", "数量": "qty",
        "单价": "price", "单位成本": "cost", "日期": "date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    required = ["txn_id", "item", "qty", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        rev = {v: k for k, v in col_map.items()}
        raise ValueError(
            f"CSV 缺少必须列：{[rev.get(c, c) for c in missing]}\n"
            f"必须包含：交易编号, 商品名称, 数量, 日期"
        )

    df["item"] = df["item"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    if "price" not in df.columns:
        df["price"] = 0.0
    else:
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    if "cost" not in df.columns:
        df["cost"] = df["price"] * 0.4
    else:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0)

    df = df.dropna(subset=["date", "item"])
    df = df[df["qty"] > 0]
    return df.sort_values(["date", "item"])


# ═══════════════════════════════════════════════════════════
#  第二部分：按天按品类汇总
# ═══════════════════════════════════════════════════════════

def aggregate_daily(txn: pd.DataFrame) -> pd.DataFrame:
    """将交易流水汇总为每日每品类的销量序列，缺失日期填 0。"""
    txn = txn.copy()
    txn["total_revenue"] = txn["price"] * txn["qty"]

    daily = txn.groupby(["date", "item"], observed=False).agg(
        demand_qty=("qty", "sum"),
        revenue=("total_revenue", "sum"),
        txn_count=("qty", "count"),
        avg_cost=("cost", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    items = sorted(daily["item"].unique().tolist())
    idx = pd.MultiIndex.from_product([all_dates, items], names=["date", "item"])
    full = daily.set_index(["date", "item"]).reindex(idx).reset_index()
    full["demand_qty"] = full["demand_qty"].fillna(0.0)
    full["revenue"] = full["revenue"].fillna(0.0)
    full["txn_count"] = full["txn_count"].fillna(0).astype(int)

    item_costs = daily.groupby("item")["avg_cost"].mean()
    for it in items:
        mask = (full["item"] == it) & (full["avg_cost"].isna())
        full.loc[mask, "avg_cost"] = item_costs.get(it, 0.0)

    return full.sort_values(["item", "date"])


# ═══════════════════════════════════════════════════════════
#  第三部分：特征工程
# ═══════════════════════════════════════════════════════════

def _add_weekend_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["is_weekend"] = df["date"].apply(lambda d: 1 if d.weekday() >= 5 else 0)
    return df


def _add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from chinesecalendar import is_holiday
        df["is_holiday"] = df["date"].apply(
            lambda d: 1 if is_holiday(d.to_pydatetime().date()) else 0
        )
    except ImportError:
        df["is_holiday"] = 0
    return df


def _add_promotion_flag(df: pd.DataFrame, promotion_dates: set | None) -> pd.DataFrame:
    if promotion_dates:
        df["is_promotion"] = df["date"].apply(
            lambda d: 1 if pd.Timestamp(d) in promotion_dates else 0
        )
    else:
        df["is_promotion"] = 0
    return df


def build_features(
    daily: pd.DataFrame,
    city: str | None = None,
    promotion_dates: set[pd.Timestamp] | None = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """为每日汇总数据添加外部特征列。

    返回带有 is_weekend、is_holiday、temperature_max、precipitation_sum、
    is_promotion 列的 DataFrame。天气获取失败时静默回退。
    """
    if cfg is None:
        cfg = Config()

    df = daily.copy()
    df = _add_weekend_flag(df)
    df = _add_holiday_flag(df)
    df = _add_promotion_flag(df, promotion_dates)

    lat, lon = CITY_COORDS.get(city or "无天气数据", (None, None))
    if lat is not None and lon is not None:
        try:
            weather = fetch_weather(
                city=str(city),  # type: ignore[arg-type]
                start_date=df["date"].min().strftime("%Y-%m-%d"),
                end_date=df["date"].max().strftime("%Y-%m-%d"),
                cache_dir=cfg.weather_cache_dir,
            )
            if not weather.empty:
                weather["date"] = pd.to_datetime(weather["date"])
                df = df.merge(weather, on="date", how="left")
                # 缺失值用当月均值回退
                df["temperature_max"] = df.groupby(
                    df["date"].dt.month
                )["temperature_max"].transform(lambda x: x.fillna(x.mean()))
                df["temperature_max"] = df["temperature_max"].fillna(20.0)
                df["precipitation_sum"] = df["precipitation_sum"].fillna(0.0)
            else:
                df["temperature_max"] = 20.0
                df["precipitation_sum"] = 0.0
        except Exception:
            df["temperature_max"] = 20.0
            df["precipitation_sum"] = 0.0
    else:
        df["temperature_max"] = 20.0
        df["precipitation_sum"] = 0.0

    return df


# ═══════════════════════════════════════════════════════════
#  第四部分：天气数据获取与缓存
# ═══════════════════════════════════════════════════════════

def _weather_cache_path(city: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{city}_weather.parquet"


def _load_weather_cache(city: str, cache_dir: str) -> pd.DataFrame:
    path = _weather_cache_path(city, cache_dir)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame(columns=["date", "temperature_max", "precipitation_sum"])


def _save_weather_cache(city: str, cache_dir: str, df: pd.DataFrame):
    path = _weather_cache_path(city, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        df.to_csv(str(path).replace(".parquet", ".csv"), index=False, encoding="utf-8-sig")


def fetch_weather(
    city: str,
    start_date: str,
    end_date: str,
    cache_dir: str = "weather_cache",
) -> pd.DataFrame:
    """获取指定城市的每日天气数据（最高温 + 降水量）。

    先从本地缓存加载已有数据，仅向 open-meteo API 请求缺失的日期。
    完全免费，无需 API Key。
    """
    lat, lon = CITY_COORDS.get(city, (None, None))
    if lat is None or lon is None:
        return pd.DataFrame()

    cache = _load_weather_cache(city, cache_dir)
    needed_start = pd.Timestamp(start_date)
    needed_end = pd.Timestamp(end_date)

    if not cache.empty:
        cache["date"] = pd.to_datetime(cache["date"])
        cache_min = cache["date"].min()
        cache_max = cache["date"].max()
        # 计算缺失的日期段
        gaps: list[tuple[str, str]] = []
        if needed_start < cache_min:
            gaps.append((needed_start.strftime("%Y-%m-%d"),
                         (cache_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        if needed_end > cache_max:
            gaps.append(((cache_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                         needed_end.strftime("%Y-%m-%d")))
    else:
        gaps = [(needed_start.strftime("%Y-%m-%d"), needed_end.strftime("%Y-%m-%d"))]

    if not gaps:
        mask = (cache["date"] >= needed_start) & (cache["date"] <= needed_end)
        return cache[mask].reset_index(drop=True)

    new_data = []
    for gs, ge in gaps:
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": gs,
                "end_date": ge,
                "daily": "temperature_2m_max,precipitation_sum",
                "timezone": "Asia/Shanghai",
            }
            resp = requests.get(ARCHIVE_API, params=params, timeout=15)
            resp.raise_for_status()
            body = resp.json()
            if "daily" in body:
                rows = pd.DataFrame({
                    "date": pd.to_datetime(body["daily"]["time"]),
                    "temperature_max": body["daily"]["temperature_2m_max"],
                    "precipitation_sum": body["daily"]["precipitation_sum"],
                })
                new_data.append(rows)
        except Exception:
            continue

    if new_data:
        fetched = pd.concat(new_data, ignore_index=True)
        if not cache.empty and not fetched.empty:
            cache = pd.concat([cache, fetched], ignore_index=True)
        elif not fetched.empty:
            cache = fetched
        cache = cache.drop_duplicates(subset="date").sort_values("date")
        _save_weather_cache(city, cache_dir, cache)

    mask = (cache["date"] >= needed_start) & (cache["date"] <= needed_end)
    return cache[mask].reset_index(drop=True)


def fetch_weather_forecast(
    city: str,
    forecast_days: int = 7,
) -> pd.DataFrame:
    """获取未来 N 天的天气预报。"""
    lat, lon = CITY_COORDS.get(city, (None, None))
    if lat is None or lon is None:
        return pd.DataFrame()

    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,precipitation_sum",
            "forecast_days": forecast_days,
            "timezone": "Asia/Shanghai",
        }
        resp = requests.get(FORECAST_API, params=params, timeout=15)
        resp.raise_for_status()
        body = resp.json()
        if "daily" in body:
            return pd.DataFrame({
                "date": pd.to_datetime(body["daily"]["time"]),
                "temperature_max": body["daily"]["temperature_2m_max"],
                "precipitation_sum": body["daily"]["precipitation_sum"],
            })
    except Exception:
        pass
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
#  第五部分：基线预测模型
# ═══════════════════════════════════════════════════════════

def forecast_ma(series: np.ndarray, horizon: int, window: int = 7) -> np.ndarray:
    w = min(window, len(series))
    if w == 0:
        return np.zeros(horizon)
    return np.repeat(float(series[-w:].mean()), horizon)


def forecast_ewma(series: np.ndarray, horizon: int, alpha: float = 0.3) -> np.ndarray:
    if len(series) == 0:
        return np.zeros(horizon)
    level = float(series[0])
    for x in series[1:]:
        level = alpha * float(x) + (1 - alpha) * level
    return np.repeat(level, horizon)


def forecast_seasonal_naive(series: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    if len(series) < period:
        return np.repeat(float(series[-1]) if len(series) else 0.0, horizon)
    last_period = series[-period:].astype(float)
    reps = int(np.ceil(horizon / period))
    return np.tile(last_period, reps)[:horizon]


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(
        np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    )))


# ═══════════════════════════════════════════════════════════
#  第六部分：Poisson GLM 训练与预测
# ═══════════════════════════════════════════════════════════

def _build_glm_formula(has_weather: bool, has_promotion: bool) -> str:
    parts = ["demand_qty ~ C(item)", "is_weekend", "is_holiday"]
    if has_weather:
        parts.extend(["temperature_max", "precipitation_sum"])
    if has_promotion:
        parts.append("is_promotion")
    return " + ".join(parts)


def train_glm(df: pd.DataFrame) -> tuple:
    """训练 pooled Poisson GLM。返回 (model, used_features)。

    数据不足或收敛失败时 model 为 None。
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    has_weather = (
        "temperature_max" in df.columns
        and "precipitation_sum" in df.columns
        and df["temperature_max"].notna().any()
    )
    has_promotion = "is_promotion" in df.columns and df["is_promotion"].sum() > 0

    formula = _build_glm_formula(has_weather, has_promotion)
    features = ["is_weekend", "is_holiday"]
    if has_weather:
        features.extend(["temperature_max", "precipitation_sum"])
    if has_promotion:
        features.append("is_promotion")

    if len(df) < 30:
        return None, features

    try:
        model = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
        if hasattr(model, "converged") and not model.converged:
            return None, features
        return model, features
    except Exception:
        return None, features


def predict_glm(model, future_df: pd.DataFrame) -> np.ndarray:
    preds = model.predict(future_df)
    return np.asarray(preds, dtype=float)


def glm_coefficient_table(model) -> pd.DataFrame:
    """提取 GLM 系数表（含 p 值、置信区间），方便 Streamlit 展示。"""
    summary = model.summary2().tables[1]
    df = summary.copy()
    df = df.reset_index().rename(columns={"index": "特征"})
    name_map = {
        "Intercept": "截距",
        "is_weekend": "周末",
        "is_holiday": "节假日",
        "temperature_max": "最高温度",
        "precipitation_sum": "降水量",
        "is_promotion": "促销",
        "C(item)[T.柠檬茶]": "品类_柠檬茶",
        "C(item)[T.珍珠奶茶]": "品类_珍珠奶茶",
        "C(item)[T.杨枝甘露]": "品类_杨枝甘露",
        "C(item)[T.金桔柠檬]": "品类_金桔柠檬",
        "C(item)[T.芝士奶盖]": "品类_芝士奶盖",
        "C(item)[T.芒果冰沙]": "品类_芒果冰沙",
        "C(item)[T.西瓜汁]": "品类_西瓜汁",
    }
    df["特征"] = df["特征"].map(lambda x: name_map.get(str(x), str(x)))
    return df


# ═══════════════════════════════════════════════════════════
#  第七部分：回测与模型选择
# ═══════════════════════════════════════════════════════════

def backtest_and_select(
    series: np.ndarray,
    features_df: pd.DataFrame | None,
    item_name: str,
    cfg: Config,
) -> dict:
    """对给定品类做回测，比较 GLM + 3 基线模型，选 MAE 最低的。

    Returns:
        {best_model, scores: {model_name: mae}, glm_summary: ...}
    """
    n = len(series)
    result: dict = {
        "best_model": "数据不足",
        "scores": {},
        "glm_summary": None,
        "tomorrow_pred": 0.0,
    }

    if n < cfg.min_history_days:
        if n > 0:
            result["best_model"] = "数据不足"
            result["tomorrow_pred"] = max(float(series[-1]), 0.0)
        return result

    test_n = min(cfg.backtest_days, max(1, n // 4))
    train_series = series[:-test_n]
    test_series = series[-test_n:]

    # === 基线模型回测 ===
    baseline_models = {
        "7日移动平均": lambda: forecast_ma(train_series, test_n, cfg.ma_window),
        "指数加权平滑": lambda: forecast_ewma(train_series, test_n, cfg.ewma_alpha),
        "周季节朴素": lambda: forecast_seasonal_naive(train_series, test_n, cfg.seasonal_period),
    }
    for name, fn in baseline_models.items():
        result["scores"][name] = compute_mae(test_series, fn())

    # === GLM 回测 ===
    glm_model = None
    if features_df is not None and len(features_df) >= 30:
        # 只取该品类的特征行
        train_features = features_df.iloc[:-test_n]
        test_features = features_df.iloc[-test_n:]
        glm_model, used_features = train_glm(train_features)
        if glm_model is not None:
            glm_preds = predict_glm(glm_model, test_features)
            result["scores"]["泊松回归(GLM)"] = compute_mae(test_series, glm_preds)

    best_model = min(result["scores"], key=result["scores"].get) if result["scores"] else "数据不足"
    result["best_model"] = best_model

    # === 用最优模型预测明天 ===
    if best_model == "泊松回归(GLM)" and glm_model is not None:
        # 需要用未来1天的特征来预测 — run_pipeline 会在外部做这个，这里先用基线
        # 但回测结束时没有未来特征，暂时用最新特征行的均值
        if features_df is not None:
            last_features = features_df.iloc[-1:].copy()
            result["tomorrow_pred"] = max(float(predict_glm(glm_model, last_features)[0]), 0.0)
    elif best_model == "7日移动平均":
        result["tomorrow_pred"] = max(float(forecast_ma(series, 1, cfg.ma_window)[0]), 0.0)
    elif best_model == "指数加权平滑":
        result["tomorrow_pred"] = max(float(forecast_ewma(series, 1, cfg.ewma_alpha)[0]), 0.0)
    elif best_model == "周季节朴素":
        result["tomorrow_pred"] = max(float(forecast_seasonal_naive(series, 1, cfg.seasonal_period)[0]), 0.0)
    elif n > 0:
        result["tomorrow_pred"] = max(float(series[-1]), 0.0)

    # GLM 摘要（用于仪表板展示系数表）
    if glm_model is not None:
        try:
            result["glm_summary"] = glm_coefficient_table(glm_model)
        except Exception:
            result["glm_summary"] = None

    return result


# ═══════════════════════════════════════════════════════════
#  第八部分：泊松报童模型
# ═══════════════════════════════════════════════════════════

def newsvendor_poisson(lam: float, price: float, cost: float) -> dict:
    """泊松报童模型计算最优备货量。"""
    if lam <= 0:
        return {
            "Q_star": 0, "profit": 0.0, "lost_sales": 0.0,
            "waste": 0.0, "margin": 0.0, "fr": 0.0,
        }
    Cu = max(price - cost, 0.01)
    Co = max(cost, 0.01)
    Fr = Cu / (Cu + Co)
    Q_star = int(poisson.ppf(Fr, lam))
    max_d = max(Q_star, int(lam * 5))
    probs = np.array([poisson.pmf(d, lam) for d in range(max_d + 1)])
    expected_sales = sum(min(d, Q_star) * probs[d] for d in range(max_d + 1))
    expected_waste = max(Q_star - expected_sales, 0.0)
    expected_lost = max(lam - expected_sales, 0.0)
    expected_profit = Cu * expected_sales - Co * expected_waste
    return {
        "Q_star": Q_star,
        "profit": round(expected_profit, 2),
        "lost_sales": round(expected_lost, 2),
        "waste": round(expected_waste, 2),
        "margin": round(Q_star - lam, 1),
        "fr": round(Fr, 3),
    }


# ═══════════════════════════════════════════════════════════
#  第九部分：主流程
# ═══════════════════════════════════════════════════════════

def run_pipeline(
    csv_path: str | None = None,
    city: str = "无天气数据",
    horizon: int = 3,
    promotion_dates: set[pd.Timestamp] | None = None,
    output_dir: str | None = None,
    use_demo: bool = False,
    daily_df: pd.DataFrame | None = None,
) -> dict:
    """执行完整流水线。

    可通过 csv_path（从文件加载）或 daily_df（直接传入已汇总数据）调用。
    后者供 Streamlit 仪表板使用，避免重复加载。

    Returns:
        {
            daily: pd.DataFrame,          # 含特征的每日汇总
            forecast_df: pd.DataFrame,    # 预测结果
            backtest_df: pd.DataFrame,    # 回测分数
            ordering_df: pd.DataFrame,    # 备货建议
            glm_summary: pd.DataFrame,    # GLM 系数
            weather_available: bool,
            glm_available: bool,
            messages: list[str],          # 给用户的警告
        }
    """
    cfg = Config()
    messages: list[str] = []

    # 1-2. 加载 + 汇总
    if daily_df is not None:
        daily = daily_df.copy()
        txn_count_msg = "直接传入汇总数据"
    elif use_demo:
        demo_path = generate_demo_data()
        txn = load_and_clean(demo_path)
        daily = aggregate_daily(txn)
        txn_count_msg = f"演示数据，{len(txn)} 条交易"
    elif csv_path:
        txn = load_and_clean(csv_path)
        daily = aggregate_daily(txn)
        txn_count_msg = f"{len(txn)} 条交易"
    else:
        demo_path = generate_demo_data()
        txn = load_and_clean(demo_path)
        daily = aggregate_daily(txn)
        txn_count_msg = f"演示数据，{len(txn)} 条交易"

    items = sorted(daily["item"].unique().tolist())
    all_dates = sorted(daily["date"].unique())

    if len(all_dates) < cfg.min_history_days:
        messages.append(
            f"仅 {len(all_dates)} 天数据，最少需 {cfg.min_history_days} 天。"
            f"GLM 不可用，将回退到基线模型。"
        )

    # 3. 特征工程
    daily_with_features = build_features(daily, city, promotion_dates, cfg)
    weather_available = (
        "temperature_max" in daily_with_features.columns
        and daily_with_features["temperature_max"].notna().any()
        and city != "无天气数据"
    )

    # 4. 训练 GLM（全历史数据）
    glm_model = None
    glm_features: list[str] = []
    glm_available = False
    if len(daily_with_features) >= 30:
        glm_model, glm_features = train_glm(daily_with_features)
        glm_available = glm_model is not None

    if not glm_available and len(daily_with_features) >= 14:
        messages.append("GLM 未收敛或数据不足，使用基线模型预测。")

    # 5. 构建未来特征（用于预测）
    tomorrow_start = all_dates[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(tomorrow_start, periods=horizon, freq="D")
    future_items = items

    future_rows = []
    for d in future_dates:
        for it in future_items:
            future_rows.append({"date": d, "item": it, "demand_qty": 0.0})
    future_df = pd.DataFrame(future_rows)
    future_df = _add_weekend_flag(future_df)
    future_df = _add_holiday_flag(future_df)
    future_df = _add_promotion_flag(future_df, promotion_dates)

    # 天气：（1）先试预报 API，（2）回退到历史同期均值
    weather_forecast = pd.DataFrame()
    if weather_available and city != "无天气数据":
        weather_forecast = fetch_weather_forecast(city, horizon)
    if not weather_forecast.empty:
        weather_forecast["date"] = pd.to_datetime(weather_forecast["date"])
        future_df = future_df.merge(weather_forecast, on="date", how="left")
        future_df["temperature_max"] = future_df["temperature_max"].fillna(20.0)
        future_df["precipitation_sum"] = future_df["precipitation_sum"].fillna(0.0)
    else:
        future_df["temperature_max"] = 20.0
        future_df["precipitation_sum"] = 0.0

    # 6. 预测未来 horizon 天
    forecast_rows = []
    if glm_available:
        for d in future_dates:
            day_mask = future_df["date"] == d
            day_features = future_df[day_mask]
            preds = predict_glm(glm_model, day_features)
            for i, it in enumerate(future_items):
                forecast_rows.append({
                    "date": d,
                    "item": it,
                    "predicted_lambda": max(float(preds[i]), 0.0),
                    "model": "泊松回归(GLM)",
                })
    else:
        for item in items:
            item_daily = daily_with_features[daily_with_features["item"] == item]
            series = item_daily["demand_qty"].to_numpy()
            bt = backtest_and_select(series, None, item, cfg)
            best = bt["best_model"]
            if best == "7日移动平均":
                preds = forecast_ma(series, horizon, cfg.ma_window)
            elif best == "指数加权平滑":
                preds = forecast_ewma(series, horizon, cfg.ewma_alpha)
            elif best == "周季节朴素":
                preds = forecast_seasonal_naive(series, horizon, cfg.seasonal_period)
            else:
                preds = np.repeat(float(series[-1]) if len(series) else 0.0, horizon)
            for j, d in enumerate(future_dates):
                forecast_rows.append({
                    "date": d,
                    "item": item,
                    "predicted_lambda": max(float(preds[j]), 0.0),
                    "model": best,
                })
    forecast_df = pd.DataFrame(forecast_rows)

    # 7. 回测所有品类
    backtest_rows = []
    for item in items:
        item_daily = daily_with_features[daily_with_features["item"] == item]
        series = item_daily["demand_qty"].to_numpy()
        item_features = None
        if glm_available:
            item_features = daily_with_features[daily_with_features["item"] == item]
        bt = backtest_and_select(series, item_features, item, cfg)
        for model_name, mae in bt["scores"].items():
            backtest_rows.append({
                "item": item,
                "model": model_name,
                "mae": round(mae, 2),
                "is_best": model_name == bt["best_model"],
            })
    backtest_df = pd.DataFrame(backtest_rows)

    # 8. Newsvendor 备货计算
    ordering_rows = []
    item_prices: dict[str, tuple[float, float]] = {}
    for item in items:
        item_daily = daily[daily["item"] == item]
        recent = item_daily.tail(30)
        avg_price = float(recent["revenue"].sum() / max(recent["demand_qty"].sum(), 1))
        avg_cost = float(recent["avg_cost"].mean())
        item_prices[item] = (avg_price, avg_cost)

    for _, row in forecast_df.iterrows():
        it = row["item"]
        lam = row["predicted_lambda"]
        price, cost = item_prices.get(it, (0.0, 0.0))
        nv = newsvendor_poisson(lam, price, cost)
        ordering_rows.append({
            "date": row["date"],
            "item": it,
            "predicted_lambda": round(lam, 1),
            "Q_star": nv["Q_star"],
            "safety_margin": nv["margin"],
            "expected_profit": nv["profit"],
            "expected_waste": nv["waste"],
            "expected_lost_sales": nv["lost_sales"],
            "critical_fractile": nv["fr"],
        })
    ordering_df = pd.DataFrame(ordering_rows)

    # 9. GLM 摘要
    glm_summary_df = None
    if glm_model is not None:
        try:
            glm_summary_df = glm_coefficient_table(glm_model)
        except Exception:
            glm_summary_df = None

    # 10. 终端输出
    tomorrow = all_dates[-1] + pd.Timedelta(days=1)
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    print()
    print("=" * 72)
    print(f"    茶饮店明日备货建议 —— {tomorrow.strftime('%Y-%m-%d')}"
          f"（{weekday_names[tomorrow.weekday()]}）")
    if glm_available:
        print(f"    核心模型：泊松回归(GLM)  |  天气：{'有' if weather_available else '无'}"
              f"  |  城市：{city}")
    else:
        print(f"    核心模型：基线模型（GLM不可用）  |  城市：{city}")
    print("=" * 72)
    print()

    today_ordering = ordering_df[ordering_df["date"] == ordering_df["date"].min()]
    header = (
        f"{'品类':<12s} {'预测':>6s} {'备货':>6s} {'余量':>5s} {'利润':>7s}  {'模型':<16s}"
    )
    print(header)
    print("-" * 72)
    for _, r in today_ordering.sort_values("Q_star", ascending=False).iterrows():
        model_used = forecast_df[
            (forecast_df["item"] == r["item"]) & (forecast_df["date"] == r["date"])
        ]["model"].values[0]
        print(
            f"{r['item']:<12s} {r['predicted_lambda']:>5.1f}杯 "
            f"{int(r['Q_star']):>4d}杯 {r['safety_margin']:>+4.1f} "
            f"元{r['expected_profit']:>6.0f}  {model_used:<16s}"
        )
    print("-" * 72)
    print()
    if messages:
        for m in messages:
            print(f"[!] {m}")
    print(f"[+] 数据范围：{all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}，"
          f"{len(all_dates)} 天，{len(items)} 品类，{txn_count_msg}")

    if output_dir:
        out_path = Path(output_dir) / f"tea_forecast_v2_{tomorrow.strftime('%Y%m%d')}.csv"
        ordering_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[+] 结果已导出: {out_path}")

    return {
        "daily": daily_with_features,
        "forecast_df": forecast_df,
        "backtest_df": backtest_df,
        "ordering_df": ordering_df,
        "glm_summary": glm_summary_df,
        "weather_available": weather_available,
        "glm_available": glm_available,
        "messages": messages,
        "city": city,
        "config": cfg,
    }


# ═══════════════════════════════════════════════════════════
#  第十部分：演示数据生成器（增强版）
# ═══════════════════════════════════════════════════════════

def generate_demo_data(out_path: str | None = None) -> str:
    """生成含周末/节假日/天气/促销效应的 90 天模拟交易数据。"""
    if out_path is None:
        out_path = str(Path(__file__).parent / "tea_shop_demo_v2.csv")

    rng = np.random.default_rng(42)
    items = {
        "珍珠奶茶": {"price": 15, "cost": 5, "base": 28, "weekend_boost": 1.3,
                      "heat_boost": 1.15, "cold_penalty": 0.90},
        "杨枝甘露": {"price": 18, "cost": 7, "base": 14, "weekend_boost": 1.4,
                      "heat_boost": 1.05, "cold_penalty": 0.85},
        "柠檬茶":   {"price": 12, "cost": 3, "base": 35, "weekend_boost": 1.2,
                      "heat_boost": 1.25, "cold_penalty": 0.92},
        "金桔柠檬": {"price": 12, "cost": 4, "base": 5, "weekend_boost": 1.5,
                      "heat_boost": 1.10, "cold_penalty": 0.90},
        "芝士奶盖": {"price": 20, "cost": 8, "base": 10, "weekend_boost": 1.3,
                      "heat_boost": 1.05, "cold_penalty": 0.88},
        "芒果冰沙": {"price": 22, "cost": 9, "base": 7, "weekend_boost": 1.6,
                      "heat_boost": 1.30, "cold_penalty": 0.70},
        "西瓜汁":   {"price": 10, "cost": 3, "base": 9, "weekend_boost": 1.7,
                      "heat_boost": 1.20, "cold_penalty": 0.82},
    }

    # 模拟温度序列（夏季：5月初到7月底，温度逐渐升高）
    dates = pd.date_range("2026-04-24", "2026-07-22", freq="D")
    base_temp = 20 + 15 * np.sin(np.linspace(0, np.pi, len(dates)))
    temp_noise = rng.normal(0, 3, len(dates))
    temps = np.clip(base_temp + temp_noise, 8, 38)

    # 降雨（约30%的日子有雨）
    rain_prob = 0.30
    rain_mask = rng.random(len(dates)) < rain_prob
    precip = np.where(rain_mask, rng.exponential(5, len(dates)), 0)

    # 节假日（五一、端午附近）
    holidays: set[pd.Timestamp] = {
        pd.Timestamp("2026-05-01"), pd.Timestamp("2026-05-02"),
        pd.Timestamp("2026-05-03"), pd.Timestamp("2026-05-04"),
        pd.Timestamp("2026-05-05"),
        pd.Timestamp("2026-06-19"), pd.Timestamp("2026-06-20"),
        pd.Timestamp("2026-06-21"),
    }

    # 促销日期（随机选10天）
    promo_dates: set[pd.Timestamp] = set(
        rng.choice(dates, size=10, replace=False)
    )

    txn_id = 0
    rows = []
    for i, d in enumerate(dates):
        is_weekend = 1 if d.dayofweek >= 5 else 0
        is_holiday = 1 if d in holidays else 0
        is_promo = 1 if d in promo_dates else 0
        t = temps[i]
        r = precip[i]

        for item, info in items.items():
            lam = float(info["base"])
            if is_weekend:
                lam *= info["weekend_boost"]
            if is_holiday:
                lam *= 1.20
            if is_promo:
                lam *= 1.30
            # 温度效应
            if t > 30:
                lam *= info["heat_boost"]
            elif t < 12:
                lam *= info["cold_penalty"]
            # 雨天轻微降低
            if r > 2:
                lam *= 0.92

            qty = rng.poisson(lam)
            if qty <= 0:
                continue
            n_txn = min(int(qty), rng.integers(1, min(4, int(qty) + 1)))
            splits = rng.multinomial(int(qty), np.ones(n_txn) / n_txn)
            for s in splits:
                if s <= 0:
                    continue
                txn_id += 1
                rows.append({
                    "交易编号": f"TXN_{txn_id:06d}",
                    "商品名称": item,
                    "数量": int(s),
                    "单价": info["price"],
                    "单位成本": info["cost"],
                    "日期": d.strftime("%Y-%m-%d"),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[+] 演示数据已生成: {out_path} ({len(df)} 条交易, {len(dates)} 天, {len(items)} 品类)")
    return out_path


# ═══════════════════════════════════════════════════════════
#  CLI 入口
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="茶饮店销量预测 + 智能备货系统 V2（泊松回归 + Streamlit 仪表板）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tea_shop_v2.py --demo                          # 生成演示数据并分析
  python tea_shop_v2.py --input sales.csv --city 北京    # 用自己的数据
  python tea_shop_v2.py --input sales.csv --city 北京 --horizon 7
        """,
    )
    parser.add_argument("--input", "-i", help="交易 CSV 文件路径")
    parser.add_argument("--output", "-o", default=None, help="输出目录")
    parser.add_argument("--city", default="无天气数据", help="城市名称（用于天气数据）")
    parser.add_argument("--horizon", type=int, default=3, help="预测天数（1-7）")
    parser.add_argument("--demo", action="store_true", help="生成演示数据并跑一遍")
    args = parser.parse_args()

    horizon = max(1, min(7, args.horizon))
    run_pipeline(
        csv_path=args.input if not args.demo else None,
        city=args.city,
        horizon=horizon,
        output_dir=args.output,
        use_demo=(args.demo or not args.input),
    )


if __name__ == "__main__":
    main()
