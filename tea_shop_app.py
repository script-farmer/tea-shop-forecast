#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
茶饮店销量预测 + 智能备货系统 V2 — Streamlit 仪表板
====================================================
启动方式：
  streamlit run tea_shop_app.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 确保 tea_shop_v2.py 在路径中
DESKTOP = Path(__file__).parent
sys.path.insert(0, str(DESKTOP))

from tea_shop_v2 import (  # type: ignore[import-not-found]
    CITY_COORDS,
    Config,
    aggregate_daily,
    load_and_clean,
    run_pipeline,
    generate_demo_data,
)

st.set_page_config(
    page_title="茶饮店销量预测 V2",
    page_icon="",
    layout="wide",
)
st.title("茶饮店销量预测 + 智能备货系统 V2")
st.caption("泊松回归（GLM） x 报童模型  |  周末 节假日 天气 促销全特征")

# ═══════════════════════════════════════════════════════════
#  侧边栏
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("配置")

    csv_file = st.file_uploader(
        "上传交易 CSV",
        type=["csv"],
        help="必须包含列：交易编号, 商品名称, 数量, 日期。可选：单价, 单位成本",
    )
    use_demo = st.checkbox("或使用演示数据", value=True)

    city = st.selectbox(
        "城市（天气数据）",
        options=list(CITY_COORDS.keys()),
        index=list(CITY_COORDS.keys()).index("无天气数据"),
        help="选择最近的城市以获取历史和预报天气数据。选择「无天气数据」将仅使用日历特征。",
    )

    horizon = st.slider(
        "预测天数",
        min_value=1,
        max_value=7,
        value=3,
        help="预测未来几天的销量",
    )

    promo_dates = st.date_input(
        "促销日期（可选）",
        value=[],
        help="选择计划促销的日期。可多选。",
    )

    run_btn = st.button("开始分析", type="primary", width="stretch")

    st.divider()
    st.caption("V2 新特性：")
    st.caption("Poisson 回归 + 天气 + 节假日")
    st.caption("Plotly 交互图表，透明可解释")
    st.caption("天气数据来自 open-meteo，免费无需 Key")

# ═══════════════════════════════════════════════════════════
#  执行管道
# ═══════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None
if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False

if run_btn:
    st.session_state.run_clicked = True
    promo_set: set[pd.Timestamp] | None = None
    if promo_dates:
        promo_set = {pd.Timestamp(d) for d in promo_dates}

    with st.spinner("正在加载数据、获取天气、训练模型……"):
        if csv_file is not None:
            # 保存上传文件到临时路径
            tmp_path = DESKTOP / "_uploaded_temp.csv"
            with open(tmp_path, "wb") as f:
                f.write(csv_file.getbuffer())
            results = run_pipeline(
                csv_path=str(tmp_path),
                city=city,
                horizon=horizon,
                promotion_dates=promo_set,
            )
            tmp_path.unlink(missing_ok=True)
        elif use_demo:
            results = run_pipeline(use_demo=True, city=city, horizon=horizon, promotion_dates=promo_set)
        else:
            results = run_pipeline(use_demo=True, city=city, horizon=horizon, promotion_dates=promo_set)

    st.session_state.results = results

results = st.session_state.results

if results is None and not st.session_state.run_clicked:
    st.info("请在侧边栏配置参数，然后点击「开始分析」。")
    # 展示空状态预览图
    st.markdown("""
    ### V2 升级亮点

    | 对比维度 | V1 | V2 |
    |---------|-----|-----|
    | 预测模型 | 移动平均/指数平滑/周季节 | **Poisson 回归 (GLM)** |
    | 特征 | 仅历史销量 | 周末 + 节假日 + 温度 + 降雨 + 促销 |
    | 界面 | 命令行黑箱 | **Streamlit 交互仪表板** |
    | 可解释性 | 无 | GLM 系数 + p 值 + 模型 PK |
    | 预测范围 | 仅明天 | **1-7 天** |
    | 天气 | 无 | open-meteo 免费 API |
    """)
    st.stop()

if results is None:
    st.stop()

# ═══════════════════════════════════════════════════════════
#  数据提取
# ═══════════════════════════════════════════════════════════

daily = results["daily"]
forecast_df = results["forecast_df"]
backtest_df = results["backtest_df"]
ordering_df = results["ordering_df"]
glm_summary = results["glm_summary"]
weather_available = results["weather_available"]
glm_available = results["glm_available"]
messages = results["messages"]
city_used = results["city"]

items = sorted(daily["item"].unique().tolist())
all_dates = sorted(daily["date"].unique())
primary_model = "泊松回归(GLM)" if glm_available else "基线模型"

# ═══════════════════════════════════════════════════════════
#  警告信息
# ═══════════════════════════════════════════════════════════

for msg in messages:
    st.warning(msg)
if not weather_available:
    st.info("未启用天气数据。GLM 仅使用日历特征（周末/节假日）。选择城市可获取免费天气数据。")

# ═══════════════════════════════════════════════════════════
#  Tab 1：总览
# ═══════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "总览", "预测详情", "备货建议", "模型对比", "数据探索",
])

with tab1:
    st.subheader("关键指标")

    total_revenue = daily["revenue"].sum()
    total_days = len(all_dates)
    n_items = len(items)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("历史天数", f"{total_days} 天")
    with col2:
        st.metric("品类数", f"{n_items}")
    with col3:
        st.metric("总交易额", f"元{total_revenue:,.0f}")
    with col4:
        st.metric("核心模型", primary_model)

    st.divider()

    # 明天汇总
    first_date = ordering_df["date"].min()
    today_order = ordering_df[ordering_df["date"] == first_date]

    st.subheader(f"明日备货概览（{pd.Timestamp(first_date).strftime('%Y-%m-%d')}）")

    # 柱状图：预测 vs 备货
    fig_overview = go.Figure()
    today_order_sorted = today_order.sort_values("Q_star", ascending=True)
    fig_overview.add_trace(go.Bar(
        y=today_order_sorted["item"],
        x=today_order_sorted["predicted_lambda"],
        name="预测销量（λ）",
        orientation="h",
        marker_color="#5470c6",
    ))
    fig_overview.add_trace(go.Bar(
        y=today_order_sorted["item"],
        x=today_order_sorted["Q_star"] - today_order_sorted["predicted_lambda"],
        name="安全余量",
        orientation="h",
        marker_color="#91cc75",
    ))
    fig_overview.update_layout(
        barmode="stack",
        title="各品类预测销量 + 安全余量",
        xaxis_title="杯数",
        yaxis_title="",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_overview, width="stretch")

    # 模型表现对比
    col_a, col_b = st.columns(2)
    with col_a:
        if not backtest_df.empty:
            avg_scores = backtest_df.groupby("model")["mae"].mean().reset_index()
            avg_scores = avg_scores.sort_values("mae", ascending=True)
            fig_mae = px.bar(
                avg_scores,
                x="model",
                y="mae",
                title="各模型平均 MAE（越低越好）",
                color="model",
                color_discrete_sequence=px.colors.qualitative.Set2,
                text_auto=".2f",
            )
            fig_mae.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_mae, width="stretch")
    with col_b:
        if not backtest_df.empty:
            best_counts = backtest_df[backtest_df["is_best"]].groupby("model").size().reset_index(name="count")
            fig_pie = px.pie(
                best_counts,
                values="count",
                names="model",
                title=f"最佳模型分布（共 {n_items} 个品类）",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, width="stretch")

# ═══════════════════════════════════════════════════════════
#  Tab 2：预测详情
# ═══════════════════════════════════════════════════════════

with tab2:
    st.subheader("品类预测详情")

    selected_item_t2 = st.selectbox("选择品类", items, key="item_select_t2")

    item_daily = daily[daily["item"] == selected_item_t2].sort_values("date")
    item_forecast = forecast_df[forecast_df["item"] == selected_item_t2].sort_values("date")

    # 历史 + 预测叠加图
    fig_detail = go.Figure()
    # 历史（最近60天）
    hist_data = item_daily.tail(60)
    fig_detail.add_trace(go.Scatter(
        x=hist_data["date"],
        y=hist_data["demand_qty"],
        mode="lines+markers",
        name="历史销量",
        line=dict(color="#5470c6", width=2),
        marker=dict(size=4),
    ))
    # 预测
    if not item_forecast.empty:
        fig_detail.add_trace(go.Scatter(
            x=item_forecast["date"],
            y=item_forecast["predicted_lambda"],
            mode="lines+markers",
            name="预测 λ",
            line=dict(color="#ee6666", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))
    fig_detail.update_layout(
        title=f"{selected_item_t2} — 历史销量与预测",
        xaxis_title="日期",
        yaxis_title="销量（杯）",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig_detail, width="stretch")

    # 预测数据表
    st.markdown("**预测数据表**")
    display_cols = ["date", "predicted_lambda", "model"]
    display_names = {"date": "日期", "predicted_lambda": "预测销量（杯）", "model": "使用模型"}
    st.dataframe(
        item_forecast[display_cols].rename(columns=display_names),
        width="stretch",
        hide_index=True,
    )

    # 特征贡献说明
    if glm_available and glm_summary is not None:
        st.markdown("**GLM 系数（对数尺度）**")
        st.caption("正系数 = 该特征增加需求，负系数 = 减少需求。exp(系数) = 需求倍数。")
        st.dataframe(
            glm_summary,
            width="stretch",
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════
#  Tab 3：备货建议
# ═══════════════════════════════════════════════════════════

with tab3:
    st.subheader("智能备货建议（泊松报童模型）")

    # 汇总卡片
    total_q = int(ordering_df["Q_star"].sum())
    total_profit = ordering_df["expected_profit"].sum()
    total_waste = ordering_df["expected_waste"].sum()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("预测期总备货量", f"{total_q} 杯")
    with c2:
        st.metric("预期总利润", f"元{total_profit:,.0f}")
    with c3:
        st.metric("预期总浪费", f"{total_waste:.1f} 杯")

    st.divider()

    # 备货柱状图：所有品类 x 所有预测日
    fig_order = px.bar(
        ordering_df,
        x="date",
        y="Q_star",
        color="item",
        title="各品类每日建议备货量",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_order.update_layout(
        xaxis_title="日期",
        yaxis_title="备货量（杯）",
        height=400,
    )
    st.plotly_chart(fig_order, width="stretch")

    # 备货明细表
    st.markdown("**备货明细表**")
    table_cols = ["date", "item", "predicted_lambda", "Q_star", "safety_margin",
                  "expected_profit", "expected_waste", "expected_lost_sales"]
    table_names = {
        "date": "日期", "item": "品类", "predicted_lambda": "预测λ",
        "Q_star": "最优备货", "safety_margin": "安全余量",
        "expected_profit": "预期利润", "expected_waste": "预期浪费",
        "expected_lost_sales": "预期缺货",
    }
    ordered_df_display = ordering_df[table_cols].rename(columns=table_names).sort_values(
        ["日期", "最优备货"], ascending=[True, False]
    )
    st.dataframe(ordered_df_display, width="stretch", hide_index=True)

    # CSV 下载
    csv_data = ordering_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="下载备货建议 CSV",
        data=csv_data,
        file_name=f"tea_order_{date.today().isoformat()}.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════════════════
#  Tab 4：模型对比
# ═══════════════════════════════════════════════════════════

with tab4:
    st.subheader("模型回测对比")

    if not backtest_df.empty:
        # 每个品类分组柱状图
        fig_comp = px.bar(
            backtest_df,
            x="item",
            y="mae",
            color="model",
            barmode="group",
            title="各品类、各模型 MAE 对比（越低越好）",
            color_discrete_sequence=px.colors.qualitative.Set2,
            text_auto=".2f",
        )
        fig_comp.update_layout(
            xaxis_title="品类",
            yaxis_title="MAE（平均绝对误差）",
            height=400,
        )
        st.plotly_chart(fig_comp, width="stretch")

        # 回测成绩表
        st.markdown("**回测成绩表**")
        pivot = backtest_df.pivot(index="item", columns="model", values="mae")
        # 每行最佳标记
        st.dataframe(
            pivot.style.highlight_min(axis=1, color="#91cc75"),
            width="stretch",
        )
    else:
        st.info("回测数据不可用。")

    st.divider()

    # GLM 系数详情
    if glm_summary is not None:
        st.subheader("GLM 系数详情")
        st.caption("特征对需求的影响大小。Coef > 0 表示该特征增加需求，P>|z| 越小越显著。")
        st.dataframe(glm_summary, width="stretch", hide_index=True)
    else:
        st.info("GLM 模型未训练或无系数可展示。")

# ═══════════════════════════════════════════════════════════
#  Tab 5：数据探索
# ═══════════════════════════════════════════════════════════

with tab5:
    st.subheader("数据探索")

    # 工作日 vs 周末
    col_x, col_y = st.columns(2)
    with col_x:
        if "is_weekend" in daily.columns:
            weekend_data = daily.groupby("is_weekend")["demand_qty"].mean().reset_index()
            weekend_data["label"] = weekend_data["is_weekend"].map({0: "工作日", 1: "周末"})
            fig_we = px.bar(
                weekend_data,
                x="label",
                y="demand_qty",
                title="工作日 vs 周末 日均销量",
                color="label",
                color_discrete_sequence=["#5470c6", "#ee6666"],
                text_auto=".1f",
            )
            fig_we.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="杯")
            st.plotly_chart(fig_we, width="stretch")
    with col_y:
        if "is_holiday" in daily.columns:
            holiday_data = daily.groupby("is_holiday")["demand_qty"].mean().reset_index()
            holiday_data["label"] = holiday_data["is_holiday"].map({0: "普通日", 1: "节假日"})
            fig_hol = px.bar(
                holiday_data,
                x="label",
                y="demand_qty",
                title="普通日 vs 节假日 日均销量",
                color="label",
                color_discrete_sequence=["#5470c6", "#fac858"],
                text_auto=".1f",
            )
            fig_hol.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="杯")
            st.plotly_chart(fig_hol, width="stretch")

    # 温度 vs 需求 散点图
    if weather_available and "temperature_max" in daily.columns:
        st.divider()
        st.markdown("**温度与销量关系**")
        scatter_data = daily[daily["demand_qty"] > 0].copy()
        if not scatter_data.empty:
            fig_scatter = px.scatter(
                scatter_data,
                x="temperature_max",
                y="demand_qty",
                color="item",
                trendline="lowess",
                title="温度 vs 销量（按品类着色）",
                color_discrete_sequence=px.colors.qualitative.Set2,
                opacity=0.6,
            )
            fig_scatter.update_layout(
                xaxis_title="最高温度（℃）",
                yaxis_title="日销量（杯）",
                height=450,
            )
            st.plotly_chart(fig_scatter, width="stretch")

    # 促销对比
    if "is_promotion" in daily.columns and daily["is_promotion"].sum() > 0:
        st.divider()
        promo_compare = daily.groupby("is_promotion")["demand_qty"].mean().reset_index()
        promo_compare["label"] = promo_compare["is_promotion"].map({0: "无促销", 1: "促销日"})
        fig_promo = px.bar(
            promo_compare,
            x="label",
            y="demand_qty",
            title="促销 vs 非促销 日均销量",
            color="label",
            color_discrete_sequence=["#5470c6", "#ee6666"],
            text_auto=".1f",
        )
        fig_promo.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="杯")
        st.plotly_chart(fig_promo, width="stretch")

    # 按星期几的需求分布
    st.divider()
    st.markdown("**星期几需求分布**")
    daily_temp = daily.copy()
    daily_temp["weekday"] = daily_temp["date"].apply(lambda d: ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][d.weekday()])
    weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_avg = daily_temp.groupby("weekday")["demand_qty"].mean().reset_index()
    weekday_avg["weekday"] = pd.Categorical(weekday_avg["weekday"], categories=weekday_order, ordered=True)
    weekday_avg = weekday_avg.sort_values("weekday")
    fig_wd = px.bar(
        weekday_avg,
        x="weekday",
        y="demand_qty",
        title="各星期几日均销量",
        color="weekday",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text_auto=".1f",
    )
    fig_wd.update_layout(showlegend=False, height=350, xaxis_title="", yaxis_title="杯")
    st.plotly_chart(fig_wd, width="stretch")

# ═══════════════════════════════════════════════════════════
#  页脚
# ═══════════════════════════════════════════════════════════

st.divider()
st.caption(
    f"数据范围：{all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}"
    f"  |  {len(items)} 个品类  |  核心模型：{primary_model}"
    f"  |  天气数据：{'有' if weather_available else '无'}（{city_used}）"
)
