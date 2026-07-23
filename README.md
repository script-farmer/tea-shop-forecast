# 茶饮店销量预测 + 智能备货系统 V2

泊松回归（GLM）+ 报童模型，Streamlit 交互仪表板。整合周末、节假日、天气、促销等外部特征，透明可解释。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 方式一：Streamlit 仪表板（推荐）
streamlit run tea_shop_app.py

# 方式二：命令行
python tea_shop_v2.py --demo
python tea_shop_v2.py --demo --city 北京 --horizon 7
```

## 核心特性

- **Poisson 回归 (GLM)**：statsmodels GLM 拟合，特征包括周末、节假日、温度、降雨、促销
- **3 个基线模型对比**：7日移动平均、指数加权平滑、周季节朴素
- **泊松报童模型**：`Q* = poisson.ppf(Cu/(Cu+Co), λ)` 计算最优备货量
- **免费天气数据**：open-meteo API，无需注册，自动缓存
- **中国节假日**：chinesecalendar 库，自动识别法定节假日和调休
- **1-7 天预测**：可选择预测天数，天气预报自动匹配
- **CSV 导出**：备货建议一键下载

## CSV 数据格式

| 列名 | 必填 | 说明 |
|------|------|------|
| 交易编号 | 是 | 唯一标识 |
| 商品名称 | 是 | 品类名 |
| 数量 | 是 | 销售数量（杯） |
| 单价 | 否 | 售价，没有则无法计算利润 |
| 单位成本 | 否 | 原料成本，没有则默认=单价×40% |
| 日期 | 是 | YYYY-MM-DD 格式 |

## 仪表板功能

| Tab | 内容 |
|-----|------|
| 总览 | KPI 卡片 + 预测 vs 备货柱状图 + 模型 MAE 对比 + 最佳模型分布 |
| 预测详情 | 品类选择器 + 历史销量与预测叠加图 + GLM 系数表 |
| 备货建议 | Newsvendor 备货表 + 每日 Q* 柱状图 + CSV 下载 |
| 模型对比 | 4 模型 × 每品类分组柱状图 + 回测成绩热力图 |
| 数据探索 | 工作日/周末对比 + 节假日对比 + 温度-销量散点图 + 星期几分布 |

## 原理

### 泊松回归

```
log(λ) = β₀ + β₁·is_weekend + β₂·is_holiday + β₃·temperature_max
       + β₄·precipitation_sum + β₅·is_promotion + Σγᵢ·C(itemᵢ)
```

系数直接可解释：exp(β₁) = 周末相对工作日的需求倍数。

### 报童模型

```
Cu = 售价 − 成本（缺货损失）
Co = 成本（过剩损失）
Fr = Cu / (Cu + Co)
Q* = poisson.ppf(Fr, λ)
```

## 依赖

`streamlit plotly pandas numpy scipy statsmodels requests chinesecalendar`

## 天气数据

支持 14 个中国城市，通过 open-meteo 免费 API 获取历史和预报数据。选"无天气数据"时仅用日历特征。数据本地 Parquet 缓存，增量更新。

## License

MIT
