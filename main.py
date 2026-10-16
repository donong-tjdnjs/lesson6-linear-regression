import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. 페이지 기본 설정 및 제목
st.set_page_config(page_title="서울 연평균 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")
st.markdown("---")


# 2. 데이터 불러오기 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    # UTF-8 인코딩으로 데이터 읽기
    df = pd.read_csv(url, encoding="utf-8")

    # 날짜 데이터를 datetime 형식으로 변환 후 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 평균기온 데이터가 결측치인 행 제거
    df = df.dropna(subset=["평균기온"])

    # 연도별 관측일수 및 평균기온 계산
    yearly_summary = (
        df.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 조건 필터링: 2025년 이하 & 관측일수가 300일 이상인 해만 포함
    filtered_df = yearly_summary[
        (yearly_summary["연도"] <= 2025) & (yearly_summary["관측일수"] >= 300)
    ].sort_values("연도")

    return filtered_df


data = load_and_preprocess_data()

# 데이터 기본 정보 산출
count_years = len(data)
start_year = int(data["연도"].min())
end_year = int(data["연도"].max())

# 1) 전체 기간 선형 회귀 계산 (1차 다항식 모델)
x_all = data["연도"].values
y_all = data["연평균기온"].values

slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
trend_100y_all = slope_all * 100  # 100년당 상승 기온 (°C)

# 상관계수 계산
corr_matrix = np.corrcoef(x_all, y_all)
correlation = corr_matrix[0, 1]

# 2) 최근 20년 데이터 필터링 및 선형 회귀 계산
recent_20_start = end_year - 19
recent_20_data = data[data["연도"] >= recent_20_start]

x_recent = recent_20_data["연도"].values
y_recent = recent_20_data["연평균기온"].values

slope_recent, intercept_recent = np.polyfit(x_recent, y_recent, 1)
trend_100y_recent = slope_recent * 100  # 최근 20년 추세 기준 100년당 상승 기온 (°C)


# 3. 데이터 요약 통계 및 기울기(100년당 상승폭) 비교 표시
st.subheader("📊 핵심 요약 및 상승 속도 비교 (100년당 기온 변화량)")

# 기울기 비교 대형 카트 배치
col_trend1, col_trend2, col_trend3 = st.columns(3)
col_trend1.metric(
    label=f"🌐 전체 기간({start_year}~{end_year}) 상승 속도",
    value=f"+{trend_100y_all:.2f} °C / 100년",
    delta=f"연간 약 +{slope_all:.4f} °C",
)

# 전체 기간 속도 대비 최근 20년의 가속도 차이 계산
delta_diff = trend_100y_recent - trend_100y_all
col_trend2.metric(
    label=f"🔥 최근 20년({recent_20_start}~{end_year}) 상승 속도",
    value=f"+{trend_100y_recent:.2f} °C / 100년",
    delta=f"전체 기간 대비 {delta_diff:+.2f} °C/100년 차이",
)

col_trend3.metric(
    label="상관계수 (r)",
    value=f"{correlation:.4f}",
    delta=f"총 {count_years}개 분석 연도",
)

st.caption(
    f"ℹ️ **기초 데이터 기준:** {start_year}년부터 {end_year}년까지 관측일수 300일 이상인 총 **{count_years}개 해**의 관측값을 기반으로 합니다."
)

st.markdown("---")


# 4. 슬라이더 및 예측 기온 표시 구역
st.subheader("🔮 연도별 예상 연평균기온 예측")

selected_year = st.slider(
    "예측하고자 하는 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2030,
    step=1,
)

# 전체 기간 회귀 모델 기반 예측값
predicted_temp_all = slope_all * selected_year + intercept_all
# 최근 20년 회귀 모델 기반 예측값
predicted_temp_recent = slope_recent * selected_year + intercept_recent

p_col1, p_col2 = st.columns(2)
p_col1.metric(
    label=f"📅 {selected_year}년 예상 기온 (전체 기간 모델)",
    value=f"{predicted_temp_all:.2f} °C",
    delta=f"1900년 대비 약 +{(slope_all * (selected_year - 1900)):.2f} °C 변화 추정",
)

p_col2.metric(
    label=f"📅 {selected_year}년 예상 기온 (최근 20년 추세 모델)",
    value=f"{predicted_temp_recent:.2f} °C",
    delta=f"전체 모델 대비 {predicted_temp_recent - predicted_temp_all:+.2f} °C 차이",
)

st.markdown("---")


# 5. Plotly 산점도 및 두 회귀 직선 시각화
st.subheader("📈 서울 연평균기온 추이 및 추세선 비교")

# 회귀 직선 생성을 위한 연도 범위 (1900년 ~ 2100년)
line_x = np.linspace(1900, 2100, 201)
line_y_all = slope_all * line_x + intercept_all
line_y_recent = slope_recent * line_x + intercept_recent

fig = go.Figure()

# 관측 데이터 산점도 추가
fig.add_trace(
    go.Scatter(
        x=data["연도"],
        y=data["연평균기온"],
        mode="markers",
        name="실제 관측 연평균기온",
        marker=dict(size=8, color="#1f77b4", opacity=0.7),
        hovertemplate="<b>연도:</b> %{x}년<br><b>연평균기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

# 전체 기간 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y_all,
        mode="lines",
        name=f"전체 기간 추세선 (+{trend_100y_all:.2f}°C/100년)",
        line=dict(color="#d62728", width=2.5, dash="dash"),
        hovertemplate="<b>[전체 모델] 연도:</b> %{x:.0f}년<br><b>예상기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

# 최근 20년 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y_recent,
        mode="lines",
        name=f"최근 20년 추세선 (+{trend_100y_recent:.2f}°C/100년)",
        line=dict(color="#ff7f0e", width=2.5),
        hovertemplate="<b>[최근 20년 모델] 연도:</b> %{x:.0f}년<br><b>예상기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

# 선택한 예측 지점 표시 (전체 모델 기준)
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp_all],
        mode="markers+text",
        name=f"선택 연도 ({selected_year}년)",
        marker=dict(size=14, color="#2ca02c", symbol="star"),
        text=[f"{selected_year}년 ({predicted_temp_all:.2f}°C)"],
        textposition="top center",
        hovertemplate="<b>선택 연도:</b> %{x}년<br><b>예측기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

fig.update_layout(
    title="서울 연평균기온 변화 및 모델별 미래 예측 비교",
    xaxis_title="연도 (Year)",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"💡 **분석 결과:** 전체 기간 분석 결과 100년에 **약 +{trend_100y_all:.2f} °C** 상승하는 추세이며, "
    f"최근 20년 데이터만 따로 추출하면 100년에 **약 +{trend_100y_recent:.2f} °C** 상승하여 "
    f"온난화 속도가 더 가속화되고 있음을 확인할 수 있습니다."
)
