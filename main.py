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

# 선형 회귀 계산 (1차 다항식 모델)
x = data["연도"].values
y = data["연평균기온"].values

# slope(기울기), intercept(절편)
slope, intercept = np.polyfit(x, y, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(x, y)
correlation = corr_matrix[0, 1]


# 3. 데이터 요약 통계 정보 표시
st.subheader("📊 데이터 기본 정보")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 개수", f"{count_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{correlation:.4f}")

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

# 선택한 연도의 예상 기온 계산 (Linear Regression)
predicted_temp = slope * selected_year + intercept

st.metric(
    label=f"📅 {selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C",
    delta=f"1900년 대비 약 +{(slope * (selected_year - 1900)):.2f} °C 변화 추정",
)

st.markdown("---")


# 5. Plotly 산점도 및 회귀 직선 시각화
st.subheader("📈 서울 연평균기온 추이 및 회귀 직선")

# 회귀 직선 생성을 위한 연도 범위 (1900년 ~ 2100년까지 확장 표시)
line_x = np.linspace(1900, 2100, 201)
line_y = slope * line_x + intercept

fig = go.Figure()

# 관측 데이터 산점도 추가
fig.add_trace(
    go.Scatter(
        x=data["연도"],
        y=data["연평균기온"],
        mode="markers",
        name="실제 관측 연평균기온",
        marker=dict(size=8, color="#1f77b4"),
        hovertemplate="<b>연도:</b> %{x}년<br><b>연평균기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="선형 회귀 직선",
        line=dict(color="#d62728", width=2, dash="dash"),
        hovertemplate="<b>연도:</b> %{x:.0f}년<br><b>예상기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

# 사용자 선택 연도 강조 점 추가
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers+text",
        name="선택한 예측 지점",
        marker=dict(size=14, color="#2ca02c", symbol="star"),
        text=[f"{selected_year}년 ({predicted_temp:.2f}°C)"],
        textposition="top center",
        hovertemplate="<b>선택 연도:</b> %{x}년<br><b>예측기온:</b> %{y:.2f} °C<extra></extra>",
    )
)

fig.update_layout(
    title="서울 연평균기온 변화 및 미래 예측 모델",
    xaxis_title="연도 (Year)",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend=dict(x=0.01, y=0.99),
)

st.plotly_chart(fig, use_container_width=True)

st.info(
    f"💡 **분석 요약:** 회귀 직선 공식은 **`y = {slope:.4f} * x + ({intercept:.2f})`** 이며, "
    f"사용된 데이터는 {start_year}년부터 {end_year}년까지 총 {count_years}개 해의 관측값입니다. "
    f"상관계수는 **{correlation:.4f}**로 강한 양의 상관관계를 보여줍니다."
)
