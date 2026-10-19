import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
SEED = 42

st.title("🎬 영화 유형 나누기")
st.caption("영화의 흥행 규모와 롱런 특성을 바탕으로 비슷한 영화끼리 세 유형으로 묶어 봅니다.")

FEATURE_LABELS = {
    "first_scrn_log": "스크린 수(로그)",
    "total_audi_log": "누적 관객(로그)",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수"
}
ORIGINAL_LABELS = {
    "first_scrn": "스크린 수",
    "total_audi": "누적 관객",
    "days_in_top10": "10위권 일수",
    "long_run_index": "롱런 지수"
}
FEATURES = list(FEATURE_LABELS.keys())
AVERAGE_COLUMNS = list(ORIGINAL_LABELS.keys())


@st.cache_data(ttl=3600)
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    return df


try:
    raw = load_data(DATA_URL)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 데이터 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()

required = [
    "movieCd", "movieNm", "openDt", "genre", "nation", "first_scrn",
    "first_show", "first_date", "peak", "first_week_audi", "total_audi",
    "days_in_top10"
]
missing_columns = [c for c in required if c not in raw.columns]
if missing_columns:
    st.error("데이터에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

df = raw.copy()

numeric_cols = ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 로그 변환은 양수에서만 가능하므로, 유효한 값인지 먼저 확인한다.
valid_mask = (
    df["first_scrn"].notna() & (df["first_scrn"] > 0) &
    df["total_audi"].notna() & (df["total_audi"] > 0) &
    df["days_in_top10"].notna() &
    df["first_week_audi"].notna() & (df["first_week_audi"] > 0)
)
usable = df.loc[valid_mask].copy()
usable["first_scrn_log"] = np.log10(usable["first_scrn"])
usable["total_audi_log"] = np.log10(usable["total_audi"])
usable["long_run_index"] = (
    usable["total_audi"] / usable["first_week_audi"]
).clip(upper=20)

# 상용로그와 비율 계산 후에도 결측 또는 무한대가 남은 행은 제외한다.
usable = usable.replace([np.inf, -np.inf], np.nan)
usable = usable.dropna(subset=FEATURES + AVERAGE_COLUMNS + ["movieNm"])
usable = usable.reset_index(drop=True)

st.write(
    f"**전체 영화 {len(raw):,}편 · 분석 가능한 영화 {len(usable):,}편**"
    f"  (필수 속성 누락, 0 이하 스크린/누적 관객, 첫 주 관객 0인 영화 제외)"
)

if len(usable) < 3:
    st.error("유효한 영화가 3편 미만이라 세 유형으로 나눌 수 없습니다.")
    st.stop()

st.subheader("1. 유형을 나눌 속성 고르기")
selected_features = st.multiselect(
    "K-평균 분류에 사용할 속성 (둘 이상 선택)",
    options=FEATURES,
