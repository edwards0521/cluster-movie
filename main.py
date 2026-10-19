with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_x"
    )
with col_y:
    y_index = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=y_index,
        format_func=lambda x: FEATURE_LABELS[x],
        key="scatter_y"
    )

fig2 = px.scatter(
    usable,
    x=x_feature,
    y=y_feature,
    color="유형",
    hover_name="movieNm",
    hover_data={
        "유형": True,
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "first_week_audi": ":,",
        x_feature: True,
        y_feature: True,
    },
    category_orders={"유형": ["㉮", "㉯", "㉰"]},
    labels={
        x_feature: FEATURE_LABELS[x_feature],
        y_feature: FEATURE_LABELS[y_feature],
        "유형": "영화 유형",
        "first_scrn": "스크린 수",
        "total_audi": "누적 관객",
        "days_in_top10": "10위권 일수",
        "first_week_audi": "첫 주 관객"
    },
    title="선택한 두 속성으로 본 영화 유형"
)
fig2.update_traces(marker={"size": 7, "opacity": 0.75})
fig2.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 분류에 사용할 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox(
            "X축 속성", selected_features, index=0,
            format_func=lambda x: FEATURE_LABELS[x], key="x3"
        )
    with c2:
        y3 = st.selectbox(
            "Y축 속성", selected_features, index=1,
            format_func=lambda x: FEATURE_LABELS[x], key="y3"
        )
    with c3:
        z3 = st.selectbox(
            "Z축 속성", selected_features, index=2,
            format_func=lambda x: FEATURE_LABELS[x], key="z3"
        )

    fig3 = px.scatter_3d(
        usable,
        x=x3, y=y3, z=z3,
        color="유형",
        hover_name="movieNm",
        hover_data={
            "유형": True,
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "first_week_audi": ":,"
        },
        category_orders={"유형": ["㉮", "㉯", "㉰"]},
        labels={
            x3: FEATURE_LABELS[x3],
            y3: FEATURE_LABELS[y3],
            z3: FEATURE_LABELS[z3],
            "유형": "영화 유형",
            "first_scrn": "스크린 수",
            "total_audi": "누적 관객",
            "days_in_top10": "10위권 일수",
            "first_week_audi": "첫 주 관객"
        },
        title="세 속성으로 본 영화 유형"
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("3차원 그래프는 마우스로 드래그해 회전하고, 휠이나 확대 도구로 크기를 조절할 수 있습니다.")

st.subheader("4. 유형별 요약")
summary = usable.groupby("유형").agg(
    편수=("movieNm", "count"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객_평균=("total_audi", "mean"),
    십위권일수_평균=("days_in_top10", "mean"),
    롱런지수_평균=("long_run_index", "mean")
).reindex(["㉮", "㉯", "㉰"]).reset_index()

summary = summary.rename(columns={
    "유형": "영화 유형",
    "편수": "영화 편수",
    "스크린수_평균": "스크린 수 평균",
    "누적관객_평균": "누적 관객 평균",
    "십위권일수_평균": "10위권 일수 평균",
    "롱런지수_평균": "롱런 지수 평균"
})
display_summary = summary.copy()
for col in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"]:
    display_summary[col] = display_summary[col].round(2)
display_summary["영화 편수"] = display_summary["영화 편수"].astype(int)
display_summary["스크린 수 평균"] = display_summary["스크린 수 평균"].map(lambda x: f"{x:,.2f}")
display_summary["누적 관객 평균"] = display_summary["누적 관객 평균"].map(lambda x: f"{x:,.2f}")
display_summary["10위권 일수 평균"] = display_summary["10위권 일수 평균"].map(lambda x: f"{x:,.2f}")
display_summary["롱런 지수 평균"] = display_summary["롱런 지수 평균"].map(lambda x: f"{x:,.2f}")
st.dataframe(display_summary, hide_index=True, use_container_width=True)

st.subheader("5. 유형별 누적 관객 상위 5편")
for cluster_name in ["㉮", "㉯", "㉰"]:
    subset = usable[usable["유형"] == cluster_name].sort_values(
        "total_audi", ascending=False
    ).head(5)
    with st.expander(f"{cluster_name} 유형 · 누적 관객 상위 {len(subset)}편", expanded=True):
        if subset.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top = subset[["movieNm", "total_audi", "first_week_audi", "days_in_top10"]].copy()
            top.columns = ["영화 제목", "누적 관객", "첫 주 관객", "10위권 일수"]
            top["누적 관객"] = top["누적 관객"].map(lambda x: f"{int(x):,}")
            top["첫 주 관객"] = top["첫 주 관객"].map(lambda x: f"{int(x):,}")
            top["10위권 일수"] = top["10위권 일수"].map(lambda x: f"{int(x):,}")
            st.dataframe(top, hide_index=True, use_container_width=True)

st.caption("참고: 로그 변환 속성은 분류와 산점도에서 로그 값으로 표시됩니다. 유형별 요약 표의 평균은 원래 단위 기준입니다.")
