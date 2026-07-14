import os

import requests
import streamlit as st


st.set_page_config(page_title="患者端导诊分诊助手", layout="wide")
st.title("患者端导诊分诊助手")
st.caption("普通门诊导诊 · Agentic RAG · 证据返回")


def _short_text(text, limit=180):
    text = " ".join(str(text).split())
    return text[:limit] if len(text) > limit else text


def _score_text(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _render_summary(data):
    col1, col2, col3 = st.columns(3)
    with col1:
        if data.get("is_emergency"):
            st.metric("导诊状态", "需线下处理")
        else:
            st.metric("推荐科室", data.get("domain") or "-")
    with col2:
        st.write("备选科室:", ", ".join(data.get("candidate_domains", [])) or "-")
        st.write("路由方式:", data.get("route_method") or "-")
    with col3:
        st.write("路由原因:", data.get("route_reason") or "-")
        if data.get("is_emergency"):
            st.error(data.get("emergency_reason", ""))


def _render_evidence(item):
    metadata = item.get("metadata", {}) or {}
    section = metadata.get("section_title") or metadata.get("section") or "-"
    source = item.get("source") or metadata.get("source") or "-"
    domain = item.get("domain") or metadata.get("domain") or "-"
    retrieval_source = item.get("retrieval_source") or "-"
    score = _score_text(item.get("score"))
    content = _short_text(item.get("content", ""), 200)
    with st.container(border=True):
        st.markdown(f"**{section}**")
        st.caption(f"{domain} · {retrieval_source} · score {score}")
        st.caption(f"来源: {source}")
        st.write(content)


def _render_debug(data):
    debug = data.get("debug_info", {}) or {}
    vector_hits = debug.get("vector_hits", []) or []
    bm25_hits = debug.get("bm25_hits", []) or []
    fused_hits = debug.get("fused_hits", []) or []
    reranked_hits = debug.get("reranked_hits", []) or []
    st.caption(
        f"调试摘要: vector {len(vector_hits)} 条, bm25 {len(bm25_hits)} 条, "
        f"融合 {len(fused_hits)} 条, 重排 {len(reranked_hits)} 条"
    )
    with st.expander("查看调试详情"):
        st.write("检索查询:", debug.get("retrieval_query") or "-")
        st.write("选择的领域:", ", ".join(debug.get("selected_domains", [])) or "-")
        st.write("Planner 决策:")
        st.json(debug.get("planner", {}))
        st.write("Workflow steps:")
        st.json(debug.get("workflow_steps", []))
        st.write("仅展示前 2 条命中:")
        st.json(
            {
                "vector_hits": vector_hits[:2],
                "bm25_hits": bm25_hits[:2],
                "fused_hits": fused_hits[:2],
                "reranked_hits": reranked_hits[:2],
            }
        )


api_url = st.text_input("后端地址", value=os.getenv("STREAMLIT_API_URL", "http://127.0.0.1:8000"))
query = st.text_area("输入症状描述", value="最近胃痛反酸，饭后更明显。")

if st.button("开始导诊"):
    if not query.strip():
        st.warning("请输入症状描述")
    else:
        try:
            base_url = api_url.rstrip("/")
            resp = requests.post(base_url + "/chat", json={"query": query}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            _render_summary(data)
            st.subheader("回答")
            st.markdown(data.get("answer", ""))
            evidence = data.get("evidence", []) or []
            if data.get("is_emergency"):
                st.caption("该输入触发安全提示，不展示普通门诊检索证据。")
            elif evidence:
                st.subheader(f"证据片段（前 {min(len(evidence), 3)} 条）")
                for item in evidence[:3]:
                    _render_evidence(item)
            else:
                st.subheader("证据片段")
                st.caption("无返回证据")
            _render_debug(data)
        except Exception as exc:
            st.error(f"请求失败: {exc}")
