"""Dashboard: what users ask, how satisfied they are, how fast the system is.

Reads the interactions table directly (same SQLite file as the API, shared
through the data/ volume): the dashboard only needs data, not the model, so
it does not go through the API. Read-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from api.database import InteractionStore

st.set_page_config(page_title="Puls-Events, tableau de bord", layout="wide")
st.title("Tableau de bord")

SERIES = "#2a78d6"  # single series everywhere: one hue, no legend needed
INTENT_LABELS = {"events": "Événements (RAG)", "chat": "Conversation", "off_topic": "Hors sujet"}
FEEDBACK_LABELS = {1: "Positif", -1: "Négatif"}


@st.cache_data(ttl=30)
def load(limit: int = 5000) -> tuple[pd.DataFrame, dict]:
    store = InteractionStore()
    return pd.DataFrame(store.list_interactions(limit=limit)), store.summary()


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


df, summary = load()
if st.button("Rafraîchir"):
    load.clear()
    st.rerun()

if df.empty:
    st.info("Aucune interaction enregistrée pour l'instant. Posez des questions depuis la page de chat.")
    st.stop()

df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df["jour"] = df["created_at"].dt.tz_convert("Europe/Paris").dt.date
df["intention"] = df["intent"].map(INTENT_LABELS).fillna(df["intent"])
df["avis"] = df["feedback_score"].map(FEEDBACK_LABELS).fillna("Sans avis")
events = df[df["intent"] == "events"]

# --- KPI row -----------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Questions", summary["total"])
k2.metric("Satisfaction", pct(summary["satisfaction"]), help="Part de pouces levés parmi les réponses notées.")
k3.metric("Taux de retour", pct(summary["feedback_rate"]), help="Part des réponses ayant reçu un vote.")
k4.metric("Latence médiane (RAG)", f"{events['latency_ms'].median() / 1000:.1f} s" if not events.empty else "n/a")
k5.metric("Hors sujet / conversation", pct((df["intent"] != "events").mean()),
          help="Questions traitées sans RAG grâce au classifieur d'intention.")

# --- Charts --------------------------------------------------------------------


def bar(data: pd.DataFrame, x: str, y: str, title: str, horizontal: bool = False) -> alt.Chart:
    enc_value = alt.X if horizontal else alt.Y
    enc_cat = alt.Y if horizontal else alt.X
    return (
        alt.Chart(data, title=title)
        .mark_bar(color=SERIES, cornerRadiusEnd=4, **({"size": 18} if horizontal else {}))
        .encode(
            enc_cat(f"{x}:N" if horizontal else f"{x}:T", title=None, sort="-x" if horizontal else None),
            enc_value(f"{y}:Q", title=None),
            tooltip=[alt.Tooltip(f"{x}:{'N' if horizontal else 'T'}", title=x.capitalize()), alt.Tooltip(f"{y}:Q", title="Nombre")],
        )
        .properties(height=240)
    )


c1, c2 = st.columns(2)
per_day = df.groupby("jour").size().reset_index(name="questions")
per_day["jour"] = pd.to_datetime(per_day["jour"])
c1.altair_chart(bar(per_day, "jour", "questions", "Questions par jour"), use_container_width=True)
by_intent = df.groupby("intention").size().reset_index(name="questions")
c2.altair_chart(bar(by_intent, "intention", "questions", "Questions par intention", horizontal=True), use_container_width=True)

c3, c4 = st.columns(2)
if not events.empty:
    latency = events.groupby("jour")["latency_ms"].median().div(1000).reset_index(name="secondes")
    latency["jour"] = pd.to_datetime(latency["jour"])
    c3.altair_chart(
        alt.Chart(latency, title="Latence médiane des réponses RAG (s)")
        .mark_line(color=SERIES, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=SERIES))
        .encode(
            alt.X("jour:T", title=None),
            alt.Y("secondes:Q", title=None, scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("jour:T", title="Jour"), alt.Tooltip("secondes:Q", title="Secondes", format=".1f")],
        )
        .properties(height=240),
        use_container_width=True,
    )
by_feedback = df.groupby("avis").size().reset_index(name="réponses")
c4.altair_chart(bar(by_feedback, "avis", "réponses", "Avis des utilisateurs", horizontal=True), use_container_width=True)

# --- Negative feedback ---------------------------------------------------------
st.subheader("Réponses jugées mauvaises")
negatives = df[df["feedback_score"] == -1]
if negatives.empty:
    st.write("Aucun avis négatif.")
else:
    for _, row in negatives.head(20).iterrows():
        with st.expander(f"#{row['id']} · {row['question']}"):
            st.markdown(f"**Commentaire :** {row['feedback_comment'] or '(aucun)'}")
            st.markdown(f"**Réponse :** {row['answer']}")
            for s in row["sources"] or []:
                st.markdown(f"- [{s['title']}]({s['url']}) · {s.get('date_range') or ''}")

# --- All interactions ----------------------------------------------------------
st.subheader("Toutes les interactions")
choice = st.selectbox("Filtrer par avis", ["Tous", "Positif", "Négatif", "Sans avis"])
view = df if choice == "Tous" else df[df["avis"] == choice]
st.dataframe(
    view[["id", "created_at", "question", "intention", "avis", "feedback_comment", "latency_ms"]].rename(columns={
        "created_at": "date", "feedback_comment": "commentaire", "latency_ms": "latence (ms)"}),
    use_container_width=True, hide_index=True,
)

# --- Offline evaluation ----------------------------------------------------------
st.subheader("Dernière évaluation hors ligne")
results = Path("evaluation/results")
rag_path, intent_path = results / "latest.json", results / "intent_latest.json"
if rag_path.exists():
    s = json.loads(rag_path.read_text(encoding="utf-8"))["summary"]
    e1, e2, e3, e4, e5 = st.columns(5)
    e1.metric("Rappel retrieval", f"{s['retrieval_recall']:.2f}")
    e2.metric("Précision retrieval", f"{s['retrieval_precision']:.2f}")
    e3.metric("Fidélité (Ragas)", f"{s['faithfulness']:.2f}" if s.get("faithfulness") is not None else "n/a")
    e4.metric("Pertinence (Ragas)", f"{s['answer_relevancy']:.2f}" if s.get("answer_relevancy") is not None else "n/a")
    if intent_path.exists():
        e5.metric("Classifieur d'intention", f"{json.loads(intent_path.read_text(encoding='utf-8'))['accuracy']:.0%}")
    st.caption("Jeu de 14 questions annotées, scripts/evaluate_rag.py et scripts/evaluate_intent.py.")
else:
    st.caption("Aucun résultat d'évaluation trouvé dans evaluation/results/.")
if os.environ.get("LANGFUSE_BASE_URL"):
    st.caption("Le détail de chaque question (étapes, tokens, score utilisateur) est dans Langfuse, onglet Tracing.")
