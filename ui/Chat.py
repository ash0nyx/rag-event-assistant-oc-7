"""Streamlit chat for the Puls-Events assistant.

The page only does HTTP: it sends the question to the FastAPI service
(POST /ask) and the user's vote to POST /feedback. The RAG, the intent
classifier and the database all live behind the API, so this page holds no
model and no index, and any other front-end could replace it.

Run (API must be running):
    uv run streamlit run ui/Chat.py
Docker: `docker compose up` starts it on http://localhost:8501.
"""

from __future__ import annotations

import os
import re

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES = [
    "Quels concerts de jazz ce week-end ?",
    "Une activité gratuite pour les enfants ?",
    "Une exposition de photographie en novembre ?",
    "Quel temps fera-t-il demain ?",
]

st.set_page_config(page_title="Puls-Events, assistant", page_icon="🏛️", layout="centered")


# --- API calls ---------------------------------------------------------------


def ask_api(question: str) -> dict:
    response = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
    response.raise_for_status()
    return response.json()


def send_feedback(interaction_id: int, score: int, comment: str | None = None) -> bool:
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={"interaction_id": interaction_id, "score": score, "comment": comment},
            timeout=10,
        )
        return response.ok
    except requests.RequestException:
        return False


def on_vote(interaction_id: int) -> None:
    """Callback of st.feedback: 1 = thumbs up, 0 = thumbs down. A down vote is
    sent at once; the optional comment, if typed, replaces it afterwards."""
    value = st.session_state[f"vote_{interaction_id}"]
    if value is None:
        return
    score = 1 if value == 1 else -1
    if send_feedback(interaction_id, score):
        st.session_state.votes[interaction_id] = score
        st.toast("Merci pour votre retour.")
    else:
        st.toast("Le retour n'a pas pu être enregistré.")


# --- State -------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # each: role, content, and for answers: sources, intent, interaction_id
if "votes" not in st.session_state:
    st.session_state.votes = {}  # interaction_id -> 1 / -1
if "comments_sent" not in st.session_state:
    st.session_state.comments_sent = set()


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Puls-Events")
    st.write("Recommandations d'événements culturels à Paris, à partir de l'agenda officiel de la Ville.")
    try:
        api_ok = requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.RequestException:
        api_ok = False
    st.caption(f"API : {'connectée' if api_ok else 'injoignable'} ({API_URL})")
    st.caption("Chaque question est traitée indépendamment : l'assistant ne garde pas l'historique.")
    # Examples live in the sidebar too: drawn on every run, so a click always
    # registers. A click starts a fresh conversation with that question, so the
    # answer appears at the top instead of below the fold.
    st.subheader("Exemples")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"side_example_{i}", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending = example
    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Messages ----------------------------------------------------------------


_DETAIL = re.compile(r"^(Dates|Date|Lieu|Tarif|Prix|Public)\s*:", re.IGNORECASE)


def format_answer(text: str) -> str:
    """Turn the API's plain-text answer into chat markdown. The API returns
    blocks separated by blank lines: an event title, then "Dates :",
    "Lieu :", "Tarif :" lines. Here the title becomes bold and each line
    keeps its own line (markdown would otherwise merge single newlines).
    Other blocks (an introduction sentence) are shown as they are."""
    blocks = []
    for block in text.strip().split("\n\n"):
        lines = [line.strip() for line in block.strip().split("\n") if line.strip()]
        if not lines:
            continue
        escaped = [line.replace("*", "\\*").replace("_", "\\_") for line in lines]
        for i in range(len(lines) - 1):
            # A title is the line just before a "Dates : / Lieu : ..." line
            # (an intro line may share the block, so not always the first one).
            if not _DETAIL.match(lines[i]) and _DETAIL.match(lines[i + 1]):
                escaped[i] = f"**{escaped[i]}**"
        blocks.append("  \n".join(escaped))  # two spaces + newline = markdown line break
    return "\n\n".join(blocks)


def render_answer(message: dict) -> None:
    st.markdown(format_answer(message["content"]))
    if message.get("sources"):
        with st.expander(f"Sources ({len(message['sources'])} événements)"):
            for s in message["sources"]:
                st.markdown(f"- [{s['title']}]({s['url']}) · {s.get('date_range') or ''} · {s.get('venue') or ''}")
    interaction_id = message.get("interaction_id")
    if interaction_id is None:
        return
    st.feedback("thumbs", key=f"vote_{interaction_id}", on_change=on_vote, args=(interaction_id,))
    if st.session_state.votes.get(interaction_id) == -1 and interaction_id not in st.session_state.comments_sent:
        with st.form(f"comment_{interaction_id}", clear_on_submit=True, border=False):
            comment = st.text_input("Qu'est-ce qui n'allait pas ? (optionnel)", max_chars=1000)
            if st.form_submit_button("Envoyer") and comment.strip():
                if send_feedback(interaction_id, -1, comment):
                    st.session_state.comments_sent.add(interaction_id)
                    st.toast("Commentaire enregistré, merci.")
                    st.rerun()


st.title("Que faire à Paris ?")
# Read the question before drawing the page: a widget's value is known for the
# whole run wherever it is drawn. The suggestions below are then shown only
# while the chat is truly empty, never next to a question being answered
# (a button drawn once and gone on the next run loses its click).
question = st.chat_input("Votre question") or st.session_state.pop("pending", None)
if not st.session_state.messages and not question:
    st.write("Posez une question sur les sorties culturelles à Paris. Quelques idées :")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"main_example_{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # A plain if/else, not a one-line `x if c else y`: Streamlit "magic"
        # displays the value of any bare expression, here None.
        if message["role"] == "assistant":
            render_answer(message)
        else:
            st.write(message["content"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche des événements..."):
            try:
                data = ask_api(question)
                message = {
                    "role": "assistant", "content": data["answer"], "sources": data["sources"],
                    "intent": data["intent"], "interaction_id": data["interaction_id"],
                }
            except requests.RequestException as err:
                message = {"role": "assistant", "content": f"L'API ne répond pas ({err.__class__.__name__}). Vérifiez qu'elle tourne sur {API_URL}."}
        st.session_state.messages.append(message)
        render_answer(message)
