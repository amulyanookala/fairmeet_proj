import os
import matplotlib.pyplot as plt
import nltk
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

from modules.audio import transcribe_audio
from modules.analysis import (
    parse_transcript,
    extract_keywords,
    calculate_agenda_score,
    build_summary,
    fairness_score,
    add_labels,
    get_verdict,
    get_recommendations,
)
from modules.database import init_db, save_meeting, get_history, delete_meeting
from modules.pdf_report import generate_pdf

st.set_page_config(page_title="FairMeet", layout="wide")
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

if "db_init" not in st.session_state:
    init_db()
    st.session_state["db_init"] = True


def apply_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef5ff 0%, #dcecff 100%);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
    }

    div[data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3d91 0%, #1456c3 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        background: linear-gradient(135deg, #0f62fe, #5b8def);
        padding: 24px;
        border-radius: 20px;
        color: white;
        margin-top: 0rem;
        margin-bottom: 18px;
        box-shadow: 0 8px 22px rgba(15,98,254,0.20);
    }

    .verdict-good, .verdict-mid, .verdict-bad {
        padding: 14px;
        border-radius: 14px;
        font-weight: 700;
        border-left: 6px solid;
        margin-bottom: 10px;
    }

    .verdict-good {
        background: #e6f6ff;
        color: #0b5394;
        border-color: #1c7ed6;
    }

    .verdict-mid {
        background: #e9f0ff;
        color: #174ea6;
        border-color: #3b82f6;
    }

    .verdict-bad {
        background: #dcebff;
        color: #103d77;
        border-color: #2563eb;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f2f7ff 100%);
        border: 1px solid #d6e6ff;
        padding: 10px;
        border-radius: 14px;
        box-shadow: 0 4px 10px rgba(18,76,163,0.05);
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(90deg, #0f62fe, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        min-height: 44px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #dce9ff;
        border-radius: 12px 12px 0 0;
        color: #123c73;
        padding: 10px 16px;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: #0f62fe !important;
        color: white !important;
    }

    h1, h2, h3 {
        color: #173b75;
    }
    </style>
    """, unsafe_allow_html=True)


def show_header():
    st.markdown("""
    <div class="hero">
        <h1 style="margin:0;">FairMeet</h1>
        <p style="margin:8px 0 0 0;">Smart Participation and Engagement Analytics for Online Meetings</p>
    </div>
    """, unsafe_allow_html=True)


def show_verdict_box(verdict: str):
    css_class = (
        "verdict-good"
        if "Healthy" in verdict
        else "verdict-mid"
        if "Moderately" in verdict
        else "verdict-bad"
    )
    st.markdown(f'<div class="{css_class}">{verdict}</div>', unsafe_allow_html=True)


def plot_bar(df, x, y, title):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(df[x], df[y])
    ax.set_title(title)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_pie(df):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.pie(df["Participation %"], labels=df["Speaker"], autopct="%1.1f%%")
    ax.set_title("Speaker Share")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def reset_analysis():
    for key in ["analysis_ready", "result", "raw_audio_text"]:
        st.session_state.pop(key, None)


def analyze(meeting_name, agenda_text, transcript_text):
    df = parse_transcript(transcript_text, sia)
    if df.empty:
        return False

    summary, total_words = build_summary(df)
    summary = add_labels(summary)

    full_text = " ".join(df["Message"].tolist())
    fairness = fairness_score(summary)
    agenda, matched, missing = calculate_agenda_score(full_text, agenda_text)
    engagement = round((fairness + agenda) / 2, 2)

    st.session_state["result"] = {
        "meeting_name": meeting_name,
        "summary": summary,
        "total_words": total_words,
        "fairness": fairness,
        "agenda": agenda,
        "engagement": engagement,
        "verdict": get_verdict(fairness, agenda, summary),
        "matched": matched,
        "missing": missing,
        "keywords": extract_keywords(full_text),
        "recs": get_recommendations(summary, agenda),
    }
    st.session_state["analysis_ready"] = True
    return True


def input_screen():
    left, right = st.columns(2)
    transcript_text = ""

    with left:
        st.subheader("Meeting Setup")
        meeting_name = st.text_input("Meeting Name", "Team Meeting")
        agenda_text = st.text_area(
            "Meeting Agenda",
            "project progress dashboard testing database presentation",
            height=150,
        )
        mode = st.radio(
            "Input Type",
            ["Text Transcript", "Upload Transcript File", "Audio File", "Record Audio"],
            horizontal=True,
        )

    with right:
        st.subheader("Input Source")

        if mode == "Text Transcript":
            transcript_text = st.text_area(
                "Transcript",
                """Alice: Good morning everyone
Bob: Today we will discuss project progress
Alice: I completed the dashboard module
Charlie: I am working on the database connection
Bob: We need faster testing
Charlie: I will complete SQL integration today""",
                height=260,
            )

        elif mode == "Upload Transcript File":
            txt_file = st.file_uploader("Upload Transcript (.txt)", type=["txt"])
            if txt_file is not None:
                transcript_text = txt_file.read().decode("utf-8")
                transcript_text = st.text_area("Transcript Preview", transcript_text, height=260)

        elif mode == "Audio File":
            audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
            if audio_file is not None:
                st.audio(audio_file)

                if st.button("Transcribe Audio File", use_container_width=True):
                    with st.spinner("Transcribing audio..."):
                        st.session_state["raw_audio_text"] = transcribe_audio(audio_file)

            raw_text = st.session_state.get("raw_audio_text", "")
            if raw_text:
                transcript_text = st.text_area("Auto Transcript", raw_text, height=260)

        else:
            recorded_audio = st.audio_input("Record Audio")
            if recorded_audio is not None:
                st.audio(recorded_audio)

                if st.button("Transcribe Recorded Audio", use_container_width=True):
                    with st.spinner("Transcribing recorded audio..."):
                        st.session_state["raw_audio_text"] = transcribe_audio(recorded_audio)

            raw_text = st.session_state.get("raw_audio_text", "")
            if raw_text:
                transcript_text = st.text_area("Auto Transcript", raw_text, height=260)

    _, center_col, _ = st.columns([1, 1.1, 1])
    with center_col:
        if st.button("Analyze Meeting", type="primary", use_container_width=True):
            if analyze(meeting_name, agenda_text, transcript_text):
                st.rerun()
            else:
                st.error("Transcript is invalid. Use speaker-formatted transcript.")


def result_screen():
    r = st.session_state["result"]

    top_left, top_right = st.columns([1, 5], gap="small")
    with top_left:
        if st.button("← Back", use_container_width=True):
            reset_analysis()
            st.rerun()

    with top_right:
        st.subheader(f"Analysis Dashboard — {r['meeting_name']}")

    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("Speakers", len(r["summary"]))
    m2.metric("Words", int(r["total_words"]))
    m3.metric("Fairness", r["fairness"])
    m4.metric("Agenda", r["agenda"])
    m5.metric("Engagement", r["engagement"])

    st.subheader("Final Verdict")
    show_verdict_box(r["verdict"])

    if r["fairness"] > 75:
        st.success("High confidence in balanced discussion.")
    elif r["fairness"] > 50:
        st.warning("Moderate confidence in fairness.")
    else:
        st.error("Low fairness detected.")

    tab1, tab2, tab3 = st.tabs(["Overview", "Insights", "Report"])

    with tab1:
        a, b = st.columns(2, gap="small")
        with a:
            st.subheader("Participation Chart")
            plot_bar(r["summary"], "Speaker", "Participation %", "Participation")

        with b:
            st.subheader("Sentiment Chart")
            plot_bar(r["summary"], "Speaker", "Sentiment", "Sentiment")

        c, d = st.columns([1, 2], gap="small")
        with c:
            st.subheader("Participation Distribution")
            plot_pie(r["summary"])

        with d:
            st.subheader("Speaker Summary")
            display_df = r["summary"].copy()
            display_df["Sentiment"] = display_df["Sentiment"].round(2)
            display_df["Participation %"] = display_df["Participation %"].round(1)
            display_df["Turn %"] = display_df["Turn %"].round(1)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=min(420, 60 + len(display_df) * 35),
            )

    with tab2:
        a, b = st.columns(2, gap="small")

        with a:
            st.subheader("Topic Insights")
            st.write("**Top Keywords:**", ", ".join(r["keywords"]) or "None")
            st.write("**Matched Agenda Words:**", ", ".join(r["matched"]) or "None")
            st.write("**Missing Agenda Words:**", ", ".join(r["missing"]) or "None")

        with b:
            st.subheader("Recommendations")
            for rec in r["recs"]:
                if any(x in rec for x in ["Encourage", "Reduce", "Negative", "dominating", "drifting"]):
                    st.warning(rec)
                else:
                    st.success(rec)

    with tab3:
        st.subheader("Save and Export")
        a, b = st.columns(2, gap="small")

        with a:
            if st.button("Save Meeting", use_container_width=True):
                save_meeting(
                    r["meeting_name"],
                    r["fairness"],
                    r["agenda"],
                    r["engagement"],
                    r["verdict"],
                )
                st.success("Meeting saved successfully.")
                st.rerun()

        with b:
            pdf_path = generate_pdf(
                r["meeting_name"],
                r["fairness"],
                r["agenda"],
                r["engagement"],
                r["verdict"],
                r["keywords"],
                r["recs"],
            )
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Download PDF Report",
                    f,
                    file_name=f"{r['meeting_name']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            os.remove(pdf_path)


def history_screen():
    st.subheader("Saved Meetings")
    df = get_history()

    search = st.text_input("Search Meeting")
    if search:
        df = df[df["meeting_name"].astype(str).str.contains(search, case=False, na=False)]

    if df.empty:
        st.info("No meetings found.")
        return

    st.dataframe(df, use_container_width=True, height=420, hide_index=True)

    st.divider()
    st.subheader("Delete Meeting")

    selected_id = st.selectbox("Select Meeting ID", df["id"])

    if st.button("Delete Selected Meeting"):
        delete_meeting(selected_id)
        st.success("Deleted successfully.")
        st.rerun()


apply_css()
show_header()

if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False

menu = st.sidebar.radio("Navigation", ["Demo", "History"])

if menu == "History":
    history_screen()
else:
    if st.session_state["analysis_ready"]:
        result_screen()
    else:
        input_screen()