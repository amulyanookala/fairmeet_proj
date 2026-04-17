import re
from collections import Counter
import pandas as pd

STOP_WORDS = {
    "the", "is", "am", "are", "was", "were", "a", "an", "and", "or", "to", "of", "in", "on",
    "for", "with", "this", "that", "it", "we", "i", "you", "he", "she", "they", "them",
    "our", "us", "be", "as", "at", "by", "from", "will", "can", "could", "should", "have",
    "has", "had", "do", "does", "did", "but", "if", "so", "not", "yes", "no"
}


def parse_transcript(text, sia):
    rows = []

    for line in text.strip().split("\n"):
        if ":" not in line:
            continue

        speaker, message = line.split(":", 1)
        speaker = speaker.strip()
        message = message.strip()

        if speaker and message:
            rows.append(
                {
                    "Speaker": speaker,
                    "Message": message,
                    "Words": len(message.split()),
                    "Sentiment": sia.polarity_scores(message)["compound"],
                }
            )

    return pd.DataFrame(rows)


def extract_keywords(text, n=6):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(n)]


def calculate_agenda_score(transcript_text, agenda_text):
    transcript_words = set(extract_keywords(transcript_text, 50))
    agenda_words = set(extract_keywords(agenda_text, 50))

    if not agenda_words:
        return 0.0, [], []

    matched = sorted(transcript_words & agenda_words)
    missing = sorted(agenda_words - transcript_words)
    score = round((len(matched) / len(agenda_words)) * 100, 2)

    return score, matched, missing


def detect_interruptions(df):
    counts = {}
    prev = None

    for speaker in df["Speaker"]:
        if prev is not None and speaker != prev:
            counts[speaker] = counts.get(speaker, 0) + 1
        prev = speaker

    return counts


def build_summary(df):
    summary = df.groupby("Speaker").agg(
        Words=("Words", "sum"),
        Turns=("Message", "count"),
        Sentiment=("Sentiment", "mean"),
    ).reset_index()

    total_words = summary["Words"].sum()
    total_turns = summary["Turns"].sum()

    if total_words > 0:
        summary["Participation %"] = ((summary["Words"] / total_words) * 100).round(2)
    else:
        summary["Participation %"] = 0.0

    if total_turns > 0:
        summary["Turn %"] = ((summary["Turns"] / total_turns) * 100).round(2)
    else:
        summary["Turn %"] = 0.0

    interruptions = detect_interruptions(df)
    summary["Interruptions"] = summary["Speaker"].map(interruptions).fillna(0).astype(int)

    return summary, total_words


def fairness_score(summary):
    if summary.empty:
        return 0.0

    ideal = 100 / len(summary)
    p_dev = sum(abs(x - ideal) for x in summary["Participation %"]) / len(summary)
    t_dev = sum(abs(x - ideal) for x in summary["Turn %"]) / len(summary)
    int_penalty = summary["Interruptions"].sum() * 2
    dom_penalty = sum(max(0, x - 50) for x in summary["Participation %"])

    score = 100 - (p_dev * 1.8) - (t_dev * 1.5) - int_penalty - dom_penalty
    return round(max(0, min(100, score)), 2)


def get_label(row):
    if row["Participation %"] > 45:
        return "Dominating"
    if row["Participation %"] < 10:
        return "Low Participation"
    if row["Interruptions"] > 2:
        return "Interruptive"
    if row["Sentiment"] < -0.2:
        return "Negative Tone"
    return "Balanced"


def add_labels(summary):
    summary["Label"] = summary.apply(get_label, axis=1)
    return summary


def get_verdict(fairness, agenda, summary):
    if fairness >= 80 and agenda >= 75:
        return "Healthy and Fair Meeting"
    if summary["Participation %"].max() > 50:
        return "Dominated Meeting"
    if agenda < 50:
        return "Off-Topic Meeting"
    return "Moderately Balanced Meeting"


def get_recommendations(summary, agenda):
    recs = []

    if (summary["Participation %"] < 15).any():
        recs.append("Encourage quieter participants to contribute more.")
    else:
        recs.append("Participation is reasonably balanced.")

    if (summary["Interruptions"] > 2).any():
        recs.append("Reduce interruptions for smoother discussion.")
    else:
        recs.append("Interruptions are under control.")

    if (summary["Participation %"] > 45).any():
        recs.append("One participant may be dominating the meeting.")

    if (summary["Sentiment"] < 0).any():
        recs.append("Negative tone detected in part of the conversation.")
    else:
        recs.append("Overall meeting tone is positive or neutral.")

    if agenda < 50:
        recs.append("Discussion is drifting from the agenda.")
    elif agenda < 75:
        recs.append("Agenda coverage is moderate.")
    else:
        recs.append("Meeting discussion is well aligned with the agenda.")

    return recs