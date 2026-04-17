import re
from collections import Counter
import pandas as pd

STOP_WORDS = {
    "the","is","am","are","was","were","a","an","and","or","to","of","in","on",
    "for","with","this","that","it","we","i","you","he","she","they","them",
    "our","us","be","as","at","by","from","will","can","could","should","have",
    "has","had","do","does","did","but","if","so","not","yes","no"
}

def parse_transcript(text, sia):
    rows = []
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        speaker, message = line.split(":", 1)
        speaker, message = speaker.strip(), message.strip()
        if speaker and message:
            rows.append({
                "Speaker": speaker,
                "Message": message,
                "Words": len(message.split()),
                "Sentiment": sia.polarity_scores(message)["compound"]
            })
    return pd.DataFrame(rows)

def extract_keywords(text, n=6):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(n)]

def agenda_score(transcript_text, agenda_text):
    t = set(extract_keywords(transcript_text, 50))
    a = set(extract_keywords(agenda_text, 50))
    if not a:
        return 0.0, [], []
    matched = sorted(t & a)
    missing = sorted(a - t)
    score = round((len(matched) / len(a)) * 100, 2)
    return score, matched, missing