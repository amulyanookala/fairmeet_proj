import os
import tempfile
import whisper

model = whisper.load_model("base")


def transcribe_audio(uploaded_file):
    if uploaded_file is None:
        return ""

    # reset pointer in case Streamlit already read the file
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    data = uploaded_file.read()
    if not data:
        return "Speaker 1: Audio file is empty or could not be read."

    suffix = os.path.splitext(getattr(uploaded_file, "name", "audio.wav"))[1].lower()
    if not suffix:
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name

    try:
        if os.path.getsize(path) == 0:
            return "Speaker 1: Audio file is empty."

        result = model.transcribe(path)
        text = result.get("text", "").strip()

        if not text:
            return "Speaker 1: No speech detected in the audio."

        parts = [p.strip() for p in text.split(".") if p.strip()]
        lines = []
        speaker = 1

        for part in parts:
            lines.append(f"Speaker {speaker}: {part}")
            speaker = 2 if speaker == 1 else 1

        return "\n".join(lines) if lines else f"Speaker 1: {text}"

    except Exception as e:
        return f"Speaker 1: Audio transcription failed - {str(e)}"

    finally:
        if os.path.exists(path):
            os.remove(path)