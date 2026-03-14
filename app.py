from flask import Flask, request, Response, jsonify
import requests
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "PTT relay is running.", 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

TEXT_MODEL = "gpt-4o-mini"
STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"

SYSTEM_PROMPT = (
    "You are a concise push-to-talk embedded assistant. "
    "Keep replies short, clear, and natural."
)

def extract_output_text(data):
    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts = []
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    t = content.get("text", "")
                    if t:
                        parts.append(t)
    return "".join(parts).strip()

@app.route("/ptt", methods=["POST"])
def ptt():
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is missing"}), 500

    wav_bytes = request.data
    if not wav_bytes:
        return jsonify({"error": "No audio received"}), 400

    # 1) Speech to text
    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav")
    }
    data = {
        "model": STT_MODEL,
        "language": "en"
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files=files,
        data=data,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({"stage": "transcription", "error": r.text}), 500

    transcript = r.json().get("text", "").strip()
    if not transcript:
        transcript = "I didn't catch that."

    # 2) Text reply
    body = {
        "model": TEXT_MODEL,
        "max_output_tokens": 80,
        "input": [
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": SYSTEM_PROMPT}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": transcript}
                ]
            }
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json=body,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({"stage": "response", "error": r.text, "transcript": transcript}), 500

    reply_text = extract_output_text(r.json())
    if not reply_text:
        reply_text = "Sorry, I do not have an answer right now."

    # 3) Text to speech
    tts_body = {
        "model": TTS_MODEL,
        "voice": "alloy",
        "input": reply_text,
        "response_format": "wav"
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        json=tts_body,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({"stage": "tts", "error": r.text, "reply": reply_text}), 500

    return Response(r.content, mimetype="audio/wav")
