from flask import Flask, request, Response, jsonify
import requests
import os
import time
import uuid
import urllib.parse

app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------- Models --------
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-4.1-mini")
STT_MODEL  = os.environ.get("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL  = os.environ.get("TTS_MODEL", "gpt-4o-mini-tts")

# Toggle live web lookup for current-info questions
ENABLE_WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true"

SYSTEM_PROMPT = (
    "You are a concise push-to-talk embedded assistant. "
    "Be intelligent, practical, and natural. "
    "Keep replies short enough to sound good when spoken aloud. "
    "If the user asks for current or live information, use web search when available. "
    "Avoid long lists unless specifically requested."
)

# -------- Helpers --------

def now_ms():
    return int(time.time() * 1000)

def safe_header_value(text: str, max_len: int = 1500) -> str:
    """
    HTTP headers must stay small and ASCII-safe.
    URL-encode and truncate.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len - 1] + "…"
    return urllib.parse.quote(text, safe="")

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

def looks_like_live_info_question(text: str) -> bool:
    """
    Very simple heuristic to decide whether to enable web search.
    """
    if not text:
        return False

    q = text.lower()
    live_keywords = [
        "today", "now", "latest", "current", "news", "weather",
        "price", "stock", "score", "recent", "live", "this week",
        "this month", "yesterday", "tomorrow", "who is the current",
        "what happened", "update", "traffic"
    ]
    return any(k in q for k in live_keywords)

def make_request_headers(request_id: str):
    return {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "X-Client-Request-Id": request_id
    }

# -------- Health --------

@app.route("/", methods=["GET"])
def home():
    return "PTT relay is running.", 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200

# -------- Main PTT endpoint --------

@app.route("/ptt", methods=["POST"])
def ptt():
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is missing"}), 500

    request_id = str(uuid.uuid4())
    t0 = now_ms()

    wav_bytes = request.data
    if not wav_bytes:
        return jsonify({"error": "No audio received"}), 400

    # ------------------------------------------------------------
    # 1) Speech-to-text
    # ------------------------------------------------------------
    stt_start = now_ms()

    files = {
        "file": ("audio.wav", wav_bytes, "audio/wav")
    }
    data = {
        "model": STT_MODEL,
        "language": "en"
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=make_request_headers(request_id),
        files=files,
        data=data,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({
            "stage": "transcription",
            "error": r.text,
            "request_id": request_id
        }), 500

    stt_json = r.json()
    transcript = stt_json.get("text", "").strip()
    if not transcript:
        transcript = "I didn't catch that."

    stt_ms = now_ms() - stt_start

    # ------------------------------------------------------------
    # 2) Text reply (Responses API)
    # ------------------------------------------------------------
    llm_start = now_ms()

    use_web_search = ENABLE_WEB_SEARCH and looks_like_live_info_question(transcript)

    body = {
        "model": TEXT_MODEL,
        "max_output_tokens": 120,
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

    if use_web_search:
        body["tools"] = [{"type": "web_search_preview"}]

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            **make_request_headers(request_id),
            "Content-Type": "application/json"
        },
        json=body,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({
            "stage": "response",
            "error": r.text,
            "transcript": transcript,
            "request_id": request_id
        }), 500

    response_json = r.json()
    reply_text = extract_output_text(response_json)
    if not reply_text:
        reply_text = "Sorry, I do not have an answer right now."

    llm_ms = now_ms() - llm_start

    # ------------------------------------------------------------
    # 3) Text-to-speech
    # ------------------------------------------------------------
    tts_start = now_ms()

    tts_body = {
        "model": TTS_MODEL,
        "voice": "alloy",
        "input": reply_text,
        "response_format": "pcm"
    }

    r = requests.post(
        "https://api.openai.com/v1/audio/speech",
        headers={
            **make_request_headers(request_id),
            "Content-Type": "application/json"
        },
        json=tts_body,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({
            "stage": "tts",
            "error": r.text,
            "reply": reply_text,
            "transcript": transcript,
            "request_id": request_id
        }), 500

    tts_ms = now_ms() - tts_start
    total_ms = now_ms() - t0

    # Put useful metadata in headers so the ESP can show text
    headers = {
        "X-Audio-Format": "pcm_s16le_24khz_mono",
        "X-Request-Id": request_id,
        "X-Transcript": safe_header_value(transcript),
        "X-Reply-Text": safe_header_value(reply_text),
        "X-Text-Model": TEXT_MODEL,
        "X-STT-Model": STT_MODEL,
        "X-TTS-Model": TTS_MODEL,
        "X-Web-Search": "1" if use_web_search else "0",
        "X-Time-STT-Ms": str(stt_ms),
        "X-Time-LLM-Ms": str(llm_ms),
        "X-Time-TTS-Ms": str(tts_ms),
        "X-Time-Total-Ms": str(total_ms),
    }

    return Response(
        r.content,
        mimetype="application/octet-stream",
        headers=headers
    )

# Optional debug endpoint for quick testing in browser/postman
@app.route("/ptt-text", methods=["POST"])
def ptt_text():
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY is missing"}), 500

    payload = request.get_json(silent=True) or {}
    transcript = (payload.get("text") or "").strip()
    if not transcript:
        return jsonify({"error": "No text provided"}), 400

    request_id = str(uuid.uuid4())
    use_web_search = ENABLE_WEB_SEARCH and looks_like_live_info_question(transcript)

    body = {
        "model": TEXT_MODEL,
        "max_output_tokens": 120,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": transcript}]
            }
        ]
    }

    if use_web_search:
        body["tools"] = [{"type": "web_search_preview"}]

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            **make_request_headers(request_id),
            "Content-Type": "application/json"
        },
        json=body,
        timeout=120
    )

    if r.status_code != 200:
        return jsonify({
            "stage": "response",
            "error": r.text,
            "request_id": request_id
        }), 500

    reply_text = extract_output_text(r.json()) or "Sorry, I do not have an answer right now."

    return jsonify({
        "request_id": request_id,
        "transcript": transcript,
        "reply": reply_text,
        "web_search_used": use_web_search,
        "text_model": TEXT_MODEL
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
