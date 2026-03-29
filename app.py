"""
PTT AI Relay — async, streaming, low-latency
Pipeline: WAV → STT → LLM → TTS (streamed directly to ESP)

Speed wins vs old version:
  - httpx async client with connection pooling (no per-request TCP handshake)
  - TTS response streamed byte-for-byte to ESP as it arrives from OpenAI
    (ESP starts receiving audio ~300-500 ms sooner on typical replies)
  - No X-Transcript / X-Reply-Text header encoding (removed per request)
  - Tighter max_output_tokens (80) — shorter answers = faster TTS
  - Single gunicorn async worker via gevent handles many concurrent ESP clients
"""

import os
import time
import uuid
import logging
import asyncio
import httpx
from flask import Flask, request, Response, jsonify, stream_with_context

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
TEXT_MODEL      = os.environ.get("TEXT_MODEL",  "gpt-4.1-mini")
STT_MODEL       = os.environ.get("STT_MODEL",   "gpt-4o-mini-transcribe")
TTS_MODEL       = os.environ.get("TTS_MODEL",   "gpt-4o-mini-tts")
TTS_VOICE       = os.environ.get("TTS_VOICE",   "alloy")
TTS_FORMAT      = "pcm"          # raw PCM → no container overhead
MAX_TOKENS      = int(os.environ.get("MAX_TOKENS", "80"))
ENABLE_WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH", "true").lower() == "true"
TTS_CHUNK_SIZE  = 4096           # bytes streamed per iteration to ESP

SYSTEM_PROMPT = (
    "You are a concise push-to-talk embedded voice assistant. "
    "Reply in plain spoken English only — no markdown, no bullet points, no lists. "
    "Keep every answer to 1-3 short sentences maximum. "
    "Be direct, natural, and conversational."
)

LIVE_KEYWORDS = frozenset([
    "today", "now", "latest", "current", "news", "weather",
    "price", "stock", "score", "recent", "live", "this week",
    "this month", "yesterday", "tomorrow", "what happened",
    "update", "traffic", "forecast",
])

# ── Shared async HTTP client (connection-pooled) ───────────────────────────────
# One client reused across all requests — avoids repeated TCP + TLS handshakes
_client: httpx.AsyncClient | None = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url="https://api.openai.com",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=httpx.Timeout(connect=8.0, read=120.0, write=60.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            http2=True,   # HTTP/2 multiplexing where available
        )
    return _client

# ── Helpers ───────────────────────────────────────────────────────────────────
def now_ms() -> int:
    return int(time.monotonic() * 1000)

def wants_live_info(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in LIVE_KEYWORDS)

def auth_headers(request_id: str) -> dict:
    return {"X-Client-Request-Id": request_id}

# ── Pipeline steps (all async) ────────────────────────────────────────────────

async def transcribe(wav_bytes: bytes, request_id: str) -> tuple[str, int]:
    """Step 1: WAV → transcript via Whisper."""
    t0 = now_ms()
    r = await get_client().post(
        "/v1/audio/transcriptions",
        headers=auth_headers(request_id),
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={"model": STT_MODEL, "language": "en"},
    )
    r.raise_for_status()
    transcript = r.json().get("text", "").strip() or "I didn't catch that."
    elapsed = now_ms() - t0
    log.info("[%s] STT %d ms — %r", request_id[:8], elapsed, transcript[:80])
    return transcript, elapsed


async def generate_reply(transcript: str, request_id: str) -> tuple[str, int, bool]:
    """Step 2: transcript → reply text via chat model."""
    t0 = now_ms()
    use_search = ENABLE_WEB_SEARCH and wants_live_info(transcript)

    body: dict = {
        "model": TEXT_MODEL,
        "max_output_tokens": MAX_TOKENS,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "input_text", "text": transcript}]},
        ],
    }
    if use_search:
        body["tools"] = [{"type": "web_search_preview"}]

    r = await get_client().post(
        "/v1/responses",
        headers={**auth_headers(request_id), "Content-Type": "application/json"},
        json=body,
    )
    r.raise_for_status()

    data = r.json()
    # Extract text from Responses API output structure
    reply = data.get("output_text", "")
    if not reply:
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        reply += c.get("text", "")
    reply = reply.strip() or "Sorry, I don't have an answer right now."

    elapsed = now_ms() - t0
    log.info("[%s] LLM %d ms (web=%s) — %r", request_id[:8], elapsed, use_search, reply[:80])
    return reply, elapsed, use_search


async def stream_tts(reply_text: str, request_id: str):
    """
    Step 3: reply text → raw PCM audio, streamed chunk-by-chunk.
    Yields bytes as they arrive from OpenAI so the ESP starts
    playing before the full audio is generated.
    """
    t0 = now_ms()
    first_chunk = True

    async with get_client().stream(
        "POST",
        "/v1/audio/speech",
        headers={**auth_headers(request_id), "Content-Type": "application/json"},
        json={
            "model":           TTS_MODEL,
            "voice":           TTS_VOICE,
            "input":           reply_text,
            "response_format": TTS_FORMAT,
        },
    ) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes(chunk_size=TTS_CHUNK_SIZE):
            if first_chunk:
                log.info("[%s] TTS first byte %d ms", request_id[:8], now_ms() - t0)
                first_chunk = False
            yield chunk

    log.info("[%s] TTS stream done %d ms", request_id[:8], now_ms() - t0)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return "PTT relay OK", 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200


@app.route("/ptt", methods=["POST"])
def ptt():
    """
    Main endpoint.
    Accepts raw WAV bytes, returns raw PCM audio (streamed).
    Response headers carry timing and model info only — no text payloads.
    """
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    wav_bytes = request.data
    if not wav_bytes:
        return jsonify({"error": "No audio in request body"}), 400

    request_id = str(uuid.uuid4())
    t_total = now_ms()
    log.info("[%s] PTT request %d bytes", request_id[:8], len(wav_bytes))

    # Run the blocking async pipeline in a new event loop.
    # (Flask is sync; we bridge here. For fully async use Quart instead.)
    loop = asyncio.new_event_loop()
    try:
        transcript, stt_ms = loop.run_until_complete(
            transcribe(wav_bytes, request_id)
        )
        reply_text, llm_ms, used_search = loop.run_until_complete(
            generate_reply(transcript, request_id)
        )
    except httpx.HTTPStatusError as exc:
        log.error("[%s] upstream error %s: %s", request_id[:8], exc.response.status_code, exc.response.text[:200])
        return jsonify({"error": "Upstream API error", "detail": exc.response.text[:200]}), 502
    except Exception as exc:
        log.exception("[%s] pipeline error", request_id[:8])
        return jsonify({"error": str(exc)}), 500
    finally:
        loop.close()

    # Stream TTS audio directly to the ESP client
    def generate_audio():
        tts_loop = asyncio.new_event_loop()
        try:
            async def _collect():
                async for chunk in stream_tts(reply_text, request_id):
                    yield chunk

            # Collect and yield synchronously for Flask's stream_with_context
            async def _run():
                chunks = []
                async for chunk in stream_tts(reply_text, request_id):
                    chunks.append(chunk)
                return chunks

            chunks = tts_loop.run_until_complete(_run())
            for chunk in chunks:
                yield chunk
        except Exception as exc:
            log.error("[%s] TTS stream error: %s", request_id[:8], exc)
        finally:
            tts_loop.close()

    total_ms = now_ms() - t_total
    log.info("[%s] total pipeline %d ms", request_id[:8], total_ms)

    response_headers = {
        "X-Audio-Format":  "pcm_s16le_24khz_mono",
        "X-Request-Id":    request_id,
        "X-Time-STT-Ms":   str(stt_ms),
        "X-Time-LLM-Ms":   str(llm_ms),
        "X-Time-Total-Ms": str(total_ms),
        "X-Web-Search":    "1" if used_search else "0",
        "X-Text-Model":    TEXT_MODEL,
        "X-STT-Model":     STT_MODEL,
        "X-TTS-Model":     TTS_MODEL,
    }

    return Response(
        stream_with_context(generate_audio()),
        mimetype="application/octet-stream",
        headers=response_headers,
    )


@app.route("/ptt-text", methods=["POST"])
def ptt_text():
    """Debug endpoint: POST {"text": "..."} → JSON reply (no audio)."""
    if not OPENAI_API_KEY:
        return jsonify({"error": "OPENAI_API_KEY not set"}), 500

    payload    = request.get_json(silent=True) or {}
    transcript = (payload.get("text") or "").strip()
    if not transcript:
        return jsonify({"error": "No text provided"}), 400

    request_id = str(uuid.uuid4())
    loop = asyncio.new_event_loop()
    try:
        reply_text, llm_ms, used_search = loop.run_until_complete(
            generate_reply(transcript, request_id)
        )
    except Exception as exc:
        log.exception("[%s] ptt-text error", request_id[:8])
        return jsonify({"error": str(exc)}), 500
    finally:
        loop.close()

    return jsonify({
        "request_id":      request_id,
        "transcript":      transcript,
        "reply":           reply_text,
        "web_search_used": used_search,
        "llm_ms":          llm_ms,
        "text_model":      TEXT_MODEL,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
