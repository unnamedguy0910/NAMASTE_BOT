from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import tempfile
import traceback
from dotenv import load_dotenv
from pathlib import Path
import whisper
from langdetect import detect  # 🔥 for lightweight language detection

# 🔊 ElevenLabs SDK v1 (streaming)
from elevenlabs import ElevenLabs

# ====== PATH SETUP ======
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_FILE = DATA_DIR / "college_index.faiss"
META_FILE = DATA_DIR / "college_meta.pkl"

# ====== LOAD ENV ======
load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("🚨 GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
if not ELEVEN_API_KEY:
    raise ValueError("🚨 ELEVEN_API_KEY not found in .env file.")

ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")  # fallback demo voice

# Initialize ElevenLabs client
el_client = ElevenLabs(api_key=ELEVEN_API_KEY)

# ====== MODELS ======
model = SentenceTransformer("all-MiniLM-L6-v2")
stt_model = whisper.load_model("base")  # load once globally

# ====== FASTAPI APP ======
app = FastAPI()

# ====== CORS ======
origins = ["http://localhost", "http://localhost:5500", "http://127.0.0.1", "http://127.0.0.1:5500", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Schemas ======
class QueryRequest(BaseModel):
    question: str

# ====== Utilities ======
def search_college_data(query: str, top_k: int = 3):
    if not INDEX_FILE.exists() or not META_FILE.exists():
        raise FileNotFoundError(f"Missing FAISS index or metadata in {DATA_DIR}")
    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0] if i != -1]
    return results

def ask_funny_gemini(prompt: str) -> str:
    funny_prompt = f"""
    You are the official AI voice of the Founder Director of the Uttar Pradesh State Institute of Forensic Sciences,
Dr. G.K. Goswami. You must always speak respectfully, use professional yet warm tone,and 
answer as if you are Dr. Goswami himself talking to students, staff or visitors.
if someone is talking in hindi then talk in hindi, if someone is talking in english then talk in english, if someone is talking in marathi then talk in marathi and if someone is talking in bengali then talk in bengali.
and If someone says he is dr gk goswami that means you have to answer very respectfullyand bit funny and show the thanks to him and make him feels great.
You may use Hindi, English, Marathi or Bengali depending on the question.
Never repeat yourself unnecessarily.
    You are the college director. Be respectful, crisp, and slightly witty (English/Hindi/Hinglish).
    Rules:
    - Max 2 sentences, under 25 words.
    - Stay on college context; if off-topic, gently steer back.
    - No vulgarity or insults.

    User asked: {prompt}
    """
    model_gemini = genai.GenerativeModel("gemini-2.5-flash")
    response = model_gemini.generate_content(funny_prompt)
    generation_config={"temperature": 0.8}
    return (response.text or "").strip()

def text_to_speech_11labs(text: str, output_path: str) -> str | None:
    try:
        audio_stream = el_client.text_to_speech.convert(
            voice_id=ELEVEN_VOICE_ID,
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="mp3_44100_128",
        )
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                if isinstance(chunk, (bytes, bytearray)):
                    f.write(chunk)
                else:
                    maybe_bytes = getattr(chunk, "audio", None) or (chunk.get("audio") if isinstance(chunk, dict) else None)
                    if isinstance(maybe_bytes, (bytes, bytearray)):
                        f.write(maybe_bytes)
        return output_path
    except Exception as e:
        print(f"❌ ElevenLabs TTS Error: {e}")
        return None

# ====== NEW: Restricted STT ======
ALLOWED_LANGS = {"en": "en", "hi": "hi", "bn": "bn", "mr": "mr"}

def restricted_transcribe(audio_path: str) -> str:
    """
    Run Whisper but restrict to English, Hindi, Bengali, Marathi
    """
    # First quick transcription without forcing language
    rough = stt_model.transcribe(audio_path)
    rough_text = (rough.get("text") or "").strip()

    try:
        detected = detect(rough_text)
    except Exception:
        detected = "en"

    if detected not in ALLOWED_LANGS:
        detected = "en"

    # Now transcribe again with language lock
    final = stt_model.transcribe(audio_path, language=ALLOWED_LANGS[detected])
    return (final.get("text") or "").strip()

# ====== Routes ======
@app.get("/")
def root():
    return {"ok": True, "service": "Namaste Bot API"}

@app.post("/ask")
async def ask_question(data: QueryRequest):
    try:
        question = data.question.strip()
        results = search_college_data(question)
        if results:
            context = "\n".join(results)
            prompt = f"Context: {context}\nQuestion: {question}"
            answer = ask_funny_gemini(prompt)
        else:
            prompt = f"(No relevant data) Question: {question}"
            answer = ask_funny_gemini(prompt)
        return {"status": "answer", "answer": answer}
    except Exception as e:
        print("🔥 ERROR in /ask:", e)
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}

@app.post("/voice")
async def voice_interaction(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
            tmp_in.write(await file.read())
            input_path = tmp_in.name

        # 🎙️ STT with restriction
        question = restricted_transcribe(input_path)
        print(f"🎤 Transcribed (restricted): {question}")

        # 🔎 RAG + Gemini
        results = search_college_data(question)
        if results:
            context = "\n".join(results)
            prompt = f"Context: {context}\nQuestion: {question}"
            answer = ask_funny_gemini(prompt)
        else:
            prompt = f"(No relevant data) Question: {question}"
            answer = ask_funny_gemini(prompt)

        # 🔊 TTS
        output_audio_path = input_path.replace(".wav", "_reply.mp3")
        saved = text_to_speech_11labs(answer, output_audio_path)

        try:
            os.remove(input_path)
        except Exception:
            pass

        if not saved:
            return {"status": "error", "error": "TTS generation failed"}

        return FileResponse(output_audio_path, media_type="audio/mpeg", filename="response.mp3")

    except Exception as e:
        print("🔥 ERROR in /voice:", e)
        print(traceback.format_exc())
        return {"status": "error", "error": str(e)}
