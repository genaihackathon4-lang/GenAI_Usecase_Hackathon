import os
import tempfile
import re
import json
import logging
import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage, texttospeech
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from Backend.tools.processing_tool import process_document
import google.adk as adk
import socketio
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware  

# ===== Logging Setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_logger")

# ===== GCS Config =====
BUCKET_NAME = "ai-analyst-uploads-files1"
storage_client = storage.Client()

# ===== Request Schema =====
class DocRequest(BaseModel):
    bucket_name: str
    file_paths: list[str]

# ===== JSON Utilities =====
def fill_json(data, key_path, value):
    keys = key_path.split(".")
    d = data
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

# ===== TTS Helper (in-memory) =====
def synthesize_speech_base64(text: str):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    audio_bytes = response.audio_content
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_b64

# ===== Agents =====

instruction = """
 You are a Data Ingestion and Structuring Agent for startup evaluation.
 
 Tasks:
 1. You MUST call the `process_document` tool with the input {"bucket_name": "...", "file_paths": ["..."]}.
 2. Then analyze text and Output must be *only* valid JSON without Markdown or extra text with this schema:
 
 {
  "startup_name": "string or null",
   "traction": {
    "current_mrr": number or null,
     "mrr_growth_trend": "string or null",
    "active_customers": number or null,
     "other_metrics": ["string", "string"]
   },
   "financials": {
     "ask_amount": number or null,
     "equity_offered": number or null,
     "implied_valuation": number or null,
     "revenue": number or null,
     "burn_rate": number or null
   },
   "team": {
     "ceo": "string or null",
     "cto": "string or null",
     "other_key_members": ["string", "string"]
   },
    "market": {
     "market_size_claim": "string or null",
     "target_market": "string or null"
   },
   "product_description": "string or null",
   "document_type": "pitch_deck | transcript | financial_statement | other"
 }
 
  Rules:
  - No hallucinations.
  - Numbers extracted exactly.
  - Missing = null.
  - Final output must be valid JSON only.
 """
# Root Agent: Document ingestion
root_agent = Agent(
    name="doc_ingest_agent",
    model="gemini-2.0-flash",
    instruction=instruction,
    tools=[process_document]
)

# Question Agent: Generate questions for null fields
question_agent = Agent(
    name="question_agent",
    model="gemini-2.0-flash",
    instruction="""
You are a Question Generation Agent.

Input: JSON object called `structured_json`.
Task:
1. Identify all null fields.
2. Generate human-friendly questions to fill them.
3. Return strictly JSON:

{
  "structured_json": { ... },
  "questions": { "missing_field_key": "natural question", ... },
  "status": "INTERMEDIATE"
}

Rules:
- Only include null fields.
- Do not produce extra commentary or markdown.
"""
)

from pydantic import PrivateAttr

# class FillerAgent(Agent):
#     # structured_json: dict = PrivateAttr(default_factory=dict)
#     # questions: dict = PrivateAttr(default_factory=dict)

#     async def run(self, input_content, **kwargs):
#         import json, re
#         from google.genai import types

#         raw_text = input_content.parts[0].text.strip()
#         cleaned_text = re.sub(r"^```json\s*|```$", "", raw_text, flags=re.MULTILINE)

#         try:
#             content_dict = json.loads(cleaned_text)
#         except json.JSONDecodeError as e:
#             return types.Content(role="system", parts=[types.Part(text=json.dumps({"error": "Invalid JSON input"}))])

#         # Store structured JSON and questions
#         structured_json = content_dict.get("structured_json", {})
#         questions = content_dict.get("questions", {})

#         return types.Content(role="system", parts=[types.Part(text=json.dumps({"status": "READY"}))])

from typing import Dict, Any

class FillerAgent(Agent):
    structured_json: Dict[str, Any] = {}
    questions: Dict[str, str] = {}

    async def run(self, input_content, **kwargs):
        import json, re
        from google.genai import types

        raw_text = input_content.parts[0].text.strip()
        cleaned_text = re.sub(r"^```json\s*|```$", "", raw_text, flags=re.MULTILINE)

        try:
            content_dict = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            return types.Content(
                role="system",
                parts=[types.Part(text=json.dumps({"error": "Invalid JSON input"}))]
            )

        # Initialize structured_json and questions
        self.structured_json = content_dict.get("structured_json", {})
        self.questions = content_dict.get("questions", {})

        return types.Content(role="system", parts=[types.Part(text=json.dumps({"status": "READY"}))])

filler_agent = FillerAgent(
    name="filler_agent",
    model="gemini-2.0-flash",
    instruction="Ask questions to fill missing fields in JSON and return updated JSON."
)

# ===== Session =====
session_service = InMemorySessionService()
app_name = "doc_app"
user_id = "user123"
session_id = "session1"

# ===== FastAPI + Socket.IO =====
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI(title="Doc Voice Chatbot API")
app = socketio.ASGIApp(sio, app)

# @app.on_event("startup")
# async def startup_event():
#     await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

from fastapi import FastAPI
import socketio

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
sio_app = socketio.ASGIApp(sio)

app = FastAPI(title="Doc Voice Chatbot API")

# ===== Enable CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 👈 for testing; replace with your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the Socket.IO ASGI app
app.mount("/ws", sio_app)

@app.on_event("startup")
async def startup_event():
    print("FastAPI app started")
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

# ===== Upload Endpoint =====
@app.post("/upload-and-analyze")
async def upload_and_analyze(files: list[UploadFile], user_email: str = Form(...)):
    file_paths = []
    bucket = storage_client.bucket(BUCKET_NAME)
    for file in files:
        blob_name = f"{user_email}/{file.filename}"
        blob = bucket.blob(blob_name)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        blob.upload_from_filename(tmp_path)
        os.remove(tmp_path)
        file_paths.append(blob_name)

    req = DocRequest(bucket_name=BUCKET_NAME, file_paths=file_paths)
    content = types.Content(role="user", parts=[types.Part(text=req.json())])

    # Run root_agent
    runner_root = adk.Runner(agent=root_agent, app_name=app_name, session_service=session_service)
    root_output = None
    async for event in runner_root.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            root_output = event.content

    # Run question_agent
    runner_q = adk.Runner(agent=question_agent, app_name=app_name, session_service=session_service)
    question_output = None
    async for event in runner_q.run_async(user_id=user_id, session_id=session_id, new_message=root_output):
        if event.is_final_response():
            question_output = event.content

    # Prepare questions in filler agent
    await filler_agent.run(question_output)

    # Emit first question
    await emit_next_question(user_email)

    return JSONResponse({"status": "ok", "message": "Files uploaded and analysis started."})

# ===== Emit Next Question Helper (text + voice base64) =====
async def emit_next_question(user_email):
    if filler_agent.questions:
        key, question = next(iter(filler_agent.questions.items()))
        audio_b64 = synthesize_speech_base64(question)
        await sio.emit("new_question", {"key": key, "text": question, "audio_b64": audio_b64}, room=user_email)
    else:
        await sio.emit("final_json", filler_agent.structured_json, room=user_email)

# ===== Socket.IO Events =====
@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

@sio.on("answer")
async def receive_answer(sid, data):
    answer = data.get("answer")
    user_email = data.get("user_email")
    key = data.get("key")
    if not answer or not user_email or not key:
        return

    fill_json(filler_agent.structured_json, key, answer)
    filler_agent.questions.pop(key, None)
    await emit_next_question(user_email)
