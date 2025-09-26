import os
import re
import tempfile
import json
import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage, texttospeech
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types
from Backend.tools.processing_tool import process_document
import google.adk as adk
# ===== Logging Setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_logger")

# ===== GCS Config =====
cred_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if cred_env:
    try:
        creds_dict = json.loads(cred_env)
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            json.dump(creds_dict, f)
            temp_cred_file = f.name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_file
        logger.info("Loaded GCP credentials from inline JSON.")
    except json.JSONDecodeError:
        if os.path.exists(cred_env):
            logger.info(f"Using GCP credentials from file path: {cred_env}")
        else:
            raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS is invalid: {cred_env}")
else:
    logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Using default service account.")

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

# ===== TTS Helper =====
def synthesize_speech(text: str, output_path: str = "output.mp3"):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    return output_path

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

# Filler Agent: Ask human for answers
class FillerAgent(Agent):
    async def run(self, input_content, **kwargs):
        import json
        import re
        # from google.adk import types
        from google.genai import types
        # Extract JSON
        raw_text = input_content.parts[0].text.strip()
        cleaned_text = re.sub(r"^```json\s*|```$", "", raw_text, flags=re.MULTILINE)
        try:
            content_dict = json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            return types.Content(
                role="system",
                parts=[types.Part(text=json.dumps({"error": "Invalid JSON input"}))]
            )

        structured_json = content_dict.get("structured_json", {})
        questions = content_dict.get("questions", {})

        # Ask human for answers
        for key, question in questions.items():
            try:
                synthesize_speech(question, output_path=f"{key}.mp3")
            except Exception as e:
                print(f"TTS failed for {key}: {e}")
            answer = input(f"{question}: ")
            fill_json(structured_json, key, answer)

        # Step 3: Put updated structured_json back
        content_dict["structured_json"] = structured_json
        # Remove "status" field if present
        content_dict.pop("questions", None)
        content_dict.pop("status", None)

        return types.Content(
            role="system",
            parts=[types.Part(text=json.dumps(content_dict, indent=2))]
        )

        # return types.Content(
        #     role="system",
        #     parts=[types.Part(text=json.dumps(structured_json, indent=2))]
        # )

filler_agent = FillerAgent(
    name="filler_agent",
    model="gemini-2.0-flash",
    instruction="Ask questions to fill missing fields in JSON and return updated JSON."
)

# ===== Session Service =====
session_service = InMemorySessionService()
app_name = "doc_app"
user_id = "user123"
session_id = "session1"

# ===== FastAPI App =====
app = FastAPI(title="Doc Ingestion Agent API")

@app.on_event("startup")
async def startup_event():
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

@app.post("/upload-and-analyze")
async def upload_and_analyze(files: list[UploadFile], user_email: str = Form(...)):
    # Step 1: Upload files to GCS
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

    # Step 2: Run root_agent
    runner_root = adk.Runner(agent=root_agent, app_name=app_name, session_service=session_service)
    root_output = None
    async for event in runner_root.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if event.is_final_response():
            root_output = event.content
    print("ROOT AGENT",root_output)
    # Step 3: Run question_agent
    runner_q = adk.Runner(agent=question_agent, app_name=app_name, session_service=session_service)
    question_output = None
    async for event in runner_q.run_async(user_id=user_id, session_id=session_id, new_message=root_output):
        if event.is_final_response():
            question_output = event.content
    print("QUESTION AGENT",question_output)
    # Step 4: Run FillerAgent to ask human for answers
    filled_content = await filler_agent.run(question_output)
    final_json = json.loads(filled_content.parts[0].text)

    return JSONResponse(content={"filled_json": final_json})
