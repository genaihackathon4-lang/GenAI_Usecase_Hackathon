# import os
# import re
# import tempfile
# import asyncio
# # import base64
# from fastapi import FastAPI, UploadFile, Form
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from google.cloud import storage
# from google.adk.agents import Agent, SequentialAgent
# import google.adk as adk
# from google.adk.sessions import InMemorySessionService
# from google.genai import types
# # from tools.processing_tool import process_document
# from Backend.tools.processing_tool import process_document
# # from Backend.tools.email_extraction_tool import check_email_inbox
# from fastapi.middleware.cors import CORSMiddleware  



# import json
# import logging
# from google.cloud import texttospeech
# from google.cloud import speech

# import wave



# # ===== Logging Setup =====
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("pipeline_logger")
# # logger=logging.getLogger("google.adk").setLevel(logging.DEBUG)



# import tempfile

# # ===== GCS Config =====
# # credentials_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
# # if credentials_env:
# #     # Write the JSON to a temp file
# #     with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
# #         f.write(credentials_env)
# #         temp_cred_file = f.name

# #     # Point GOOGLE_APPLICATION_CREDENTIALS to the temp file
# #     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_file

# # Initialize GCS client (it will use GOOGLE_APPLICATION_CREDENTIALS)

# # ===== GCS Config =====
# cred_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

# if cred_env:
#     try:
#         # Try to parse as JSON → this means we’re running in Cloud Run with inline JSON secret
#         creds_dict = json.loads(cred_env)
#         with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
#             json.dump(creds_dict, f)
#             temp_cred_file = f.name
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_file
#         logger.info("Loaded GCP credentials from inline JSON (Cloud Run mode).")
#     except json.JSONDecodeError:
#         # If not JSON, then assume it's a file path (local dev mode)
#         if os.path.exists(cred_env):
#             logger.info(f"Using GCP credentials from file path: {cred_env}")
#         else:
#             raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS is set but invalid: {cred_env}")
# else:
#     logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Using default service account.")
# BUCKET_NAME = "ai-analyst-uploads-files1"
# storage_client = storage.Client()
# # storage_client = storage.Client()

# # ===== Request Schema =====
# class DocRequest(BaseModel):
#     bucket_name: str
#     file_paths: list[str]

# def extract_null_fields(data, path_prefix=""):
#     """
#     Recursively find all keys in a nested dictionary with null (None) values.
    
#     Args:
#         data (dict): The JSON/dictionary to scan.
#         path_prefix (str): Used internally for recursion to track nested paths.
        
#     Returns:
#         list[str]: List of key paths with null values.
#     """
#     null_keys = []
#     for key, value in data.items():
#         full_path = f"{path_prefix}.{key}" if path_prefix else key
#         if isinstance(value, dict):
#             null_keys.extend(extract_null_fields(value, full_path))
#         elif isinstance(value, list):
#             # Skip empty lists or handle differently if needed
#             continue
#         elif value is None:
#             null_keys.append(full_path)
#     return null_keys

# # Utility to fill JSON by key path
# def fill_json(data, key_path, value):
#     keys = key_path.split(".")
#     d = data
#     for k in keys[:-1]:
#         d = d[k]
#     d[keys[-1]] = value

# async def generate_questions(missing_keys: list[str], session_id: str, user_id: str) -> dict:
#     """
#     Uses the question_agent to generate human-readable questions for missing fields.

#     Args:
#         missing_keys (list[str]): List of JSON key paths with null values.
#         session_id (str): ADK session ID.
#         user_id (str): ADK user ID.

#     Returns:
#         dict: Mapping of JSON field keys to questions.
#     """
#     # Prepare JSON with missing keys
#     missing_json = {key: None for key in missing_keys}

#     # Build the content to send to question_agent
#     content = types.Content(
#         role="user",
#         parts=[types.Part(text=json.dumps(missing_json))]
#     )

#     # Create a runner for the question_agent only
#     runner = adk.Runner(agent=question_agent, app_name=app_name, session_service=session_service)

#     async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
#         if event.is_final_response():
#             raw_text = event.content.parts[0].text
#             print("******************",raw_text)
#             # Clean response (remove code blocks if any)
#             cleaned_text = re.sub(r"^```json\n|```$", "", raw_text.strip(), flags=re.MULTILINE)
#             try:
#                 questions = json.loads(cleaned_text)
#             except json.JSONDecodeError:
#                 # Fallback: return keys with simple questions
#                 questions = {key: f"Please provide value for {key}" for key in missing_keys}
#             print(questions)
#             return questions

# # ===== TTS helper =====
# def synthesize_speech(text: str, output_path: str = "output.mp3"):
#     client = texttospeech.TextToSpeechClient()
#     input_text = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
#     response = client.synthesize_speech(
#         input=input_text, voice=voice, audio_config=audio_config
#     )
#     with open(output_path, "wb") as out:
#         out.write(response.audio_content)
#     return output_path

# # ===== STT helper =====
# def recognize_speech(file_path: str):
#     client = speech.SpeechClient()
#     with open(file_path, "rb") as f:
#         audio_bytes = f.read()
#     audio = speech.RecognitionAudio(content=audio_bytes)
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="en-US",
#     )
#     response = client.recognize(config=config, audio=audio)
#     if response.results:
#         return response.results[0].alternatives[0].transcript
#     return ""


# async def ask_questions_and_collect_responses(questions: dict):
#     responses = {}
#     for field, q_text in questions.items():
#         # Generate audio for question
#         audio_file = synthesize_speech(q_text, output_path=f"{field}.mp3")
#         print(f"Audio question saved to: {audio_file}")
#         print(f"Question (text): {q_text}")

#         # --- STT step ---
#         # In real app, the founder records their answer and you provide the path to recognize_speech
#         # Here we simulate by typing
#         # To integrate fully, capture audio from frontend and save as temp WAV/MP3 file
#         # founder_audio_path = "founder_answer.wav"
#         # answer = recognize_speech(founder_audio_path)

#         answer = input(f"Founder response for '{q_text}': ")  # fallback
#         responses[field] = answer

#     return responses


# # ===== System Instruction =====
# instruction = """
# You are a Data Ingestion and Structuring Agent for startup evaluation.
 
# Tasks:
# 1. You MUST call the `process_document` tool with the input {"bucket_name": "...", "file_paths": ["..."]}.
# 2. Then analyze text and Output must be *only* valid JSON without Markdown or extra text with this schema:
 
# {
#   "startup_name": "string or null",
#   "traction": {
#     "current_mrr": number or null,
#     "mrr_growth_trend": "string or null",
#     "active_customers": number or null,
#     "other_metrics": ["string", "string"]
#   },
#   "financials": {
#     "ask_amount": number or null,
#     "equity_offered": number or null,
#     "implied_valuation": number or null,
#     "revenue": number or null,
#     "burn_rate": number or null
#   },
#   "team": {
#     "ceo": "string or null",
#     "cto": "string or null",
#     "other_key_members": ["string", "string"]
#   },
#   "market": {
#     "market_size_claim": "string or null",
#     "target_market": "string or null"
#   },
#   "product_description": "string or null",
#   "document_type": "pitch_deck | transcript | financial_statement | other"
# }
 
# # Rules:
# # - No hallucinations.
# # - Numbers extracted exactly.
# # - Missing = null.
# # - Final output must be valid JSON only.
# # """

#  # ===== Define the Agent =====
# root_agent = Agent(
#     name="doc_ingest_agent",
#     model="gemini-2.0-flash",
#     instruction=instruction,
#     tools=[process_document],
# )


# question_agent = Agent(
#     name="question_generator",
#     model="gemini-2.0-flash",
#     instruction="""
#     You are an intelligent assistant. Given a JSON with missing fields, generate a human-readable question
#     for each missing field to ask the founder. Return ONLY a JSON mapping of field keys to question texts.
#     """
# )

# sequential_pipeline = SequentialAgent(
#     name="doc_then_question_pipeline",
#     description=(
#         "This sequential agent pipeline runs in three steps:\n"
#         "1. The first agent (doc_ingest_agent) extracts and structures startup data from uploaded documents.\n"
#         "   It produces valid JSON with some fields possibly missing (set to null).\n"
#         "2. The second agent (question_generator) generates human-readable questions for each missing field,\n"
#         "   so they can be asked to the founder using TTS/STT or text.\n"
#         "3. The founder’s responses are collected and used to fill the null values in the JSON.\n"
#         "\n"
#         "The final output is a fully structured JSON where all previously null values are replaced with the founder’s answers."
#     ),
#     sub_agents=[root_agent, question_agent]
# )



# # ===== Session Service =====

# session_service = InMemorySessionService()

# app_name = "doc_app"

# user_id = "user123"

# session_id = "session1"
 
# # ===== FastAPI App =====

# app = FastAPI(title="Doc Ingestion Agent API")
 
# @app.on_event("startup")

# async def startup_event():

#     await session_service.create_session(

#         app_name=app_name, user_id=user_id, session_id=session_id

#     )
 
# @app.post("/upload-and-analyze")

# async def upload_and_analyze(files: list[UploadFile], user_email: str = Form(...)):

#     """

#     Uploads files to GCS, constructs request for the Agent, and returns structured JSON.

#     """

#     file_paths = []

#     bucket = storage_client.bucket(BUCKET_NAME)
 
#     for file in files:

#         blob_name = f"{user_email}/{file.filename}"

#         blob = bucket.blob(blob_name)
 
#         with tempfile.NamedTemporaryFile(delete=False) as tmp:

#             tmp.write(await file.read())

#             tmp_path = tmp.name
 
#         blob.upload_from_filename(tmp_path)

#         os.remove(tmp_path)
 
#         file_paths.append(blob_name)
 
#     # Build agent input

#     req = DocRequest(bucket_name=BUCKET_NAME, file_paths=file_paths)
 
#     runner = adk.Runner(agent=sequential_pipeline, app_name=app_name, session_service=session_service)

#     content = types.Content(role="user", parts=[types.Part(text=req.json())])
 
#     # async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
#     #     if event.is_final_response():
#     #         # Clean the agent response (remove markdown/code fences)
#     #         raw_text = event.content.parts[0].text
#     #         cleaned_text = re.sub(r"^```json\n|```$", "", raw_text.strip(), flags=re.MULTILINE)

#     #         try:
#     #             agent_json = json.loads(cleaned_text)
#     #         except json.JSONDecodeError:
#     #             return JSONResponse(status_code=500, content={"error": "Failed to parse agent JSON"})

#     #         # Extract null fields
#     #         missing_keys = extract_null_fields(agent_json)

#     #         # Return both the agent JSON and missing fields
#     #         return JSONResponse(content={
#     #             "response": agent_json,
#     #             "missing_fields": missing_keys
#     #         })
#     final_json = None
#     async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
#         print("DEBUG EVENT:", event.model_dump_json(indent=2))
#         if event.is_final_response() and event.author == "doc_ingest_agent":
#             raw_text = event.content.parts[0].text
#             cleaned_text = re.sub(r"^```json\n|```$", "", raw_text.strip(), flags=re.MULTILINE)
            
#             try:
#                 structured_json = json.loads(cleaned_text)
#                 print("Structured JSON:",structured_json)
#             except json.JSONDecodeError:
#                 return JSONResponse(status_code=500, content={"error": "Failed to parse JSON"})
            
#             # Extract missing fields
#             missing_keys = extract_null_fields(structured_json)
#             print(missing_keys)
            
#             # Generate questions and collect responses via TTS/STT
#             questions = await generate_questions(missing_keys, session_id=session_id, user_id=user_id)
#             print(questions)
#             founder_responses = await ask_questions_and_collect_responses(questions)
#             print("FFFFFFFFFFFFFFFF********",founder_responses)
            
#             # Fill JSON with founder responses
#             for key, answer in founder_responses.items():
#                 fill_json(structured_json, key, answer)
            
#             final_json = structured_json
#             print(final_json)

#     return JSONResponse(content={"filled_json": final_json})
import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage
from google.adk.agents import Agent, SequentialAgent
import google.adk as adk
from google.adk.sessions import InMemorySessionService
from google.genai import types
from Backend.tools.processing_tool import process_document
import json
import logging
from google.cloud import texttospeech
 
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
        logger.info("Loaded GCP credentials from inline JSON (Cloud Run mode).")
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
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    with open(output_path, "wb") as out:
        out.write(response.audio_content)
    return output_path
 
# ===== System Instruction =====
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
 
# ===== Root Agent =====
root_agent = Agent(
    name="doc_ingest_agent",
    model="gemini-2.0-flash",
    instruction=instruction,
    tools=[process_document],
)
 
# ===== Single Question & Null Field Agent using @tool =====
# class QuestionAgent(Agent):
from google.adk.agents import Agent
import google.adk as adk



question_agent = Agent(
    name="question_agent",
    model="gemini-2.0-flash",
    instruction="""
You are a Question Generation Agent.

You receive a JSON object with startup data called `structured_json`.

Your task:
1. Inspect `structured_json`.
2. Identify all fields that are `null`.
3. For each null field, generate a human-friendly question to help fill it.
4. Return output in **this exact JSON format**:

{
  "structured_json": { ... },   // same as input
  "questions": {
    "missing_field_key": "natural question",
    ...
  },
  "status": "INTERMEDIATE"      // 👈 REQUIRED marker to show pipeline must continue
}

⚠️ Rules:
- Do NOT hallucinate fields. Only include keys that are explicitly null.
- Do NOT produce extra commentary, markdown, or text outside the JSON.
- This output is NOT final — another agent will use it to ask the questions and fill the fields.
"""
)



# ===== Filler Agent =====
class FillerAgent(Agent):
    async def run(self, input_content, **kwargs):
        """
        Handles input from the previous agent.
        Accepts input either as function_response or raw text containing JSON.
        """
        import re
        import json
        from google.adk import types

        # Step 1: Extract JSON dict from input_content
        if hasattr(input_content.parts[0], "function_response") and input_content.parts[0].function_response:
            # If previous agent used a tool function
            content_dict = input_content.parts[0].function_response.response
        else:
            # If previous agent returned raw text (possibly with ```json``` markers)
            raw_text = input_content.parts[0].text.strip()

            # Remove ```json ... ``` markers, including optional leading/trailing newlines
            cleaned_text = re.sub(r"^```json\s*\n?|```$", "", raw_text, flags=re.MULTILINE)

            # Load JSON
            try:
                content_dict = json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                print("Failed to parse JSON from input_content:", e)
                return types.Content(
                    role="system",
                    parts=[types.Part(text=json.dumps({"error": "Invalid JSON input from previous agent"}))]
                )

        print("Hiii........FILLER AGENT")
        print("Received content:", content_dict)

        # Step 2: Extract structured JSON and questions
        structured_json = content_dict.get("structured_json", {})
        questions = content_dict.get("questions", {})

        # Step 3: Ask questions and fill answers
        for key, question in questions.items():
            # Optional: Generate TTS for each question
            try:
                synthesize_speech(question, output_path=f"{key}.mp3")
            except Exception as e:
                print(f"Failed to synthesize speech for {key}: {e}")

            # Simulate user answering the question
            answer = input(f"{question}: ")
            fill_json(structured_json, key, answer)

        # Step 4: Return filled JSON as types.Content
        return types.Content(
            role="system",
            parts=[types.Part(text=json.dumps(structured_json, indent=2))]
        )




filler_agent = FillerAgent(
    name="filler_agent",
    model="gemini-2.0-flash",
    instruction="""
You are the Filler Agent.

Your job:
1. Take the JSON object from the previous agent.
2. Check if it contains "structured_json" and "questions".
3. For every entry in "questions", ask the human user for an answer.
4. Insert the answers into the correct keys of "structured_json".
5. Return only the updated JSON (without the 'status' field).

Important rules:
- If no "questions" are present, just return the input "structured_json" unchanged.
- Always return valid JSON. No explanations, no text outside JSON.
- Do not invent new fields on your own. Only fill the ones asked.
"""
)




 
# ===== Sequential Pipeline =====
sequential_pipeline = SequentialAgent(
    name="doc_pipeline",
    description="doc_ingest -> question_agent (extract & questions) -> filler",
    sub_agents=[root_agent, question_agent, filler_agent]
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
 
    final_json = None
    runner = adk.Runner(agent=sequential_pipeline, app_name=app_name, session_service=session_service)
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print("DEBUG EVENT",event)
        if event.is_final_response():
            raw_text = event.content.parts[0].text
            # print("Rawwww text.....",raw_text)
            cleaned_text = re.sub(r"^```json\n|```$", "", raw_text.strip(), flags=re.MULTILINE)
            try:
                final_json = json.loads(cleaned_text)
            except json.JSONDecodeError:
                return JSONResponse(status_code=500, content={"error": "Failed to parse JSON"})
 
    return JSONResponse(content={"filled_json": final_json})