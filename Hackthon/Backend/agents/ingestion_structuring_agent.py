import os
import re
import tempfile
import asyncio
# import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage
from google.adk.agents import Agent, SequentialAgent
import google.adk as adk
from google.adk.sessions import InMemorySessionService
from google.genai import types
# from tools.processing_tool import process_document
from Backend.tools.processing_tool import process_document
# from Backend.tools.email_extraction_tool import check_email_inbox
from fastapi.middleware.cors import CORSMiddleware  
import json
import logging

# ===== Logging Setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline_logger")
# logger=logging.getLogger("google.adk").setLevel(logging.DEBUG)


# ===== GCS Config =====

import tempfile

# ===== GCS Config =====
credentials_env = os.environ.get("GOOGLE_CREDENTIALS_JSON")
if credentials_env:
    # Write the JSON to a temp file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(credentials_env)
        temp_cred_file = f.name

    # Point GOOGLE_APPLICATION_CREDENTIALS to the temp file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_cred_file

# Initialize GCS client (it will use GOOGLE_APPLICATION_CREDENTIALS)
storage_client = storage.Client()

BUCKET_NAME = "ai-analyst-uploads-files1"
# storage_client = storage.Client()

# ===== Request Schema =====
class DocRequest(BaseModel):
    bucket_name: str
    file_paths: list[str]

# ===== System Instruction =====
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


# ===== Define the Agent =====
doc_ingest_agent  = Agent(
    name="doc_ingest_agent",
    model="gemini-2.0-flash",
    instruction=instruction,
    tools=[process_document],
)



# recommendation_instruction = """
# You are the Recommendation & Scoring Agent.

# Role:
# - The final judge. You take the structured JSON data from the Ingestion Agent.
# - Apply scoring logic and generate a deal memo for investors.

# Steps:
# 1. Parse the structured JSON input.
# 2. Score the startup on:
#    - Traction (/10)
#    - Team (/10)
#    - Market (/10)
#    - Product (/10)
# 3. Apply weighted scoring (weights will be provided in input, otherwise default = Team: 0.3, Market: 0.2, Traction: 0.35, Product: 0.15).
# 4. Output a final recommendation:
#    - Verdict: Strong Pass | Pass | Weak Pass | Fail
#    - Rationale: clear strengths and weaknesses
#    - Recommendation: next steps

# Output Format Example:

# It tasks the Recommendation Agent with scoring.

# Recommendation Agent scores:
# Traction: 8/10 (strong growth, high valuation), 
# Team: 9/10, 
# Market: 6/10 (TAM inflated), 
# Product: 7/10.

# Applying the custom weights: (8*0.35) + (9*0.3) + (6*0.2) + (7*0.15) = 7.85/10

# Final Generative Output (The AI Analyst's Memo):
# **Startup X: Deal Memo**
# **Verdict: Weak Pass (Score: 7.85/10)**
# **Strengths:** Exceptional founding team with relevant pedigree and exit. Demonstrated strong initial MRR growth (50% MoM).
# **Risks & Weaknesses:** Market size is significantly inflated; realistic SAM is $5B. Pre-money valuation ask is 2x sector average for this stage. Minor data inconsistency between deck and call on MRR.
# **Recommendation:** Schedule a follow-up call to clarify market sizing assumptions and negotiate valuation down to $6-7M pre-money. Due diligence should focus on customer churn metrics.
# """

recommendation_instruction = """
You are the Recommendation & Scoring Agent.

Role:
- The final judge. You take the structured JSON data from the Ingestion Agent.
- Apply scoring logic and generate a deal memo for investors.

Steps:
1. Parse the structured JSON input.
2. Score the startup on:
   - Traction (/10)
   - Team (/10)
   - Market (/10)
   - Product (/10)
3. Apply weighted scoring (weights will be provided in input, otherwise default = Team: 0.3, Market: 0.2, Traction: 0.35, Product: 0.15).
4. Output a final recommendation:
   - Verdict: Strong Pass | Pass | Weak Pass | Fail
   - Rationale: clear strengths and weaknesses
   - Recommendation: next steps

Output Format Example:

{
  "response": {
    "Traction": "8/10 (strong growth, high valuation)",
    "Team": "9/10 (experienced founders with exits)",
    "Market": "6/10 (TAM inflated)",
    "Product": "7/10 (clear value proposition)",
    "Weighted_Score": "7.85/10",
    "Verdict": "Weak Pass",
    "Strengths": "Exceptional founding team with relevant pedigree and exit. Strong MRR growth.",
    "Risks": "Market size inflated; valuation ask is above average.",
    "Recommendation": "Schedule follow-up call to clarify assumptions and negotiate valuation."
  }
}
"""


recommendation_agent = Agent(
    name="recommendation_agent",
    model="gemini-2.0-flash",
    instruction=recommendation_instruction
)

# ===== Sequential Pipeline =====
pipeline = SequentialAgent(
    name="analysis_pipeline",
    description=(
        "This pipeline runs in two steps:\n"
        "1. The first agent (doc_ingest_agent) ONLY extracts and structures startup data into valid JSON.\n"
        "   It must NOT provide analysis, scoring, or recommendation.\n"
        "2. The second agent (recommendation_agent) ALWAYS takes that JSON as input and generates "
        "   scoring, a final recommendation, and a short memo.\n"
        "The pipeline is complete only after the second agent produces its output."
    ),
    sub_agents=[doc_ingest_agent, recommendation_agent],
)

# ===== Runner =====
session_service = InMemorySessionService()
runner = adk.Runner(agent=pipeline, app_name="startup_app", session_service=session_service)



# ===== Pipeline Runner Function =====
async def run_pipeline(file_json: dict):

    await session_service.create_session(
        app_name="startup_app",
        user_id="user123",
        session_id="session1"
    )

    content = types.Content(role="user", parts=[types.Part(text=json.dumps(file_json))])
    print(content)
    final_output = None  # 👈 hold last agent output

    async for event in runner.run_async(
        user_id="user123",
        session_id="session1",
        new_message=content
    ):
     
        if not event.content or not event.content.parts:
            continue
        # print(event.content.parts)
        for part in event.content.parts:         
            if part.text:
                raw_text = part.text.strip()
                cleaned_text = re.sub(r"^```json\s*|\s*```$", "", raw_text, flags=re.MULTILINE)
                logger.info(f"[{getattr(event, 'source_agent', 'unknown')}] TEXT: {cleaned_text}")
                final_output = cleaned_text

            elif part.function_call:
                logger.info(
                    f"[{getattr(event, 'source_agent', 'unknown')}] TOOL CALL: "
                    f"{part.function_call.name}({part.function_call.args})"
                )

    # after loop ends, final_output will be from the *last agent*
    if not final_output:
        return {"error": "Pipeline returned no output"}

    try:
        # Try parsing JSON (for rec agent you expect text, so this will fail safely)
        return json.loads(final_output)
    except json.JSONDecodeError:
        return {"report": final_output}

# ===== FastAPI App =====
app = FastAPI(title="Doc Ingestion + Recommendation API")

# ===== Enable CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 👈 for testing; replace with your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/full-analysis")
async def full_analysis(
    files: list[UploadFile],
    founder_email: str = Form(...),
    # founder_email: str | None = Form(None)   # 👈 optional founder email
):
    """
    Uploads files to GCS, constructs request for the Agent, 
    and returns structured JSON. If founder_email is provided,
    it will also trigger email_extraction.
    """
    file_paths = []
    bucket = storage_client.bucket(BUCKET_NAME)

    for file in files:
        blob_name = f"{founder_email}/{file.filename}"
        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        blob.upload_from_filename(tmp_path)
        os.remove(tmp_path)

        file_paths.append(blob_name)

    # Build agent input
    payload = {
        "bucket_name": BUCKET_NAME,
        "file_paths": file_paths,
        # "founder_email": founder_email
    }

    result = await run_pipeline(payload)

    return JSONResponse(content={"response": result})
