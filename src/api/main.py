from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import uvicorn
import time
import uuid
from src.api.schemas import InvoiceResponse, CorrectionRequest
from src.core.vision_utils import preprocess_image
from src.core.model_engine import ExtractionPipeline
from src.core.logger import get_logger
from src.core.db import init_db, save_extraction, save_correction

logger = get_logger(__name__)

app = FastAPI(
    title="Enterprise Invoice Extraction API",
    description="Multimodal pipeline with observability, confidence scoring, and HITL feedback loops.",
    version="2.0.0"
)

pipeline = None

@app.on_event("startup")
def startup_event():
    init_db()
    global pipeline
    try:
        logger.info("Initializing Extraction Pipeline...")
        pipeline = ExtractionPipeline()
    except Exception as e:
        logger.error("Failed to initialize pipeline", exception=str(e))

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    process_time_ms = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
    response.headers["X-Request-ID"] = request_id
    
    # Structured JSON Logging
    logger.info("Request Finalized", 
                request_id=request_id, 
                latency_ms=process_time_ms, 
                path=request.url.path, 
                status=response.status_code)
    return response

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024

@app.post("/api/v1/extract", response_model=InvoiceResponse)
async def extract_invoice(request: Request, file: UploadFile = File(...)):
    req_id = request.state.request_id
    if pipeline is None: raise HTTPException(status_code=503, detail="Engine unavailable.")

    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning("Invalid mime type", request_id=req_id, reason=file.content_type)
        raise HTTPException(status_code=400, detail="Invalid file type.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        logger.warning("File too large", request_id=req_id)
        raise HTTPException(status_code=413, detail="File too large.")

    # 1. Preprocessing
    t0 = time.time()
    cleaned_image_np = await run_in_threadpool(preprocess_image, contents)
    logger.info("CV Preprocessing complete", request_id=req_id, stage="CV", latency_ms=round((time.time()-t0)*1000,2))
    
    if cleaned_image_np is None: raise HTTPException(status_code=422, detail="Unreadable image.")

    # 2. Pipeline Inference
    t1 = time.time()
    try:
        extracted_data = await run_in_threadpool(pipeline.execute, cleaned_image_np)
        logger.info("Inference complete", request_id=req_id, stage="Inference", 
                    strategy=extracted_data.get("ExtractionStrategy"), 
                    latency_ms=round((time.time()-t1)*1000,2))
    except Exception as e:
        logger.error("Inference Error", request_id=req_id, exception=str(e))
        raise HTTPException(status_code=500, detail="Internal processing error.")

    # 3. Validation & Persistence
    try:
        validated_response = InvoiceResponse(**extracted_data)
        save_extraction(req_id, file.filename, extracted_data, validated_response.OverallConfidence, "SUCCESS")
        return validated_response
    except Exception as e:
        logger.error("Validation Error", request_id=req_id, exception=str(e))
        save_extraction(req_id, file.filename, extracted_data, extracted_data.get("OverallConfidence", 0.0), "VALIDATION_FAILED")
        raise HTTPException(status_code=500, detail="Data failed validation layer.")

@app.post("/api/v1/correction")
async def submit_correction(correction: CorrectionRequest):
    """Human-in-the-loop endpoint to submit corrected JSON for failed predictions."""
    try:
        save_correction(correction.request_id, correction.corrected_data.dict())
        return {"status": "success", "message": "Correction saved for future fine-tuning."}
    except Exception as e:
        logger.error("Correction Save Failed", request_id=correction.request_id, exception=str(e))
        raise HTTPException(status_code=500, detail="Failed to persist correction.")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
