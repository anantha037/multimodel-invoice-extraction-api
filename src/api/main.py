from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import logging
from src.api.schemas import InvoiceResponse
from src.core.vision_utils import preprocess_image
from src.core.model_engine import InvoiceExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Invoice Extraction API",
    description="Extracts structured JSON data from messy receipts using CV and VLMs.",
    version="1.0.0"
)

# Initialize VLM globally so it loads once on startup
# Wrapped in try/except so the app can still boot if hardware/drivers aren't fully configured yet
try:
    logger.info("Initializing VLM Engine...")
    extractor = InvoiceExtractor()
except Exception as e:
    logger.error(f"Failed to initialize VLM engine: {e}")
    extractor = None

@app.post("/api/v1/extract", response_model=InvoiceResponse)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded receipt image and return structured JSON.
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="VLM Engine is currently unavailable.")

    # 1. Read file bytes
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload file: {e}")
        raise HTTPException(status_code=400, detail="Invalid file upload.")

    # 2. Computer Vision Preprocessing
    logger.info("Running CV preprocessing pipeline...")
    cleaned_image_np = preprocess_image(contents)
    
    if cleaned_image_np is None:
        raise HTTPException(status_code=422, detail="Failed to process image. Image may be unreadable or corrupt.")

    # 3. VLM Inference
    logger.info("Running VLM inference...")
    try:
        extracted_data = extractor.extract_data(cleaned_image_np)
    except ValueError as ve:
        logger.error(f"JSON Parsing Error: {ve}")
        raise HTTPException(status_code=500, detail="Model failed to output valid structured data.")
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during VLM processing.")

    # 4. Strict Validation with Pydantic
    try:
        validated_response = InvoiceResponse(**extracted_data)
        return validated_response
    except Exception as e:
        logger.error(f"Pydantic Validation Error: {e}")
        raise HTTPException(status_code=500, detail="Extracted data failed schema validation.")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
