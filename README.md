# Enterprise Multimodal Invoice Extraction API

An industry-grade Machine Learning microservice designed to ingest raw, messy smartphone photos of receipts and output rigorously validated, structured JSON data. Built for scale, observability, and continuous model improvement.

## 🧠 System Architecture

This pipeline utilizes a multi-strategy extraction approach, ensuring high availability and fault tolerance:

1. **Preprocessing (`src/core/vision_utils.py`)**: Uses OpenCV (CLAHE, Edge Detection, Perspective Transforms) to flatten, deskew, and binarize heavily shadowed images.
2. **Primary Extractor (`src/core/model_engine.py`)**: A state-of-the-art Vision-Language Model (`Qwen2-VL-2B-Instruct`) running in `bfloat16` for fast, zero-shot structured JSON generation.
3. **Fallback Extractor (`src/core/model_engine.py`)**: If the VLM hallucinates or drops below a confidence threshold (0.65), the `ExtractionPipeline` orchestrator dynamically falls back to a deterministic OCR + Regex Engine (Tesseract).
4. **Validation Layer (`src/api/schemas.py`)**: Strict Pydantic V2 validations ensure GSTIN formats are correct and mathematically verify that Line Items sum to the Total Amount.
5. **Human-In-The-Loop Loop (`src/core/db.py`)**: All predictions, confidence scores, and request metadata are persisted to SQLite. A dedicated `/correction` endpoint allows upstream reviewers to correct bad predictions, creating a dataset for future fine-tuning.

## 🚀 Key Features
- **Concurrency**: CPU/GPU heavy ML tasks are offloaded to `run_in_threadpool`, ensuring the async FastAPI event loop never blocks.
- **Observability**: Custom JSON-structured logging tracks Request IDs, stage-level latencies, and fallback triggers natively.
- **Statistical Confidence Scoring**: Hugging Face `generate()` outputs are analyzed to map softmax logprobs into an `OverallConfidence` and `FieldConfidences` mapping for downstream UI flagging.
- **Security**: Strict MIME type (`image/jpeg`, `image/png`) and 10MB payload size enforcements protect the container from OOM attacks.

## 📁 Project Structure
```text
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI Gateway & Middleware
│   │   ├── schemas.py       # Pydantic JSON Models & Business Logic
│   ├── core/
│   │   ├── vision_utils.py  # OpenCV pipeline
│   │   ├── model_engine.py  # Multi-strategy VLM/OCR Extractors
│   │   ├── db.py            # SQLite Persistence Layer
│   │   ├── logger.py        # JSON Structured Observability
├── tests/
│   ├── data/                # Sample images & JSONs
│   ├── test_api.py          # Basic integration test
│   ├── eval.py              # Fuzzy-matching evaluation suite
├── Dockerfile               # uv-optimized container definition
├── requirements.txt         # Dependencies
└── README.md
```

## 🛠️ Setup & Execution

### Docker Deployment (Recommended)
This application is fully containerized using the ultra-fast `uv` package manager and includes all system dependencies for OpenCV and Tesseract.
```bash
docker build -t enterprise-invoice-api .
docker run -p 8000:8000 --gpus all enterprise-invoice-api
```

### Local Development
```bash
pip install uv
uv pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## 🔌 API Usage

**POST `/api/v1/extract`**
Accepts a `multipart/form-data` image upload.

**Response Example:**
```json
{
  "VendorName": "Starbucks",
  "GSTIN": "22AAAAA0000A1Z5",
  "Date": "2023-10-14",
  "TotalAmount": 15.50,
  "LineItems": [
    { "Description": "Latte", "Price": 5.50 },
    { "Description": "Sandwich", "Price": 10.00 }
  ],
  "OverallConfidence": 0.9412,
  "ExtractionStrategy": "VLM",
  "FieldConfidences": {
    "VendorName": 0.8941,
    "TotalAmount": 0.8941
  }
}
```

## 🧪 Evaluation Methodology
The repository includes a strict evaluation script (`tests/eval.py`) that calculates Levenshtein string similarity to evaluate the API against ground-truth JSONs. It provides detailed field-level accuracy and line-item count matching, simulating a true CI/CD model evaluation step.
```bash
python tests/eval.py --image tests/data/sample_receipt.jpg --true tests/data/ground_truth.json
```

## 🔮 Future Improvements
- Migrate SQLite to PostgreSQL for distributed persistence.
- Implement Celery/Redis for asynchronous queuing of massive batch extraction jobs.
- Build a UI for the Human-In-The-Loop `/correction` endpoint.
