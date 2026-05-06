# Multimodal Invoice Extraction API

An end-to-end Machine Learning microservice that takes raw, angled smartphone photos of receipts/invoices and extracts structured JSON data using a Vision-Language Model (VLM).

## 🚀 Features
- **Computer Vision Preprocessing:** Deskews, flattens, and binarizes receipts using OpenCV (CLAHE, Edge Detection, Perspective Transforms).
- **Vision-Language Model:** Utilizes `Qwen2-VL-2B-Instruct` natively for high-accuracy, zero-shot structured extraction.
- **Strict Validation:** Enforces data schemas using Pydantic V2 and FastAPI.
- **Production Ready:** Fully containerized using `uv` for blazing-fast builds.

## 📁 Project Structure
```text
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI Gateway
│   │   ├── schemas.py       # Pydantic JSON Models
│   ├── core/
│   │   ├── vision_utils.py  # OpenCV pipeline
│   │   ├── model_engine.py  # HuggingFace VLM logic
├── tests/
│   ├── data/                # Sample images
│   ├── test_api.py          # Test script
├── Dockerfile               # Container definition
├── requirements.txt         # Dependencies
└── README.md
```

## 🛠️ Setup & Execution

### Option 1: Docker (Recommended)
1. Build the image:
   ```bash
   docker build -t invoice-api .
   ```
2. Run the container:
   ```bash
   # Add --gpus all if you have an NVIDIA GPU
   docker run -p 8000:8000 invoice-api
   ```

### Option 2: Local Python Environment
1. Install dependencies:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```
2. Start the API:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

## 🧪 Testing the API
A test script and a sample image are provided. Ensure the API is running on port 8000.
```bash
python tests/test_api.py --image tests/data/sample_receipt.jpg
```
