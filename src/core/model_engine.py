import json
import re
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any
import pytesseract
from abc import ABC, abstractmethod
from src.core.logger import get_logger

logger = get_logger(__name__)

class BaseExtractor(ABC):
    @abstractmethod
    def extract_data(self, image_np: np.ndarray) -> Dict[str, Any]:
        pass

class VLMExtractor(BaseExtractor):
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        logger.info("Loading VLM model...", extra={"model": model_id})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
        except Exception as e:
            logger.error("Failed to load VLM", extra={"exception": str(e)})
            raise

    def parse_json_safely(self, text: str) -> Dict[str, Any]:
        try: return json.loads(text)
        except json.JSONDecodeError: pass
            
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except json.JSONDecodeError: pass
                
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except json.JSONDecodeError: pass

        raise ValueError("Could not parse valid JSON from VLM output.")

    def extract_data(self, image_np: np.ndarray) -> Dict[str, Any]:
        if image_np is None: raise ValueError("Invalid image provided to VLM.")
        try:
            if len(image_np.shape) == 2: image_np = np.stack((image_np,)*3, axis=-1)
            pil_img = Image.fromarray(image_np)
        except Exception as e:
            raise ValueError(f"Failed to convert image for VLM processing: {e}")

        prompt = (
            "Extract invoice data. Output ONLY valid JSON matching this exact structure:\n"
            '{"VendorName": "", "GSTIN": "", "Date": "", "TotalAmount": 0.0, "LineItems": [{"Description": "", "Price": 0.0}]}'
        )
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

        try:
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[pil_img], padding=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Extract logits to compute statistical confidence scoring
                outputs = self.model.generate(**inputs, max_new_tokens=512, return_dict_in_generate=True, output_scores=True)
                
            generated_ids = outputs.sequences
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
            
            # Confidence Logic: Aggregate logprobs/softmax scores across generated tokens
            if len(outputs.scores) > 0:
                probs = [torch.nn.functional.softmax(score, dim=-1).max().item() for score in outputs.scores]
                overall_conf = round(sum(probs) / len(probs), 4)
            else:
                overall_conf = 0.0

            parsed_data = self.parse_json_safely(output_text)
            parsed_data["OverallConfidence"] = overall_conf
            
            # Placeholder heuristic mapping for field-level confidence based on global score
            parsed_data["FieldConfidences"] = {k: round(overall_conf * 0.95, 4) for k in parsed_data.keys() if k != "LineItems"}
            parsed_data["ExtractionStrategy"] = "VLM"
            
            return parsed_data
        except Exception as e:
            raise RuntimeError(f"VLM inference failed: {e}")

class OCRExtractor(BaseExtractor):
    def extract_data(self, image_np: np.ndarray) -> Dict[str, Any]:
        logger.info("Executing OCR Fallback Strategy")
        try:
            if len(image_np.shape) == 3: image_np = image_np.astype(np.uint8)
            text = pytesseract.image_to_string(image_np)
            
            data = {
                "VendorName": "Fallback Vendor", "GSTIN": None, "Date": None, "TotalAmount": 0.0,
                "LineItems": [], "OverallConfidence": 0.45, "ExtractionStrategy": "OCR Fallback",
                "FieldConfidences": {"TotalAmount": 0.6}
            }
            total_match = re.search(r'(?i)total\s*[:$]?\s*([\d\.,]+)', text)
            if total_match:
                try: data["TotalAmount"] = float(total_match.group(1).replace(',', ''))
                except ValueError: pass
            return data
        except Exception as e:
            raise ValueError(f"OCR Fallback failed: {e}")

class ExtractionPipeline:
    def __init__(self):
        try: self.vlm = VLMExtractor()
        except Exception as e:
            logger.error("VLM failed to load. Pipeline operating in degraded mode.", extra={"exception": str(e)})
            self.vlm = None
        self.ocr = OCRExtractor()
        
    def execute(self, image_np: np.ndarray) -> Dict[str, Any]:
        if self.vlm:
            try:
                data = self.vlm.extract_data(image_np)
                
                # Dynamic Threshold Logic
                if data.get("OverallConfidence", 0.0) < 0.65:
                    raise ValueError(f"VLM Confidence ({data.get('OverallConfidence')}) is below strict threshold (0.65)")
                
                from src.api.schemas import InvoiceResponse
                InvoiceResponse(**data)  # Pre-validation step
                
                return data
            except Exception as e:
                logger.warning("VLM Strategy failed. Triggering OCR Fallback.", extra={"reason": str(e)})
        
        return self.ocr.extract_data(image_np)
