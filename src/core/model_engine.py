import json
import logging
import re
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any

logger = logging.getLogger(__name__)

class InvoiceExtractor:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initializes the VLM model and processor.
        We use bfloat16 and automatic device mapping for optimized inference.
        """
        logger.info(f"Loading VLM model: {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_id)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Attempts to extract and parse JSON from a model's string output.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Fallback: Extract json block using regex if model included markdown
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
                
        # Final fallback: Try to find anything that looks like a JSON object
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError("Could not parse valid JSON from model output.")

    def extract_data(self, image_np: np.ndarray) -> Dict[str, Any]:
        """
        Takes a preprocessed numpy image, runs it through the VLM,
        and strictly parses the output into a Python dictionary.
        """
        if image_np is None:
            raise ValueError("Invalid image provided to VLM.")

        try:
            # Convert binary numpy array (from vision_utils) to RGB PIL Image
            if len(image_np.shape) == 2:
                # If grayscale/binary, convert to 3 channels for the VLM
                image_np = np.stack((image_np,)*3, axis=-1)
            pil_img = Image.fromarray(image_np)
        except Exception as e:
            logger.error(f"Error converting image to PIL: {e}")
            raise ValueError("Failed to convert image for VLM processing.")

        prompt = (
            "Extract the invoice data from this image. Output ONLY valid JSON matching this exact structure, "
            "and absolutely no markdown formatting or extra text:\n"
            '{"VendorName": "", "GSTIN": "", "Date": "", "TotalAmount": 0.0, "LineItems": [{"Description": "", "Price": 0.0}]}'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            # Prepare inputs for Qwen2-VL
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=[pil_img], padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)

            # Generate JSON
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                
            # Trim the prompt from the output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Raw Model Output: {output_text}")
            
            # Parse strictly
            parsed_data = self.parse_json_safely(output_text)
            return parsed_data

        except Exception as e:
            logger.error(f"Error during VLM inference: {e}")
            raise
