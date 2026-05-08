from difflib import SequenceMatcher
import requests
import json
import argparse
import os

def similar(a, b):
    """Calculates Levenshtein-like string similarity (0.0 to 1.0)."""
    return SequenceMatcher(None, str(a).strip().lower(), str(b).strip().lower()).ratio()

def evaluate_extraction(image_path: str, ground_truth_path: str):
    url = "http://localhost:8000/api/v1/extract"
    
    if not os.path.exists(image_path) or not os.path.exists(ground_truth_path):
        print("Error: Files not found.")
        return

    print(f"\nRunning rigorous extraction evaluation...")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"Ground Truth: {os.path.basename(ground_truth_path)}\n")
    
    try:
        with open(image_path, "rb") as f:
            response = requests.post(url, files={"file": (os.path.basename(image_path), f, "image/jpeg")})
            
        if response.status_code != 200:
            print(f"API Error: {response.status_code} - {response.text}")
            return
            
        pred = response.json()
        with open(ground_truth_path, 'r') as f:
            true = json.load(f)
            
        print("--- Evaluation Metrics ---")
        
        # Metadata
        print(f"Engine Used: {pred.get('ExtractionStrategy')}")
        print(f"Overall Document Confidence: {pred.get('OverallConfidence', 0.0)}")
        print(f"Processing Request ID: {response.headers.get('X-Request-ID', 'N/A')}")
        print(f"Latency: {response.headers.get('X-Process-Time-Ms', 'N/A')} ms\n")
        
        fields = ["VendorName", "GSTIN", "TotalAmount", "Date"]
        total_score = 0
        
        for f in fields:
            p_val = pred.get(f)
            t_val = true.get(f)
            
            score = similar(p_val, t_val) if p_val and t_val else (1.0 if p_val == t_val else 0.0)
            total_score += score
            
            conf_str = f" [Conf: {pred.get('FieldConfidences', {}).get(f, 'N/A')}]"
            if score == 1.0:
                print(f"{f}: Exact Match ({p_val}){conf_str}")
            elif score > 0.75:
                print(f"{f}: Partial Match ({score:.2f}) (Pred: {p_val} | True: {t_val}){conf_str}")
            else:
                print(f"{f}: Failure (Pred: {p_val} | True: {t_val}){conf_str}")
                
        # Line Items Evaluation
        pred_items = len(pred.get("LineItems", []))
        true_items = len(true.get("LineItems", []))
        if pred_items == true_items:
            print(f"LineItems: Count Match ({pred_items} items)")
            total_score += 1
        else:
            print(f"LineItems: Count Mismatch (Pred: {pred_items} | True: {true_items})")
            
        accuracy = (total_score / (len(fields) + 1)) * 100
        print(f"\nOverall Pipeline Accuracy Score: {accuracy:.1f}%")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to receipt image")
    parser.add_argument("--true", type=str, required=True, help="Path to ground truth JSON")
    args = parser.parse_args()
    evaluate_extraction(args.image, args.true)
