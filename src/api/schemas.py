from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict
import re

class LineItem(BaseModel):
    Description: str = Field(description="Description or name of the item purchased")
    Price: float = Field(description="Price of the item")

class InvoiceResponse(BaseModel):
    VendorName: Optional[str] = Field(default=None)
    GSTIN: Optional[str] = Field(default=None)
    Date: Optional[str] = Field(default=None)
    TotalAmount: float = Field(default=0.0)
    LineItems: List[LineItem] = Field(default_factory=list)
    
    # Observability & Confidence fields
    OverallConfidence: float = Field(default=0.0, description="Overall document extraction confidence (0.0 - 1.0)")
    FieldConfidences: Dict[str, float] = Field(default_factory=dict, description="Field-level confidence mapping")
    ExtractionStrategy: str = Field(default="Unknown", description="VLM or OCR")

    @field_validator('GSTIN')
    @classmethod
    def validate_gstin(cls, v):
        if v:
            if len(v) < 5 or not re.search(r'[A-Za-z]', v) or not re.search(r'\d', v):
                raise ValueError("GSTIN format appears invalid (must be alphanumeric and > 5 chars)")
        return v

    @model_validator(mode='after')
    def check_totals(self) -> 'InvoiceResponse':
        if self.LineItems:
            calculated_total = sum(item.Price for item in self.LineItems)
            if abs(calculated_total - self.TotalAmount) > 1.0:
                raise ValueError(f"Line items sum ({calculated_total}) inconsistent with TotalAmount ({self.TotalAmount})")
        return self

class CorrectionRequest(BaseModel):
    request_id: str
    corrected_data: InvoiceResponse
