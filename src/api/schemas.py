from pydantic import BaseModel, Field
from typing import List, Optional

class LineItem(BaseModel):
    Description: str = Field(description="Description or name of the item purchased")
    Price: float = Field(description="Price of the item")

class InvoiceResponse(BaseModel):
    VendorName: Optional[str] = Field(default=None, description="Name of the vendor or store")
    GSTIN: Optional[str] = Field(default=None, description="GSTIN or Tax ID number")
    Date: Optional[str] = Field(default=None, description="Date of the invoice/receipt")
    TotalAmount: float = Field(default=0.0, description="Total amount paid")
    LineItems: List[LineItem] = Field(default_factory=list, description="List of individual purchased items")
