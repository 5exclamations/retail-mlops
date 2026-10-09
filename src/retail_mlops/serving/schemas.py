from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class Instance(BaseModel):
    store_id: str = Field(examples=["S01"])
    sku_id: str = Field(examples=["K001"])
    date: dt.date = Field(description="Date to forecast")
    price: float | None = Field(None, gt=0, description="Planned shelf price; defaults to last known price")
    promo: int = Field(0, ge=0, le=1, description="1 if the item is on promotion that day")


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    store_id: str
    sku_id: str
    date: dt.date
    prediction: float = Field(description="Forecast units sold (>= 0)")
    low_history: bool = Field(description="True if <28 days of history: forecast is less reliable")
    model_version: str


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    predictions: list[Prediction]


class GroundTruth(BaseModel):
    store_id: str
    sku_id: str
    date: dt.date
    units: float = Field(ge=0)
    category: str | None = None
    price: float | None = Field(None, gt=0)
    promo: int | None = Field(None, ge=0, le=1)


class GroundTruthRequest(BaseModel):
    records: list[GroundTruth] = Field(min_length=1, max_length=50000)
