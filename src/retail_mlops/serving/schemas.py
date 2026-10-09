from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

ID = r"^[A-Za-z0-9_-]{1,16}$"
MIN_DATE, MAX_DATE = dt.date(2000, 1, 1), dt.date(2100, 1, 1)


class Instance(BaseModel):
    store_id: str = Field(pattern=ID, examples=["S01"])
    sku_id: str = Field(pattern=ID, examples=["K001"])
    date: dt.date = Field(ge=MIN_DATE, le=MAX_DATE, description="Date to forecast")
    price: float | None = Field(None, gt=0, lt=1e6, description="Planned shelf price; defaults to last known price")
    promo: int = Field(0, ge=0, le=1, description="1 if the item is on promotion that day")


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    store_id: str
    sku_id: str
    date: dt.date
    prediction: float = Field(description="Forecast units sold (>= 0)")
    low_history: bool = Field(description="True if <28 days of history: forecast is less reliable")
    horizon_days: int = Field(description="Days ahead of the latest observed sales (0 = date already observed)")
    model_version: str


class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    predictions: list[Prediction]


class GroundTruth(BaseModel):
    store_id: str = Field(pattern=ID)
    sku_id: str = Field(pattern=ID)
    date: dt.date = Field(ge=MIN_DATE, le=MAX_DATE)
    units: float = Field(ge=0, le=1e6)
    category: str | None = Field(None, pattern=r"^[A-Za-z0-9_ -]{1,64}$")
    price: float | None = Field(None, gt=0, lt=1e6)
    promo: int | None = Field(None, ge=0, le=1)


class GroundTruthRequest(BaseModel):
    records: list[GroundTruth] = Field(min_length=1, max_length=50000)
