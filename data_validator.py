from datetime import datetime
from typing import Any

from pydantic import BaseModel

# ... (Previous schemas)

class FeatureSchema(BaseModel):
    ticker: str
    timestamp: str  # Kept as str to match existing structure
    fecha: str
    features: dict[str, float]
    metadata: dict[str, Any] | None = None

class TecnicoTicker(BaseModel):
    precio: float
    ma50: float
    ma200: float
    pct_sobre_ma50: float
    pct_sobre_ma200: float
    tendencia: str
    rsi: float
    senial_rsi: str
    macd: float
    senial_macd: str
    atr_pct: float
    vol_ratio: float
    soporte_20d: float
    resistencia_20d: float
    dist_soporte_pct: float
    dist_resistencia_pct: float

class ResumenTecnico(BaseModel):
    uptrend_count: int
    downtrend_count: int
    sobrecompra_count: int
    sobreventa_count: int
    macd_alcista_count: int
    macd_bajista_count: int

class AnalisisTecnicoSchema(BaseModel):
    timestamp: datetime
    regimen_mercado: str
    tecnicos: dict[str, TecnicoTicker]
    resumen: ResumenTecnico
