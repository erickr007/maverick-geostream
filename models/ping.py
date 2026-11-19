from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class Ping(BaseModel):
    ping_id: UUID
    device_id: str
    tenant_id: str
    ts: datetime
    lat: float
    lng: float
    accuracy_m: float
    speed_mps: float | None = None
    platform: str
    sdk_version: str
    is_spoofed: bool = False
