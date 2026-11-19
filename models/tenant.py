from datetime import datetime
from enum import IntEnum

from pydantic import BaseModel


class PlanTier(IntEnum):
    FREE = 0
    STARTER = 1
    GROWTH = 2
    ENTERPRISE = 3


class Industry(IntEnum):
    RETAIL = 0
    LOGISTICS = 1
    FOOD_SERVICE = 2
    FINANCE = 3
    HEALTHCARE = 4
    TRANSPORT = 5
    REAL_ESTATE = 6
    OTHER = 7


class Tenant(BaseModel):
    tenant_id: str  # e.g. "t_0001"; matches Device.tenant_id and ping.tenant_id
    name: str
    plan_tier: PlanTier
    industry: Industry
    target_devices: int
    is_whale: bool
    home_region: str
    timezone: str
    created_at: datetime
