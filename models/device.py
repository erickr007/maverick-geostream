from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Platform(str, Enum):
    IOS = "ios"
    ANDROID = "android"


class AccuracyProfile(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Behavior(str, Enum):
    STATIONARY = "stationary"
    WALKING = "walking"
    DRIVING = "driving"
    MIXED = "mixed"

    def get_expected_daily_pings(self) -> int:
        return EXPECTED_DAILY_PINGS[self]

EXPECTED_DAILY_PINGS = {
    Behavior.STATIONARY: 50, Behavior.WALKING: 300,
    Behavior.DRIVING: 1500, Behavior.MIXED: 500,
}

class SpoofPattern(str, Enum):
    NONE = "none"
    TELEPORT = "teleport"
    STATIC_WHILE_MOVING = "static_while_moving"
    FIXED_ACCURACY = "fixed_accuracy"


class Device(BaseModel):
    device_id: str
    tenant_id: str
    platform: Platform
    sdk_version: str
    accuracy_profile: AccuracyProfile
    behavior: Behavior
    expected_daily_pings: int
    home_lat: float
    home_lng: float
    anchors: list[tuple[float, float]]
    is_spoofer: bool
    spoof_pattern: SpoofPattern
    installed_at: datetime
