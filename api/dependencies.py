from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from models.applicationsettings import ApplicationSettings

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    total_pings: int = Field(default=1000)
    tenant_count: int = Field(default=10)
    simulated_days: int = Field(default=30)
    shard_target_pings: int = Field(default=1_000_000)
    start_date: str = Field(default="2024-01-01")
    global_seed: int = Field(default=42)
    chaos_rates: list[float] = Field(default_factory=lambda: [0.0])
    stream_names: list[str] = Field(default_factory=list)
    region: str = Field(default="us-east-1")
    batch_limits: dict[str, int] = Field(default_factory=dict)
    worker_counts: dict[str, int] = Field(default_factory=dict)

    # Plan step (see services/generator/plan.py PLAN_DEFAULTS for meaning and defaults).
    # None means "use the plan default".
    whale_share: float | None = None
    tenant_size_sigma: float | None = None
    behavior_mix: dict[str, float] | None = None
    ios_share: float | None = None
    sdk_versions: dict[str, float] | None = None
    accuracy_mix: dict[str, float] | None = None
    spoofer_rate: float | None = None
    spoof_pattern_mix: dict[str, float] | None = None
    metro_spread_km: float | None = None
    device_home_metro_prob: float | None = None
    max_anchors_per_device: int | None = None
    geofence_anchor_prob: float | None = None
    geofences_base: int | None = None
    geofences_per_sqrt_device: float | None = None
    geofences_max_per_tenant: int | None = None
    geofence_radius_m: list[float] | None = None
    device_block_size: int | None = None

    @field_validator("chaos_rates", "stream_names", "batch_limits", "worker_counts",
                     "behavior_mix", "sdk_versions", "accuracy_mix", "spoof_pattern_mix",
                     "geofence_radius_m", mode="before")
    @classmethod
    def parse_json_like_env(cls, value):
        if value is None:
            return value
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return [] if "chaos_rates" in cls.model_fields and value == "" else {}
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()

ApplicationSettingsDep   = Annotated[Settings,            Depends(get_settings)]