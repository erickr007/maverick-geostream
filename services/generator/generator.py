from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from random import Random, choice, randint, uniform
from typing import Any
from uuid import uuid4

import pandas as pd

from api.dependencies import Settings
from models.device import AccuracyProfile, Behavior, Device, Platform, SpoofPattern
from models.ping import Ping
from models.tenant import Industry, PlanTier, Tenant
from services.utility.ping_time_generator import PingTimeGenerator

PING_SCHEMA = {
    "ping_id": "uuid",
    "device_id": "d_0001234",
    "tenant_id": "tenant_123",
    "ts": "2026-10-14T13:05:22.481Z",
    "lat": 32.7157,
    "lng": -117.1611,
    "accuracy_m": 12.5,
    "speed_mps": 1.4,
    "platform": "ios",
    "sdk_version": "3.9.2",
    "is_spoofed": False,
}

TENANT_DISTRIBUTION = {"small": 0.5, "medium": 0.3, "large": 0.1}


class Generator:
    def __init__(self, settings: Settings | dict | None = None, *, seed: int | None = None, workers: int = 1, out_dir: str | Path = "./output"):
        if settings is None:
            settings = Settings()
        elif isinstance(settings, dict):
            settings = Settings(**settings)

        self.settings = settings
        self.seed = int(seed if seed is not None else getattr(settings, "global_seed", 42))
        self.workers = max(1, int(workers))
        self.out_dir = Path(out_dir)
        self.rng = Random(self.seed)

    def _tenant_plan_counts(self) -> tuple[int, int, int]:
        total = max(1, int(getattr(self.settings, "tenant_count", 1)))
        small = max(1, round(total * TENANT_DISTRIBUTION["small"]))
        medium = max(1, round(total * TENANT_DISTRIBUTION["medium"]))
        large = max(0, total - small - medium)
        return small, medium, large

    def generate_tenants(self) -> list[Tenant]:
        andes = []
        small, medium, large = self._tenant_plan_counts()
        tenant_id = 1

        for _ in range(small):
            andes.append(
                Tenant(
                    tenant_id=f"t_{tenant_id:04d}",
                    name=f"Tenant {tenant_id}",
                    plan_tier=PlanTier.FREE,
                    industry=Industry.RETAIL,
                    target_devices=100,
                    is_whale=False,
                    home_region="US",
                    timezone="America/Los_Angeles",
                    created_at=datetime.now(timezone.utc),
                )
            )
            tenant_id += 1

        for _ in range(medium):
            andes.append(
                Tenant(
                    tenant_id=f"t_{tenant_id:04d}",
                    name=f"Tenant {tenant_id}",
                    plan_tier=PlanTier.STARTER,
                    industry=Industry.LOGISTICS,
                    target_devices=1000,
                    is_whale=False,
                    home_region="US",
                    timezone="America/Los_Angeles",
                    created_at=datetime.now(timezone.utc),
                )
            )
            tenant_id += 1

        for _ in range(large):
            plan_tier = choice([PlanTier.GROWTH, PlanTier.ENTERPRISE])
            tenant = Tenant(
                tenant_id=f"t_{tenant_id:04d}",
                name=f"Tenant {tenant_id}",
                plan_tier=plan_tier,
                industry=choice(list(Industry)),
                target_devices=10000,
                is_whale=True,
                home_region="US",
                timezone="America/Los_Angeles",
                created_at=datetime.now(timezone.utc),
            )
            andes.append(tenant)
            tenant_id += 1

        return andes

    def generate_devices_for_tenant(self, tenant: Tenant) -> list[Device]:
        devices: list[Device] = []
        for i in range(tenant.target_devices):
            behavior = choice(list(Behavior))
            device = Device(
                device_id=f"d_{tenant.tenant_id}_{i:06d}",
                tenant_id=str(tenant.tenant_id),
                platform=choice(list(Platform)),
                sdk_version="3.9.2",
                accuracy_profile=choice(list(AccuracyProfile)),
                behavior=behavior,
                expected_daily_pings=behavior.get_expected_daily_pings(),
                home_lat=uniform(-90, 90),
                home_lng=uniform(-180, 180),
                anchors=[(uniform(-90, 90), uniform(-180, 180)) for _ in range(randint(1, 5))],
                is_spoofer=self.rng.random() < 0.10,
                spoof_pattern=choice(list(SpoofPattern)),
                installed_at=datetime.now(timezone.utc),
            )
            devices.append(device)
        return devices

    def generate_devices(self, tenants: list[Tenant] | None = None) -> list[Device]:
        tenants = tenants or self.generate_tenants()
        devices: list[Device] = []
        for tenant in tenants:
            devices.extend(self.generate_devices_for_tenant(tenant))
        return devices

    def generate_pings(self, device: Device) -> list[dict[str, Any]]:
        time_generator = PingTimeGenerator(self.settings)
        pings: list[dict[str, Any]] = []
        days = int(getattr(self.settings, "simulated_days", 1))
        for day_index in range(days):
            offsets = time_generator.ping_times(device.expected_daily_pings, day_index)
            for offset_ms in offsets:
                ts = datetime.fromtimestamp(offset_ms / 1000, tz=timezone.utc)
                ping = {
                    "ping_id": str(uuid4()),
                    "device_id": device.device_id,
                    "tenant_id": device.tenant_id,
                    "ts": ts.isoformat(),
                    "lat": round(device.home_lat + uniform(-0.05, 0.05), 6),
                    "lng": round(device.home_lng + uniform(-0.05, 0.05), 6),
                    "accuracy_m": round(uniform(10, 25), 2),
                    "speed_mps": round(uniform(0, 5), 2),
                    "platform": device.platform.value,
                    "sdk_version": device.sdk_version,
                    "is_spoofed": device.is_spoofer,
                }
                pings.append(ping)
        return pings

    def _required_shard_count(self, devices: list[Device]) -> int:
        target_pings = max(1, int(getattr(self.settings, "shard_target_pings", 1_000_000)))
        total_expected_pings = sum(
            device.expected_daily_pings * int(getattr(self.settings, "simulated_days", 1))
            for device in devices
        )
        shard_count = math.ceil(total_expected_pings / target_pings)
        return max(1, shard_count)

    def plan(self) -> dict[str, Any]:
        tenants = self.generate_tenants()
        devices = self.generate_devices(tenants)
        shard_count = self._required_shard_count(devices)
        return {
            "tenants": len(tenants),
            "devices": len(devices),
            "shards": shard_count,
            "seed": self.seed,
            "output_dir": str(self.out_dir),
        }

    def generate(self, out_dir: str | Path | None = None, *, workers: int | None = None) -> dict[str, Any]:
        if out_dir is not None:
            self.out_dir = Path(out_dir)
        if workers is not None:
            self.workers = max(1, int(workers))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        tenants = self.generate_tenants()
        devices = self.generate_devices(tenants)

        tenant_df = pd.DataFrame([t.model_dump(mode="json") for t in tenants])
        device_df = pd.DataFrame([d.model_dump(mode="json") for d in devices])
        tenant_df.to_parquet(self.out_dir / "tenants.parquet", index=False)
        device_df.to_parquet(self.out_dir / "devices.parquet", index=False)

        shard_count = self._required_shard_count(devices)
        ping_count = 0
        for shard in range(shard_count):
            pings: list[dict[str, Any]] = []
            for device in devices[shard::shard_count]:
                pings.extend(self.generate_pings(device))
            if pings:
                pd.DataFrame(pings).to_parquet(self.out_dir / f"pings_shard_{shard}.parquet", index=False)
                ping_count += len(pings)

        return {
            "device_count": len(devices),
            "tenant_count": len(tenants),
            "ping_count": ping_count,
            "shards": shard_count,
            "output_dir": str(self.out_dir),
            "files": [p.name for p in sorted(self.out_dir.glob("*.parquet"))],
        }

    def verify(self, out_dir: str | Path | None = None) -> dict[str, Any]:
        if out_dir is not None:
            self.out_dir = Path(out_dir)

        parquet_files = sorted(self.out_dir.glob("*.parquet"))
        if not parquet_files:
            return {"ok": False, "reason": "no parquet files found", "files": [], "total_rows": 0}

        checksums: dict[str, str] = {}
        total_rows = 0
        platform_counts: dict[str, int] = {}
        spoof_counts: dict[str, int] = {}

        for file_path in parquet_files:
            file_bytes = file_path.read_bytes()
            checksums[file_path.name] = hashlib.sha256(file_bytes).hexdigest()
            frame = pd.read_parquet(file_path)
            total_rows += len(frame)
            if "platform" in frame.columns:
                for platform in frame["platform"].astype(str).tolist():
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
            if "is_spoofed" in frame.columns:
                for spoof in frame["is_spoofed"].astype(str).tolist():
                    spoof_counts[spoof] = spoof_counts.get(spoof, 0) + 1

        ok = total_rows > 0 and bool(platform_counts) and bool(spoof_counts)
        return {
            "ok": ok,
            "file_count": len(parquet_files),
            "total_rows": total_rows,
            "checksums": checksums,
            "platform_distribution": platform_counts,
            "spoof_distribution": spoof_counts,
            "output_dir": str(self.out_dir),
        }

    def process_shards(self):
        return self.generate()
