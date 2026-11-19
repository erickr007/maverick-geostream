"""Deterministic, vectorized `plan` step.

Builds every table the generator needs before any pings exist:

    <out>/plan/tenants.parquet      one row per tenant (whale + long tail)
    <out>/plan/geofences.parquet    circular geofences around each tenant's metro
    <out>/plan/devices.parquet      one row per device, sorted by device_idx
    <out>/plan/shard_plan.parquet   one row per ShardSpec (volume-balanced, whole blocks)
    <out>/plan/plan_summary.json    config snapshot and expected distributions

Determinism rules followed throughout:
  * every random draw comes from a NumPy Generator seeded by SeedSequence(seed, ...)
  * no `random` module, no uuid4(), no datetime.now()
  * each table draws from its own child stream, so changing one table's logic
    never shifts another table's values
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from models.device import EXPECTED_DAILY_PINGS, Behavior, Device
from models.shardspec import ShardSpec
from models.tenant import Industry, PlanTier, Tenant

DAY_MS = 86_400_000
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LNG_EQUATOR = 111.320

# Spawn keys for the plan's random streams. Generate uses 2-element keys
# (block_id, day), so these 1-element keys can never collide with them.
_STREAM_TENANTS = 0
_STREAM_GEOFENCES = 1
_STREAM_DEVICES = 2

DEFAULT_METROS: list[dict[str, Any]] = [
    {"name": "New York", "lat": 40.7128, "lng": -74.0060, "weight": 0.20, "region": "us-east", "timezone": "America/New_York"},
    {"name": "Los Angeles", "lat": 34.0522, "lng": -118.2437, "weight": 0.16, "region": "us-west", "timezone": "America/Los_Angeles"},
    {"name": "Chicago", "lat": 41.8781, "lng": -87.6298, "weight": 0.12, "region": "us-central", "timezone": "America/Chicago"},
    {"name": "Dallas", "lat": 32.7767, "lng": -96.7970, "weight": 0.10, "region": "us-central", "timezone": "America/Chicago"},
    {"name": "Atlanta", "lat": 33.7490, "lng": -84.3880, "weight": 0.08, "region": "us-east", "timezone": "America/New_York"},
    {"name": "Seattle", "lat": 47.6062, "lng": -122.3321, "weight": 0.08, "region": "us-west", "timezone": "America/Los_Angeles"},
    {"name": "San Diego", "lat": 32.7157, "lng": -117.1611, "weight": 0.07, "region": "us-west", "timezone": "America/Los_Angeles"},
    {"name": "Miami", "lat": 25.7617, "lng": -80.1918, "weight": 0.07, "region": "us-east", "timezone": "America/New_York"},
    {"name": "Denver", "lat": 39.7392, "lng": -104.9903, "weight": 0.06, "region": "us-west", "timezone": "America/Denver"},
    {"name": "Boston", "lat": 42.3601, "lng": -71.0589, "weight": 0.06, "region": "us-east", "timezone": "America/New_York"},
]

GEOFENCE_CATEGORIES = np.array(["store", "parking_lot", "warehouse", "office", "competitor"])
GEOFENCE_CATEGORY_WEIGHTS = np.array([0.45, 0.20, 0.10, 0.10, 0.15])

_NAME_ADJ = np.array(["North", "Blue", "Summit", "Harbor", "Iron", "Golden", "Cedar", "Bright",
                      "Swift", "Prairie", "Coastal", "Silver", "Maple", "Granite", "Lumen", "Atlas"])
_NAME_NOUN = np.array(["Coffee", "Freight", "Market", "Health", "Fitness", "Grocers", "Motors",
                       "Outfitters", "Logistics", "Pharmacy", "Kitchen", "Realty", "Transit", "Bank"])
_NAME_SUFFIX = np.array(["Co", "Inc", "Group", "Labs", "Partners", "Collective", "Works", "Hub"])


# ---------------------------------------------------------------- helpers

def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream,)))


def _offset_km(lat, lng, dx_km, dy_km):
    """Shift points east by dx_km and north by dy_km (equirectangular, city scale)."""
    new_lat = lat + dy_km / KM_PER_DEG_LAT
    new_lng = lng + dx_km / (KM_PER_DEG_LNG_EQUATOR * np.cos(np.radians(lat)))
    return new_lat, new_lng


def _normalized(weights: dict[str, float], allowed: list[str], label: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate a {name: weight} mix and return (names, probabilities)."""
    unknown = set(weights) - set(allowed)
    if unknown:
        raise ValueError(f"{label}: unknown keys {sorted(unknown)}; allowed {allowed}")
    names = np.array(list(weights.keys()))
    probs = np.array([float(weights[k]) for k in names], dtype=np.float64)
    if (probs < 0).any() or probs.sum() <= 0:
        raise ValueError(f"{label}: weights must be non-negative and sum to more than 0")
    return names, probs / probs.sum()


def _allocate(total: int, weights: np.ndarray) -> np.ndarray:
    """Split `total` into integer parts proportional to `weights` (largest remainder)."""
    exact = weights / weights.sum() * total
    base = np.floor(exact).astype(np.int64)
    remainder = total - int(base.sum())
    if remainder:
        # stable tie-break keeps this deterministic
        order = np.argsort(-(exact - base), kind="stable")
        base[order[:remainder]] += 1
    return base


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _write_atomic(table: pa.Table, path: Path, *, row_group_size: int | None = None) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, row_group_size=row_group_size)
    os.replace(tmp, path)


def _start_ms(start_date: str) -> int:
    start = datetime.fromisoformat(start_date)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return _ms(start.astimezone(timezone.utc))


# ---------------------------------------------------------------- tenants

def _build_tenants(rng: np.random.Generator, cfg: dict[str, Any], n_devices: int, metros: list[dict]) -> dict[str, np.ndarray]:
    t = int(cfg["tenant_count"])
    whale_share = float(cfg["whale_share"])
    whale_devices = int(round(n_devices * whale_share))
    rest = n_devices - whale_devices
    if t > 1 and rest < t - 1:
        raise ValueError(
            f"{n_devices} devices can't give {t - 1} non-whale tenants at least one device each; "
            f"raise total_pings or lower tenant_count")

    # Long tail: lognormal weights, sorted so tenant 2 is the second largest.
    tail_weights = np.sort(rng.lognormal(mean=0.0, sigma=float(cfg["tenant_size_sigma"]), size=t - 1))[::-1]
    tail = 1 + _allocate(rest - (t - 1), tail_weights) if t > 1 else np.zeros(0, dtype=np.int64)
    target_devices = np.concatenate(([whale_devices], tail)).astype(np.int64)

    # Plan tier follows size rank, not chance.
    rank_frac = np.arange(t) / max(t, 1)
    plan_tier = np.select(
        [rank_frac < 0.05, rank_frac < 0.25, rank_frac < 0.50],
        [int(PlanTier.ENTERPRISE), int(PlanTier.GROWTH), int(PlanTier.STARTER)],
        default=int(PlanTier.FREE),
    ).astype(np.int64)
    plan_tier[0] = int(PlanTier.ENTERPRISE)

    industry = rng.integers(0, len(Industry), size=t)
    metro_weights = np.array([m["weight"] for m in metros], dtype=np.float64)
    metro_idx = rng.choice(len(metros), size=t, p=metro_weights / metro_weights.sum())

    adj = rng.integers(0, len(_NAME_ADJ), size=t)
    noun = rng.integers(0, len(_NAME_NOUN), size=t)
    suf = rng.integers(0, len(_NAME_SUFFIX), size=t)
    names = np.array([f"{_NAME_ADJ[a]} {_NAME_NOUN[b]} {_NAME_SUFFIX[c]}" for a, b, c in zip(adj, noun, suf)])

    start = _start_ms(cfg["start_date"])
    created_at = start - (rng.uniform(30, 1500, size=t) * DAY_MS).astype(np.int64)

    return {
        "tenant_idx": np.arange(t, dtype=np.int64),
        "tenant_id": np.array([f"t_{i + 1:04d}" for i in range(t)]),
        "name": names,
        "plan_tier": plan_tier,
        "industry": industry.astype(np.int64),
        "target_devices": target_devices,
        "is_whale": np.arange(t) == 0,
        "home_metro": np.array([metros[i]["name"] for i in metro_idx]),
        "home_region": np.array([metros[i]["region"] for i in metro_idx]),
        "timezone": np.array([metros[i]["timezone"] for i in metro_idx]),
        "created_at": created_at,
        "_metro_idx": metro_idx.astype(np.int64),
    }


# ---------------------------------------------------------------- geofences

def _build_geofences(rng: np.random.Generator, cfg: dict[str, Any], tenants: dict[str, np.ndarray],
                     metros: list[dict]) -> dict[str, np.ndarray]:
    devices_per = tenants["target_devices"]
    counts = np.clip(
        np.round(cfg["geofences_base"] + cfg["geofences_per_sqrt_device"] * np.sqrt(devices_per)),
        1, cfg["geofences_max_per_tenant"],
    ).astype(np.int64)
    g = int(counts.sum())

    owner = np.repeat(tenants["tenant_idx"], counts)
    metro_idx = tenants["_metro_idx"][owner]
    center_lat0 = np.array([metros[i]["lat"] for i in metro_idx])
    center_lng0 = np.array([metros[i]["lng"] for i in metro_idx])
    spread = float(cfg["metro_spread_km"])
    lat, lng = _offset_km(center_lat0, center_lng0, rng.normal(0, spread, g), rng.normal(0, spread, g))

    lo, hi = cfg["geofence_radius_m"]
    radius = rng.uniform(lo, hi, size=g)
    category = GEOFENCE_CATEGORIES[rng.choice(len(GEOFENCE_CATEGORIES), size=g,
                                              p=GEOFENCE_CATEGORY_WEIGHTS / GEOFENCE_CATEGORY_WEIGHTS.sum())]
    seq = np.arange(g) - np.repeat(np.cumsum(counts) - counts, counts)  # 0..count-1 within tenant

    return {
        "geofence_id": np.array([f"g_{i:07d}" for i in range(g)]),
        "tenant_id": tenants["tenant_id"][owner],
        "name": np.array([f"{tenants['name'][o]} #{s + 1}" for o, s in zip(owner, seq)]),
        "category": category,
        "shape_type": np.full(g, "circle"),
        "center_lat": lat,
        "center_lng": lng,
        "radius_m": radius,
        "metro": tenants["home_metro"][owner],
        "_owner": owner,
        "_counts": counts,
    }


# ---------------------------------------------------------------- devices

def _build_devices(rng: np.random.Generator, cfg: dict[str, Any], tenants: dict[str, np.ndarray],
                   geofences: dict[str, np.ndarray], metros: list[dict]) -> dict[str, Any]:
    counts = tenants["target_devices"]
    n = int(counts.sum())
    owner = np.repeat(tenants["tenant_idx"], counts)  # devices ordered by tenant, whale first

    # Metro: mostly the tenant's home metro, some anywhere.
    metro_weights = np.array([m["weight"] for m in metros], dtype=np.float64)
    other_metro = rng.choice(len(metros), size=n, p=metro_weights / metro_weights.sum())
    at_home_metro = rng.random(n) < float(cfg["device_home_metro_prob"])
    metro_idx = np.where(at_home_metro, tenants["_metro_idx"][owner], other_metro)
    m_lat = np.array([m["lat"] for m in metros])[metro_idx]
    m_lng = np.array([m["lng"] for m in metros])[metro_idx]
    spread = float(cfg["metro_spread_km"])
    home_lat, home_lng = _offset_km(m_lat, m_lng, rng.normal(0, spread, n), rng.normal(0, spread, n))

    # Categorical attributes
    platform = np.where(rng.random(n) < float(cfg["ios_share"]), "ios", "android")
    sdk_names, sdk_p = _normalized(cfg["sdk_versions"], list(cfg["sdk_versions"]), "sdk_versions")
    sdk_version = sdk_names[rng.choice(len(sdk_names), size=n, p=sdk_p)]
    acc_names, acc_p = _normalized(cfg["accuracy_mix"], ["high", "medium", "low"], "accuracy_mix")
    accuracy_profile = acc_names[rng.choice(len(acc_names), size=n, p=acc_p)]
    beh_names, beh_p = _normalized(cfg["behavior_mix"], [b.value for b in Behavior], "behavior_mix")
    beh_code = rng.choice(len(beh_names), size=n, p=beh_p)
    behavior = beh_names[beh_code]
    rates = np.array([EXPECTED_DAILY_PINGS[Behavior(b)] for b in beh_names], dtype=np.int64)
    expected_daily_pings = rates[beh_code]

    # Spoofers: about spoofer_rate of devices; only spoofers get a pattern.
    is_spoofer = rng.random(n) < float(cfg["spoofer_rate"])
    pat_names, pat_p = _normalized(cfg["spoof_pattern_mix"], ["teleport", "static_while_moving", "fixed_accuracy"],
                                   "spoof_pattern_mix")
    pattern = pat_names[rng.choice(len(pat_names), size=n, p=pat_p)]
    spoof_pattern = np.where(is_spoofer, pattern, "none")

    # Anchors: 1..max per device; some sit inside one of the tenant's geofences.
    k = rng.integers(1, int(cfg["max_anchors_per_device"]) + 1, size=n)
    a_total = int(k.sum())
    a_dev = np.repeat(np.arange(n), k)
    a_lat, a_lng = _offset_km(m_lat[a_dev], m_lng[a_dev],
                              rng.normal(0, spread, a_total), rng.normal(0, spread, a_total))

    gf_counts = geofences["_counts"]
    gf_start = np.cumsum(gf_counts) - gf_counts
    a_owner = owner[a_dev]
    eligible = at_home_metro[a_dev]  # geofences live in the tenant's home metro
    wants_gf = eligible & (rng.random(a_total) < float(cfg["geofence_anchor_prob"]))
    gf_pick = gf_start[a_owner] + np.floor(rng.random(a_total) * gf_counts[a_owner]).astype(np.int64)
    # Land well inside the circle: up to half the radius from the center.
    r_km = geofences["radius_m"][gf_pick] / 1000.0 * 0.5 * np.sqrt(rng.random(a_total))
    theta = rng.uniform(0, 2 * np.pi, a_total)
    g_lat, g_lng = _offset_km(geofences["center_lat"][gf_pick], geofences["center_lng"][gf_pick],
                              r_km * np.cos(theta), r_km * np.sin(theta))
    a_lat = np.where(wants_gf, g_lat, a_lat)
    a_lng = np.where(wants_gf, g_lng, a_lng)
    anchor_offsets = np.concatenate(([0], np.cumsum(k))).astype(np.int32)

    start = _start_ms(cfg["start_date"])
    installed_at = start - (rng.uniform(1, 730, size=n) * DAY_MS).astype(np.int64)

    return {
        "device_idx": np.arange(n, dtype=np.int64),
        "device_id": np.array([f"d_{i:09d}" for i in range(n)]),
        "tenant_id": tenants["tenant_id"][owner],
        "platform": platform,
        "sdk_version": sdk_version,
        "accuracy_profile": accuracy_profile,
        "behavior": behavior,
        "expected_daily_pings": expected_daily_pings,
        "home_lat": home_lat,
        "home_lng": home_lng,
        "anchor_lat": (anchor_offsets, a_lat),
        "anchor_lng": (anchor_offsets, a_lng),
        "is_spoofer": is_spoofer,
        "spoof_pattern": spoof_pattern,
        "installed_at": installed_at,
        "metro": np.array([metros[i]["name"] for i in range(len(metros))])[metro_idx],
        "_anchors_in_geofence": int(wants_gf.sum()),
        "_anchor_total": a_total,
        "_devices_with_gf_anchor": int(np.unique(a_dev[wants_gf]).size),
    }


# ---------------------------------------------------------------- shards

def _build_shard_plan(devices: dict[str, Any], cfg: dict[str, Any], seed: int,
                      plan_dir: Path, pings_dir: Path) -> list[ShardSpec]:
    n = len(devices["device_idx"])
    block_size = int(cfg["device_block_size"])
    n_days = int(cfg["simulated_days"])
    target = int(cfg["shard_target_pings"])

    n_blocks = math.ceil(n / block_size)
    expected = devices["expected_daily_pings"] * n_days
    block_of = devices["device_idx"] // block_size
    block_expected = np.bincount(block_of, weights=expected, minlength=n_blocks).astype(np.int64)

    if target < int(block_expected.max()):
        raise ValueError(
            f"shard_target_pings={target:,} is smaller than one block's expected pings "
            f"({int(block_expected.max()):,}); raise it or lower device_block_size")

    # Greedy: close a shard each time the running total crosses the next multiple of target.
    cumulative = np.cumsum(block_expected)
    cut = np.searchsorted(cumulative, target * np.arange(1, math.ceil(cumulative[-1] / target)), side="left") + 1
    starts = np.unique(np.concatenate(([0], cut[cut < n_blocks])))
    ends = np.concatenate((starts[1:], [n_blocks])) - 1  # inclusive

    start_ms = _start_ms(cfg["start_date"])
    specs = []
    for shard_id, (b0, b1) in enumerate(zip(starts, ends)):
        first_device = int(b0) * block_size
        last_device = min((int(b1) + 1) * block_size, n) - 1
        specs.append(ShardSpec(
            shard_id=shard_id,
            first_block=int(b0),
            last_block=int(b1),
            first_device=first_device,
            last_device=last_device,
            expected_pings=int(block_expected[b0:b1 + 1].sum()),
            start_ms=start_ms,
            n_days=n_days,
            seed=seed,
            block_size=block_size,
            plan_dir=str(plan_dir),
            out_dir=str(pings_dir),
        ))
    return specs


# ---------------------------------------------------------------- tables

def _list_array(offsets: np.ndarray, values: np.ndarray) -> pa.ListArray:
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(values, type=pa.float64()))


def _tenants_table(t: dict[str, np.ndarray]) -> pa.Table:
    return pa.table({
        "tenant_id": pa.array(t["tenant_id"], pa.string()),
        "name": pa.array(t["name"], pa.string()),
        "plan_tier": pa.array(t["plan_tier"], pa.int64()),
        "industry": pa.array(t["industry"], pa.int64()),
        "target_devices": pa.array(t["target_devices"], pa.int64()),
        "is_whale": pa.array(t["is_whale"], pa.bool_()),
        "home_metro": pa.array(t["home_metro"], pa.string()),
        "home_region": pa.array(t["home_region"], pa.string()),
        "timezone": pa.array(t["timezone"], pa.string()),
        "created_at": pa.array(t["created_at"], pa.timestamp("ms", tz="UTC")),
    })


def _geofences_table(g: dict[str, np.ndarray]) -> pa.Table:
    return pa.table({
        "geofence_id": pa.array(g["geofence_id"], pa.string()),
        "tenant_id": pa.array(g["tenant_id"], pa.string()),
        "name": pa.array(g["name"], pa.string()),
        "category": pa.array(g["category"], pa.string()),
        "shape_type": pa.array(g["shape_type"], pa.string()),
        "center_lat": pa.array(g["center_lat"], pa.float64()),
        "center_lng": pa.array(g["center_lng"], pa.float64()),
        "radius_m": pa.array(g["radius_m"], pa.float64()),
        "metro": pa.array(g["metro"], pa.string()),
    })


def _devices_table(d: dict[str, Any]) -> pa.Table:
    return pa.table({
        "device_idx": pa.array(d["device_idx"], pa.int64()),
        "device_id": pa.array(d["device_id"], pa.string()),
        "tenant_id": pa.array(d["tenant_id"], pa.string()),
        "platform": pa.array(d["platform"], pa.string()),
        "sdk_version": pa.array(d["sdk_version"], pa.string()),
        "accuracy_profile": pa.array(d["accuracy_profile"], pa.string()),
        "behavior": pa.array(d["behavior"], pa.string()),
        "expected_daily_pings": pa.array(d["expected_daily_pings"], pa.int64()),
        "home_lat": pa.array(d["home_lat"], pa.float64()),
        "home_lng": pa.array(d["home_lng"], pa.float64()),
        "anchor_lat": _list_array(*d["anchor_lat"]),
        "anchor_lng": _list_array(*d["anchor_lng"]),
        "is_spoofer": pa.array(d["is_spoofer"], pa.bool_()),
        "spoof_pattern": pa.array(d["spoof_pattern"], pa.string()),
        "installed_at": pa.array(d["installed_at"], pa.timestamp("ms", tz="UTC")),
        "metro": pa.array(d["metro"], pa.string()),
    })


def _shard_table(specs: list[ShardSpec]) -> pa.Table:
    rows = [asdict(s) for s in specs]
    return pa.table({k: [r[k] for r in rows] for k in rows[0]})


# ---------------------------------------------------------------- validation

def _validate_samples(tenants: pa.Table, devices: pa.Table, sample: int = 200) -> None:
    """Spot-check generated rows against the pydantic models."""
    for row in tenants.slice(0, sample).to_pylist():
        Tenant.model_validate(row)
    for row in devices.slice(0, sample).to_pylist():
        row["anchors"] = list(zip(row.pop("anchor_lat"), row.pop("anchor_lng")))
        Device.model_validate(row)


# ---------------------------------------------------------------- entry point

PLAN_DEFAULTS: dict[str, Any] = {
    "total_pings": 1_000_000,
    "tenant_count": 1000,
    "simulated_days": 1,
    "start_date": "2026-01-01",
    "whale_share": 0.40,
    "tenant_size_sigma": 1.5,
    "behavior_mix": {"stationary": 0.55, "walking": 0.30, "driving": 0.10, "mixed": 0.05},
    "ios_share": 0.55,
    "sdk_versions": {"4.2.0": 0.45, "4.1.3": 0.30, "3.9.2": 0.18, "3.2.1": 0.07},
    "accuracy_mix": {"high": 0.60, "medium": 0.30, "low": 0.10},
    "spoofer_rate": 0.01,
    "spoof_pattern_mix": {"teleport": 0.40, "static_while_moving": 0.30, "fixed_accuracy": 0.30},
    "metro_spread_km": 12.0,
    "device_home_metro_prob": 0.85,
    "max_anchors_per_device": 4,
    "geofence_anchor_prob": 0.35,
    "geofences_base": 2,
    "geofences_per_sqrt_device": 0.5,
    "geofences_max_per_tenant": 200,
    "geofence_radius_m": [50.0, 300.0],
    "device_block_size": 1000,
    "shard_target_pings": 5_000_000,
}


def _resolve_config(settings: Any) -> dict[str, Any]:
    """Merge settings (pydantic object or dict) over the plan defaults."""
    cfg = dict(PLAN_DEFAULTS)
    source = settings.model_dump() if hasattr(settings, "model_dump") else dict(settings or {})
    for key in PLAN_DEFAULTS:
        if source.get(key) is not None:
            cfg[key] = source[key]
    return cfg


def build_plan(settings: Any, *, seed: int | None = None, out_dir: str | Path = "./output",
               metros: list[dict] | None = None) -> dict[str, Any]:
    """Build and write the full plan. Returns a summary dict (also saved as plan_summary.json)."""
    cfg = _resolve_config(settings)
    source = settings.model_dump() if hasattr(settings, "model_dump") else dict(settings or {})
    seed = int(seed if seed is not None else source.get("global_seed", 42))
    metros = metros or DEFAULT_METROS

    out = Path(out_dir)
    plan_dir = out / "plan"
    pings_dir = out / "pings"
    plan_dir.mkdir(parents=True, exist_ok=True)

    # Size the device population from the ping budget (C1: weighted average first).
    beh_names, beh_p = _normalized(cfg["behavior_mix"], [b.value for b in Behavior], "behavior_mix")
    rates = np.array([EXPECTED_DAILY_PINGS[Behavior(b)] for b in beh_names], dtype=np.float64)
    avg_daily = float((beh_p * rates).sum())
    n_devices = max(int(cfg["tenant_count"]),
                    int(round(cfg["total_pings"] / (avg_daily * int(cfg["simulated_days"])))))

    tenants = _build_tenants(_rng(seed, _STREAM_TENANTS), cfg, n_devices, metros)
    geofences = _build_geofences(_rng(seed, _STREAM_GEOFENCES), cfg, tenants, metros)
    devices = _build_devices(_rng(seed, _STREAM_DEVICES), cfg, tenants, geofences, metros)
    specs = _build_shard_plan(devices, cfg, seed, plan_dir, pings_dir)

    tenants_tbl = _tenants_table(tenants)
    devices_tbl = _devices_table(devices)
    _validate_samples(tenants_tbl, devices_tbl)

    _write_atomic(tenants_tbl, plan_dir / "tenants.parquet")
    _write_atomic(_geofences_table(geofences), plan_dir / "geofences.parquet")
    _write_atomic(devices_tbl, plan_dir / "devices.parquet", row_group_size=100_000)
    _write_atomic(_shard_table(specs), plan_dir / "shard_plan.parquet")

    expected = devices["expected_daily_pings"] * int(cfg["simulated_days"])
    whale_mask = devices["tenant_id"] == tenants["tenant_id"][0]
    shard_sizes = np.array([s.expected_pings for s in specs])
    summary = {
        "seed": seed,
        "config": cfg,
        "plan_dir": str(plan_dir),
        "pings_dir": str(pings_dir),
        "tenants": int(len(tenants["tenant_id"])),
        "devices": int(len(devices["device_idx"])),
        "geofences": int(len(geofences["geofence_id"])),
        "avg_expected_daily_pings_per_device": round(avg_daily, 2),
        "expected_total_pings": int(expected.sum()),
        "whale": {
            "tenant_id": str(tenants["tenant_id"][0]),
            "devices": int(whale_mask.sum()),
            "expected_ping_share": round(float(expected[whale_mask].sum() / expected.sum()), 4),
        },
        "largest_non_whale_tenant_devices": int(tenants["target_devices"][1]) if len(tenants["target_devices"]) > 1 else 0,
        "median_tenant_devices": int(np.median(tenants["target_devices"])),
        "spoofers": int(devices["is_spoofer"].sum()),
        "spoofer_rate": round(float(devices["is_spoofer"].mean()), 4),
        "devices_with_geofence_anchor": devices["_devices_with_gf_anchor"],
        "anchors_in_geofences": devices["_anchors_in_geofence"],
        "anchors_total": devices["_anchor_total"],
        "blocks": int(math.ceil(len(devices["device_idx"]) / int(cfg["device_block_size"]))),
        "shards": len(specs),
        "shard_expected_pings": {
            "min": int(shard_sizes.min()),
            "max": int(shard_sizes.max()),
            "smallest_is_last": bool(shard_sizes.argmin() == len(shard_sizes) - 1),
        },
    }
    tmp = plan_dir / "plan_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, plan_dir / "plan_summary.json")
    return summary
