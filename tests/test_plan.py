"""Tests for the deterministic plan step. Run with: python -m pytest tests/test_plan.py"""
import hashlib

import numpy as np
import pyarrow.parquet as pq
import pytest

from models.devicebatch import read_devices
from models.shardspec import load_plan
from services.generator.plan import build_plan

SMALL = {
    "global_seed": 42,
    "start_date": "2026-01-01",
    "simulated_days": 1,
    "total_pings": 200_000,
    "tenant_count": 50,
    "device_block_size": 50,
    "shard_target_pings": 30_000,
}


def _table_hash(plan_dir, name, drop=()):
    table = pq.read_table(f"{plan_dir}/{name}.parquet")
    if drop:
        table = table.drop(list(drop))
    return hashlib.sha256(str(table.to_pydict()).encode()).hexdigest()


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    root = tmp_path_factory.mktemp("plans")
    a = build_plan(SMALL, out_dir=root / "a")
    b = build_plan(SMALL, out_dir=root / "b")
    c = build_plan(SMALL, seed=43, out_dir=root / "c")
    return a, b, c


@pytest.mark.parametrize("name", ["tenants", "geofences", "devices", "shard_plan"])
def test_same_seed_same_tables_different_seed_different(plans, name):
    a, b, c = plans
    drop = ("plan_dir", "out_dir") if name == "shard_plan" else ()
    ha, hb, hc = (_table_hash(p["plan_dir"], name, drop) for p in (a, b, c))
    assert ha == hb
    assert ha != hc


def test_whale_share_and_spoof_rate(plans):
    summary = plans[0]
    assert abs(summary["whale"]["expected_ping_share"] - 0.40) < 0.05
    assert summary["spoofer_rate"] < 0.05


def test_shards_cover_all_devices_and_match_read_devices(plans):
    summary = plans[0]
    specs = load_plan(f"{summary['plan_dir']}/shard_plan.parquet")
    assert specs[0].first_device == 0
    assert specs[-1].last_device == summary["devices"] - 1
    for prev, nxt in zip(specs, specs[1:]):
        assert prev.last_device + 1 == nxt.first_device
    for spec in specs:
        batch = read_devices(spec.plan_dir, spec.first_device, spec.last_device)
        assert len(batch) == spec.last_device - spec.first_device + 1
        assert int((batch.expected_daily_pings * spec.n_days).sum()) == spec.expected_pings


def test_only_spoofers_have_patterns(plans):
    devices = pq.read_table(f"{plans[0]['plan_dir']}/devices.parquet").to_pandas()
    assert (devices.loc[~devices.is_spoofer, "spoof_pattern"] == "none").all()
    assert (devices.loc[devices.is_spoofer, "spoof_pattern"] != "none").all()


def test_some_anchors_inside_geofences(plans):
    assert plans[0]["anchors_in_geofences"] > 0
