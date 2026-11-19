from models.device import AccuracyProfile, Behavior, Device, Platform, SpoofPattern
from models.ping import Ping
from models.tenant import Industry, PlanTier, Tenant
from services.generator.generator import PING_SCHEMA, Generator


def test_ping_schema_fields_match_required_contract():
    assert set(Ping.model_fields) == {
        "ping_id",
        "device_id",
        "tenant_id",
        "ts",
        "lat",
        "lng",
        "accuracy_m",
        "speed_mps",
        "platform",
        "sdk_version",
        "is_spoofed",
    }


def test_generator_schema_contains_required_fields():
    assert set(PING_SCHEMA.keys()) == {
        "ping_id",
        "device_id",
        "tenant_id",
        "ts",
        "lat",
        "lng",
        "accuracy_m",
        "speed_mps",
        "platform",
        "sdk_version",
        "is_spoofed",
    }


def test_tenant_schema_fields_match_required_contract():
    assert set(Tenant.model_fields) == {
        "tenant_id",
        "name",
        "plan_tier",
        "industry",
        "target_devices",
        "is_whale",
        "home_region",
        "timezone",
        "created_at",
    }
    assert PlanTier.FREE.value == 0
    assert Industry.RETAIL.value == 0


def test_device_schema_fields_match_required_contract():
    assert set(Device.model_fields) == {
        "device_id",
        "tenant_id",
        "platform",
        "sdk_version",
        "accuracy_profile",
        "behavior",
        "expected_daily_pings",
        "home_lat",
        "home_lng",
        "anchors",
        "is_spoofer",
        "spoof_pattern",
        "installed_at",
    }
    assert Platform.IOS.value == "ios"
    assert AccuracyProfile.HIGH.value == "high"
    assert Behavior.DRIVING.value == "driving"
    assert SpoofPattern.NONE.value == "none"


def test_generator_supports_plan_generate_and_verify_commands():
    settings = {
        "tenant_count": 2,
        "simulated_days": 2,
        "start_date": "2026-01-01",
        "global_seed": 42,
    }
    generator = Generator(settings)

    plan = generator.plan()
    assert set(plan.keys()) == {"tenants", "devices", "shards", "seed", "output_dir"}
    assert plan["tenants"] >= 1
    assert plan["shards"] >= 1

    generated = generator.generate(out_dir="./output")
    assert generated["device_count"] >= 1
    assert generated["ping_count"] >= 1

    verification = generator.verify(out_dir="./output")
    assert verification["ok"] is True
