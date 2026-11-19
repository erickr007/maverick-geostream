# generator/devices.py
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

DEVICE_COLUMNS = [
    "device_idx", "device_id", "tenant_id", "platform", "sdk_version",
    "accuracy_profile", "behavior", "expected_daily_pings",
    "home_lat", "home_lng", "anchor_lat", "anchor_lng",
    "is_spoofer", "spoof_pattern",
]


@dataclass
class DeviceBatch:
    """A shard's devices as column arrays, sorted by device_idx."""
    device_idx: np.ndarray            # int64
    device_id: np.ndarray             # str
    tenant_id: np.ndarray             # str
    platform: np.ndarray              # str
    sdk_version: np.ndarray           # str
    accuracy_profile: np.ndarray      # str
    behavior: np.ndarray              # str
    expected_daily_pings: np.ndarray  # int64
    home_lat: np.ndarray              # float64
    home_lng: np.ndarray              # float64
    anchor_lat: np.ndarray            # float64, all devices' anchors flattened
    anchor_lng: np.ndarray            # float64, flattened, same layout
    anchor_offsets: np.ndarray        # int64, length n + 1
    is_spoofer: np.ndarray            # bool
    spoof_pattern: np.ndarray         # str

    def __len__(self) -> int:
        return len(self.device_idx)

    def anchors(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Anchor latitudes and longitudes for the i-th device in this batch."""
        start, end = self.anchor_offsets[i], self.anchor_offsets[i + 1]
        return self.anchor_lat[start:end], self.anchor_lng[start:end]

    def rows_for_block(self, block_id: int, block_size: int) -> slice:
        """Positions in this batch belonging to one device block."""
        lo = block_id * block_size
        hi = lo + block_size
        start = int(np.searchsorted(self.device_idx, lo, side="left"))
        end = int(np.searchsorted(self.device_idx, hi, side="left"))
        return slice(start, end)


def _flatten_list_column(col: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """Turn a list<double> column into (flat values, offsets)."""
    lengths = pc.list_value_length(col).to_numpy(zero_copy_only=False)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    values = pc.list_flatten(col).to_numpy(zero_copy_only=False)
    return values, offsets


def read_devices(plan_dir: str, first_device: int, last_device: int) -> DeviceBatch:
    """Load devices first_device..last_device (inclusive) from the plan."""
    dataset = ds.dataset(f"{plan_dir}/devices.parquet", format="parquet")
    in_range = (ds.field("device_idx") >= first_device) & (ds.field("device_idx") <= last_device)
    table = dataset.to_table(columns=DEVICE_COLUMNS, filter=in_range).sort_by("device_idx")

    # The plan must cover this range exactly; fail loudly if it doesn't.
    expected = last_device - first_device + 1
    idx = table["device_idx"].to_numpy()
    if table.num_rows != expected or idx[0] != first_device or idx[-1] != last_device:
        raise ValueError(
            f"devices {first_device}..{last_device}: expected {expected} rows, "
            f"got {table.num_rows}"
        )

    anchor_lat, lat_offsets = _flatten_list_column(table["anchor_lat"])
    anchor_lng, lng_offsets = _flatten_list_column(table["anchor_lng"])
    if not np.array_equal(lat_offsets, lng_offsets):
        raise ValueError("anchor_lat and anchor_lng lists have mismatched lengths")

    def col(name):
        return table[name].to_numpy(zero_copy_only=False)

    return DeviceBatch(
        device_idx=idx,
        device_id=col("device_id"),
        tenant_id=col("tenant_id"),
        platform=col("platform"),
        sdk_version=col("sdk_version"),
        accuracy_profile=col("accuracy_profile"),
        behavior=col("behavior"),
        expected_daily_pings=col("expected_daily_pings"),
        home_lat=col("home_lat"),
        home_lng=col("home_lng"),
        anchor_lat=anchor_lat,
        anchor_lng=anchor_lng,
        anchor_offsets=lat_offsets,
        is_spoofer=col("is_spoofer"),
        spoof_pattern=col("spoof_pattern"),
    )