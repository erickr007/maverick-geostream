from dataclasses import dataclass

@dataclass(frozen=True)
class ShardSpec:
    shard_id: int
    first_block: int
    last_block: int
    first_device: int
    last_device: int
    expected_pings: int
    start_ms: int
    n_days: int
    seed: int
    block_size: int
    plan_dir: str
    out_dir: str

def load_plan(path) -> list[ShardSpec]:
    import pyarrow.parquet as pq
    return [ShardSpec(**row) for row in pq.read_table(path).to_pylist()]