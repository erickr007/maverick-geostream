from datetime import datetime, timezone
import numpy as np
from api.dependencies import Settings

DAY_MS = 86_400_000

class PingTimeGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.rng = np.random.default_rng(settings.global_seed)
        self.start = datetime.fromisoformat(self.settings.start_date)
        self.start_ms = int(self.start.replace(tzinfo=timezone.utc).timestamp() * 1000) 

    def ping_times(self, n_pings, day_index):
        """Sorted epoch-ms timestamps for one device on one simulated day."""
        offsets = np.sort(self.rng.integers(0, DAY_MS, size=n_pings))
        return self.start_ms + day_index * DAY_MS + offsets