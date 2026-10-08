"""Best-effort read-only Apple telemetry. Never translate this counter into DCGM idle."""

import math
import platform
import plistlib
import subprocess
import time


def local_sample(*, runner=subprocess.run):
    sample = {
        "wall_time_ns": time.time_ns(),
        "platform": platform.system(),
        "gpu_activity_percent": None,
        "gpu_memory_bytes": None,
        "scope": "system-wide Apple GPU; shared with all applications",
        "measurement": "Apple driver activity counter; not DCGM engine inactivity",
        "status": "unsupported_platform",
    }
    if platform.system() != "Darwin":
        return sample
    try:
        result = runner(
            ["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"],
            capture_output=True,
            timeout=3,
            check=True,
        )
        devices = plistlib.loads(result.stdout)
        # One system device only: do not silently average distinct GPU counter semantics.
        if len(devices) != 1:
            sample["status"] = "ambiguous_or_missing_device"
            return sample
        stats = devices[0].get("PerformanceStatistics", {})
        value = stats.get("Device Utilization %")
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and 0 <= value <= 100
        ):
            sample["gpu_activity_percent"] = value
            sample["status"] = "available"
        else:
            sample["status"] = "counter_unavailable"
        memory = stats.get("In use system memory")
        if isinstance(memory, int) and not isinstance(memory, bool) and memory >= 0:
            sample["gpu_memory_bytes"] = memory
    except (OSError, ValueError, subprocess.SubprocessError, TypeError, AttributeError):
        sample["status"] = "read_unavailable"
    return sample
