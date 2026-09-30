"""Record reproducibility metadata without guessing GPU or NPU usage."""

import platform
import subprocess


def _sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True, timeout=2,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect(device_label=None):
    info = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "device_label": device_label or "unspecified",
        "accelerator_used": "unverified",
    }
    if platform.system() == "Darwin":
        info["machine_model"] = _sysctl("hw.model")
        info["memory_bytes"] = int(_sysctl("hw.memsize") or 0) or None
        info["cpu_brand"] = _sysctl("machdep.cpu.brand_string")
    return info
