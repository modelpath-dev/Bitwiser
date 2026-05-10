"""Lightweight system telemetry: RAM, energy (powermetrics), tokens/sec."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

import psutil


@dataclass
class Vitals:
    timestamp: float
    rss_bytes: int  # process RSS
    rss_total_bytes: int  # OS-wide used memory
    cpu_pct: float
    package_watts: float | None = None  # CPU+GPU package power, if powermetrics is on


def sample_process(pid: int) -> Vitals:
    try:
        p = psutil.Process(pid)
        rss = p.memory_info().rss
        cpu = p.cpu_percent(interval=None)
    except psutil.NoSuchProcess:
        rss = 0
        cpu = 0.0
    vm = psutil.virtual_memory()
    return Vitals(
        timestamp=time.time(),
        rss_bytes=rss,
        rss_total_bytes=vm.used,
        cpu_pct=cpu,
    )


class PowermetricsSampler:
    """Read CPU+GPU package power via `sudo powermetrics`. Optional — silently no-ops
    if powermetrics isn't accessible (no sudo, not on macOS, etc.)."""

    _PATTERN = re.compile(r"(CPU Power|GPU Power|Package Power):\s+(\d+(?:\.\d+)?)\s+mW")

    def __init__(self, interval_ms: int = 500):
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.latest_watts: float | None = None
        self.available = shutil.which("powermetrics") is not None

    def start(self) -> bool:
        if not self.available:
            return False
        try:
            self._proc = subprocess.Popen(
                [
                    "sudo", "-n", "powermetrics",
                    "--samplers", "cpu_power,gpu_power",
                    "-i", str(self.interval_ms),
                    "-f", "text",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError:
            return False
        # If sudo prompts (no -n cached), powermetrics will exit quickly.
        time.sleep(0.2)
        if self._proc.poll() is not None:
            self._proc = None
            return False

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return True

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        cpu_w = gpu_w = 0.0
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            m = self._PATTERN.search(line)
            if not m:
                continue
            label, val = m.group(1), float(m.group(2)) / 1000.0
            if label == "CPU Power":
                cpu_w = val
            elif label == "GPU Power":
                gpu_w = val
                self.latest_watts = cpu_w + gpu_w
            elif label == "Package Power":
                self.latest_watts = val

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass
            self._proc = None


class VitalsRecorder:
    """Polls a child process's vitals into a list. Use as a context manager."""

    def __init__(self, pid: int, interval_s: float = 0.25, power: PowermetricsSampler | None = None):
        self.pid = pid
        self.interval_s = interval_s
        self.power = power
        self.samples: list[Vitals] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "VitalsRecorder":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = sample_process(self.pid)
            if self.power is not None:
                v.package_watts = self.power.latest_watts
            self.samples.append(v)
            time.sleep(self.interval_s)

    def peak_rss(self) -> int:
        return max((s.rss_bytes for s in self.samples), default=0)

    def avg_watts(self) -> float | None:
        vals = [s.package_watts for s in self.samples if s.package_watts is not None]
        return sum(vals) / len(vals) if vals else None


def humanize_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.2f} TB"
