"""Resource-aware Metal-accelerated inference via llama-cli."""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import psutil

from prompts import format_prompt

ROOT = Path(__file__).resolve().parent
LLAMA_BIN = ROOT / "vendor" / "llama.cpp" / "build" / "bin"
# llama-completion is the non-interactive single-prompt binary. llama-cli is
# always interactive when the model ships a chat template (Mistral, Llama-3,
# Gemma) — its `-no-cnv` flag is documented but errors at runtime
# ("--no-conversation is not supported by llama-cli; please use llama-completion
# instead"). llama-completion runs the prompt and exits cleanly.
CLI_BIN = LLAMA_BIN / "llama-completion"

# llama-completion appends this marker to stdout when generation ends. We
# buffer it out of the user-visible stream.
EOF_MARKER = "> EOF by user"


@dataclass
class RunStats:
    tokens: int
    elapsed_s: float
    peak_rss_bytes: int

    @property
    def tokens_per_s(self) -> float:
        return self.tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _physical_cores() -> int:
    n = psutil.cpu_count(logical=False)
    return n if n and n > 0 else 4


def optimal_thread_count() -> int:
    """Use all performance cores. On Apple Silicon, psutil reports physical cores;
    we use all of them (efficiency cores are mostly counterproductive for inference)."""
    cores = _physical_cores()
    # Heuristic: on M-series, P-cores ~= total physical - 4 efficiency cores for older
    # chips, but newer chips (M4/M5) flip the ratio. Use cores - 2 with a floor of 4.
    return max(4, cores - 2)


def model_will_fit(path: str | Path) -> tuple[bool, int, int]:
    """Return (fits, model_bytes, ram_bytes). Treats <0.85x available RAM as fitting."""
    size = Path(path).stat().st_size
    avail = psutil.virtual_memory().available
    return size < int(avail * 0.85), size, avail


def build_command(
    model_path: str,
    prompt: str,
    threads: int | None = None,
    n_predict: int = 256,
    ctx: int = 4096,
    use_mmap: bool = True,
    stop_tokens: list[str] | None = None,
) -> list[str]:
    if not CLI_BIN.exists():
        raise FileNotFoundError(f"llama-completion not found at {CLI_BIN}. Run ./setup.sh first.")
    cmd: list[str] = [
        str(CLI_BIN),
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "-c", str(ctx),
        "-t", str(threads or optimal_thread_count()),
        "-ngl", "99",         # offload all layers to Metal GPU
        "--no-display-prompt",
        "--simple-io",
    ]
    if not use_mmap:
        cmd.append("--no-mmap")
    if stop_tokens:
        for s in stop_tokens:
            cmd.extend(["-r", s])
    return cmd


def _kill_proc(proc: subprocess.Popen) -> None:
    """Best-effort terminate, then SIGKILL after 2s. Always reaps."""
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass
    except OSError:
        pass


def run_streaming(
    model_path: str,
    model_name: str,
    architecture: str | None,
    user_prompt: str,
    n_predict: int = 256,
    ctx: int = 4096,
    timeout_s: float = 120.0,
    runaway_blank_chars: int = 400,
    proc_sink: list | None = None,
) -> Iterator[tuple[str, RunStats | None]]:
    """Yield (token_chunk, None) as text streams in. Final yield is ('', RunStats).

    The Popen handle is appended to `proc_sink` (if provided) so the caller can
    kill it if the consumer (SSE client) disconnects. Also enforces:
      - hard wall-clock timeout (`timeout_s`)
      - runaway-whitespace guard (kills if >`runaway_blank_chars` in a row)
      - generator-level cleanup: `finally` block always reaps the subprocess."""
    fits, size, avail = model_will_fit(model_path)
    use_mmap = True
    if not fits:
        yield (
            f"[warning] model is {size/1e9:.1f} GB but only {avail/1e9:.1f} GB RAM is "
            f"free; falling back to mmap (will be slow).\n",
            None,
        )

    formatted, stop = format_prompt(model_name, architecture, user_prompt)
    cmd = build_command(
        model_path,
        formatted,
        n_predict=n_predict,
        ctx=ctx,
        use_mmap=use_mmap,
        stop_tokens=stop,
    )

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    if proc_sink is not None:
        proc_sink.append(proc)
    assert proc.stdout is not None

    peak_rss = 0
    tokens = 0
    consecutive_blank = 0
    try:
        ps = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        ps = None

    # Tail buffer: hold back the last `tail_len` chars from the stream so we
    # can drop the trailing "> EOF by user" marker after the process exits.
    tail_len = len(EOF_MARKER) + 4  # margin for surrounding newlines/spaces
    tail: list[str] = []

    def _flush_oldest() -> str | None:
        if len(tail) > tail_len:
            return tail.pop(0)
        return None

    try:
        while True:
            if time.perf_counter() - start > timeout_s:
                _kill_proc(proc)
                yield (f"\n[timeout] killed after {timeout_s:.0f}s\n", None)
                break

            ch = proc.stdout.read(1)
            if not ch:
                break

            if ch.strip():
                consecutive_blank = 0
                tokens += 1
            else:
                consecutive_blank += 1
                if consecutive_blank >= runaway_blank_chars:
                    _kill_proc(proc)
                    yield (
                        f"\n[runaway] killed after {runaway_blank_chars} "
                        f"consecutive whitespace chars (degraded quantization)\n",
                        None,
                    )
                    break

            tail.append(ch)
            old = _flush_oldest()
            if old is not None:
                yield (old, None)

            if ps is not None and tokens % 16 == 0:
                try:
                    rss = ps.memory_info().rss
                    if rss > peak_rss:
                        peak_rss = rss
                except psutil.NoSuchProcess:
                    pass

        # Process exited: decide whether the held-back tail is the EOF marker
        # (and should be dropped) or genuine output (and should be flushed).
        trailing = "".join(tail)
        if EOF_MARKER in trailing:
            keep = trailing.split(EOF_MARKER, 1)[0].rstrip()
            if keep:
                yield (keep, None)
        else:
            if trailing:
                yield (trailing, None)
    finally:
        # If the consumer abandoned us (SSE client disconnected, etc.) this
        # finally runs and reaps the subprocess so the model unloads from RAM.
        _kill_proc(proc)

    elapsed = time.perf_counter() - start
    yield ("", RunStats(tokens=tokens, elapsed_s=elapsed, peak_rss_bytes=peak_rss))


def run_blocking(
    model_path: str,
    model_name: str,
    architecture: str | None,
    user_prompt: str,
    n_predict: int = 256,
) -> tuple[str, RunStats]:
    text_parts: list[str] = []
    stats: RunStats | None = None
    for chunk, s in run_streaming(model_path, model_name, architecture, user_prompt, n_predict):
        if s is not None:
            stats = s
        else:
            text_parts.append(chunk)
    assert stats is not None
    return "".join(text_parts), stats
