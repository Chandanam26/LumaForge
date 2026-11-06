"""ONNX inference runtime used by the deployed API. No PyTorch dependency."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .hybrid import prepare
from .isp.pipeline import ISPConfig

MODEL_DIR = Path(os.environ.get("LUMAFORGE_MODELS", Path(__file__).resolve().parents[1] / "models"))
MODEL_FILES = {
    "fast": "lumanet_fast_int8.onnx",
    "quality": "lumanet_quality_fp16.onnx",
    "learned": "lumanet_learned_int8.onnx",
}
_sessions: dict[str, ort.InferenceSession] = {}


def session(name: str) -> ort.InferenceSession:
    """Lazily create one session per model; warm serverless instances reuse them."""
    if name not in _sessions:
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("LUMAFORGE_THREADS", "2"))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _sessions[name] = ort.InferenceSession(
            str(MODEL_DIR / MODEL_FILES[name]), options, providers=["CPUExecutionProvider"]
        )
    return _sessions[name]


def infer(name: str, packed: np.ndarray, noise: np.ndarray, base: np.ndarray) -> np.ndarray:
    output = session(name).run(
        ["rgb"], {"packed": packed[None], "noise": noise[None], "base": base[None]}
    )[0]
    return np.clip(output[0].transpose(1, 2, 0), 0.0, 1.0).astype(np.float32)


@dataclass
class Rendered:
    traditional: np.ndarray
    learned: np.ndarray | None
    hybrid: np.ndarray
    route: str
    model: str
    stats: dict
    controls: dict
    gains: list[float]
    timings_ms: dict[str, float]


def render(
    bayer: np.ndarray,
    camera: str | None = None,
    mode: str = "auto",
    include_learned: bool = True,
    compensation: float = 1.0,
) -> Rendered:
    """Render one RGGB mosaic through all arms of the comparison.

    ``mode`` is "auto" (use the adaptive route), "fast" or "quality".
    """
    timings: dict[str, float] = {}
    started = time.perf_counter()
    inputs = prepare(bayer, camera, ISPConfig(), compensation)
    timings["classical"] = (time.perf_counter() - started) * 1000
    result = inputs.result

    route = result.controls.route if mode == "auto" else mode
    started = time.perf_counter()
    hybrid = infer(route, inputs.packed, inputs.noise, inputs.base)
    timings["hybrid_net"] = (time.perf_counter() - started) * 1000

    learned = None
    if include_learned:
        started = time.perf_counter()
        learned = infer("learned", inputs.packed, inputs.noise, inputs.base)
        timings["learned_net"] = (time.perf_counter() - started) * 1000

    return Rendered(
        traditional=result.rgb,
        learned=learned,
        hybrid=hybrid,
        route=route,
        model=MODEL_FILES[route],
        stats=result.stats.to_dict(),
        controls=result.controls.to_dict(),
        gains=[float(g) for g in result.gains],
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )
