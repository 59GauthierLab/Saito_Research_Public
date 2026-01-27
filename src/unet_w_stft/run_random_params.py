import importlib
import random
from dataclasses import dataclass
from pathlib import Path

import config as cfg
import main as train_main


@dataclass(frozen=True)
class ParamSpec:
    lock: bool
    low: float
    high: float
    decimals: int


# ============================================================
# Random search settings
# ============================================================
# lock=True のパラメータは config.py の値をそのまま使う．
# lock=False のパラメータは指定した range からサンプルし，小数点 decimals 桁で丸める．
PARAM_SPECS: dict[str, ParamSpec] = {
    "LOG_WEIGHT": ParamSpec(lock=True, low=0.0, high=1.0, decimals=4),
    "MAG_WEIGHT": ParamSpec(lock=True, low=0.0, high=1.0, decimals=4),
    "SNR_WEIGHT": ParamSpec(lock=True, low=0.0, high=1.0, decimals=4),
    "SIL_WEIGHT": ParamSpec(lock=False, low=1e-1, high=5e-1, decimals=4),
    "SIL_ALPHA": ParamSpec(lock=False, low=1e-2, high=5e-1, decimals=4),
}


def _sample_value(spec: ParamSpec) -> float:
    lo = float(spec.low)
    hi = float(spec.high)
    if not (lo < hi):
        raise ValueError(f"invalid range: low={lo}, high={hi}")
    if lo <= 0.0:
        raise ValueError("requires low > 0")
    value = random.uniform(lo, hi)
    return round(value, int(spec.decimals))


def _format_value(value: float, decimals: int) -> str:
    return format(float(value), f".{int(decimals)}f")


def _apply_params() -> None:
    for name, spec in PARAM_SPECS.items():
        if spec.lock:
            continue
        setattr(cfg, name, _sample_value(spec))


def _update_output_dir() -> None:
    base_name = Path(cfg.OUTPUT_DIR).name
    prefix = base_name.rsplit("_", 1)[0]

    def v(name: str) -> str:
        spec = PARAM_SPECS.get(name)
        decimals = spec.decimals if spec else 6
        return _format_value(getattr(cfg, name), decimals)

    cfg.OUTPUT_DIR = str(
        Path("random_param_out")
        / f"{prefix}_{v('LOG_WEIGHT')}-{v('MAG_WEIGHT')}-{v('SNR_WEIGHT')}-{v('SIL_WEIGHT')}-{v('SIL_ALPHA')}"  # noqa: E501
    )
    Path(cfg.OUTPUT_DIR).parent.mkdir(parents=True, exist_ok=True)


def _cleanup_torch() -> None:
    try:
        import torch
    except ImportError:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def main() -> None:
    run_idx = 0
    while True:
        run_idx += 1
        importlib.reload(cfg)
        _apply_params()
        _update_output_dir()

        print(
            f"[RUN {run_idx}] \
                LOG={cfg.LOG_WEIGHT} MAG={cfg.MAG_WEIGHT} SNR={cfg.SNR_WEIGHT} SIL={cfg.SIL_WEIGHT} ALPHA={cfg.SIL_ALPHA}"  # noqa: E501
        )
        print(f"    -> {cfg.OUTPUT_DIR}")
        try:
            train_main.main()
        except KeyboardInterrupt:
            print("Interrupted. Stopping.")
            break
        _cleanup_torch()


if __name__ == "__main__":
    main()
