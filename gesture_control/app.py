import logging
import signal
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from gesture_control.metrics.logging import setup_logging
from gesture_control.pipeline import run
from gesture_control.recognizers import RecognizerConfig, build_recognizer
from gesture_control.sources import SourceConfig, build_source, get_source_config
from gesture_control.viewer import save_annotated, show

logger = logging.getLogger(__name__)


class MetricsConfig(BaseModel):
    log_level: str = "INFO"
    log_fps_every: int = 100
    display: bool = False
    output_path: Path | None = None
    max_frames: int | None = 1


class AppConfig(BaseModel):
    source: SourceConfig
    recognizer: RecognizerConfig
    runtime: MetricsConfig = MetricsConfig()
    
    @model_validator(mode='before')
    @classmethod
    def set_specific_source_config(cls, raw_data: dict):
        specific_type: str = raw_data.get("source", "").get("type", "")
        config_class = get_source_config(specific_type)
        if config_class:
            raw_data["source"] = config_class.model_validate(raw_data.get("source", {}))
        return raw_data


def load_config(path: Path) -> AppConfig:
    with path.open(encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def _install_signal_handlers(stop: threading.Event) -> None:
    def _handler(signum: int, _frame: object) -> None:
        logger.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    # SIGTERM is not supported on Windows for non-main threads; ignore if missing.
    try:
        signal.signal(signal.SIGTERM, _handler)
    except (AttributeError, ValueError):
        pass


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg.runtime.log_level)
    logger.info("config loaded path=%s", config_path)

    source = build_source(cfg.source)
    recognizer = build_recognizer(cfg.recognizer)

    stop = threading.Event()
    _install_signal_handlers(stop)

    if cfg.runtime.display:
        show(source, recognizer, stop=stop, log_fps_every=cfg.runtime.log_fps_every)
        logger.info("viewer finished")
        return

    if cfg.runtime.output_path is not None:
        save_annotated(
            source,
            recognizer,
            cfg.runtime.output_path,
            stop=stop,
            log_fps_every=cfg.runtime.log_fps_every,
            max_frames=cfg.runtime.max_frames,
        )
        logger.info("save finished")
        return

    n = 0
    for prediction in run(
        source,
        recognizer,
        stop=stop,
        log_fps_every=cfg.runtime.log_fps_every,
    ):
        n += 1
        logger.info(
            "gesture label=%s confidence=%.2f raw=%s",
            prediction.label,
            prediction.confidence,
            prediction.raw,
        )
    logger.info("pipeline finished predictions=%d", n)
