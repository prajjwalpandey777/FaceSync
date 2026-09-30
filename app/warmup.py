"""Inference warm-up (only relevant when the optional server-side mode is enabled)."""
import logging

from app.config import settings

log = logging.getLogger("facesync.warmup")


def warm_inference() -> None:
    """Load the server-side models if INFERENCE_MODE=local. In the default 'device' mode the browser
    does the work, so there is nothing to warm up."""
    if settings.INFERENCE_MODE.lower() != "local":
        return
    try:
        from app.inference import get_inference
        get_inference()
        log.info("Inference runtime is ready.")
    except Exception as exc:
        log.warning("Could not pre-warm inference: %s", exc)
