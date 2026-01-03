from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from .config import load_config
from .runner import AlertRunner


def _setup_logging(level: str) -> None:
    lvl = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Divergence Sentinel - Multi-TF alert bot")
    p.add_argument("--config", required=True, help="Path to YAML config")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    _setup_logging(cfg.app.log_level)

    runner = AlertRunner(cfg)

    async def _run() -> None:
        try:
            await runner.run_forever()
        finally:
            # Close shared REST session cleanly.
            try:
                await runner.provider.close()
            except Exception:
                pass

    try:
        asyncio.run(_run())
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        logging.getLogger("main").exception("fatal err=%s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
