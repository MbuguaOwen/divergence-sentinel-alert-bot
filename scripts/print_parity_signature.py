from __future__ import annotations

import argparse
import pprint

from divergence_alert_bot.config import load_config
from divergence_alert_bot.parity import parity_signature, PINE_DEFAULTS_REFERENCE


def main():
    p = argparse.ArgumentParser(description="Print tv_parity signature for a config")
    p.add_argument("--config", required=True, help="Path to YAML config")
    args = p.parse_args()

    cfg = load_config(args.config)
    sig = parity_signature(cfg.strategy)

    print("TV PARITY INPUTS:")
    pprint.pprint(sig)
    print("\nPINE DEFAULTS REFERENCE:")
    pprint.pprint(PINE_DEFAULTS_REFERENCE)


if __name__ == "__main__":
    main()
