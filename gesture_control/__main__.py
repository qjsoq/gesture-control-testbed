import argparse
from pathlib import Path

from gesture_control.app import main

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def cli() -> None:
    parser = argparse.ArgumentParser(prog="gesture_control")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG.name} next to the package)",
        required=False,
    )
    args = parser.parse_args()
    main(args.config)


if __name__ == "__main__":
    cli()
