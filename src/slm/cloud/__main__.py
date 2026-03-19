"""CLI entry point: python -m slm.cloud <command>."""

from __future__ import annotations

import argparse
import logging
import sys

from . import provisioner


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m slm.cloud",
        description="vast.ai cloud training pipeline for SLM",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- launch ----
    p_launch = sub.add_parser("launch", help="Launch a training run on vast.ai")
    p_launch.add_argument("--config", required=True, help="Path to experiment YAML config")
    p_launch.add_argument("--hf-repo", required=True, help="HuggingFace repo to publish model (e.g. user/model-name)")
    p_launch.add_argument("--gpu", default=None, help="Force specific GPU type (e.g. RTX_4090)")
    p_launch.add_argument("--max-price", type=float, default=0.50, help="Max $/hr (default: 0.50)")
    p_launch.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    p_launch.add_argument("--disk", type=int, default=60, help="Disk space in GB (default: 60)")
    p_launch.add_argument("--monitor", action="store_true", help="Tail training log after launch")
    p_launch.add_argument("--dry-run", action="store_true", help="Show GPU selection without creating instance")
    p_launch.add_argument("--train-module", default="slm.train", help="Training module (default: slm.train)")
    p_launch.add_argument("--pre-cmd", default="", help="Command to run before training (e.g. dataset preparation)")

    # ---- monitor ----
    p_monitor = sub.add_parser("monitor", help="Tail training log on a running instance")
    p_monitor.add_argument("--instance-id", type=int, required=True, help="vast.ai instance ID")

    # ---- status ----
    sub.add_parser("status", help="List all active vast.ai instances")

    # ---- destroy ----
    p_destroy = sub.add_parser("destroy", help="Destroy a vast.ai instance")
    p_destroy.add_argument("--instance-id", type=int, required=True, help="vast.ai instance ID")

    # Split on '--' to separate our args from extra training args
    raw = argv if argv is not None else sys.argv[1:]
    if "--" in raw:
        sep = raw.index("--")
        our_args, extra = raw[:sep], raw[sep + 1:]
    else:
        our_args, extra = raw, []

    args = parser.parse_args(our_args)

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Dispatch
    if args.command == "launch":
        extra_train_args = " ".join(extra) if extra else ""
        provisioner.launch(
            config_path=args.config,
            hf_repo=args.hf_repo,
            gpu=args.gpu,
            max_price=args.max_price,
            num_gpus=args.num_gpus,
            disk=args.disk,
            do_monitor=args.monitor,
            dry_run=args.dry_run,
            extra_train_args=extra_train_args,
            train_module=args.train_module,
            pre_train_cmd=args.pre_cmd,
        )

    elif args.command == "monitor":
        provisioner.monitor(args.instance_id)

    elif args.command == "status":
        provisioner.status()

    elif args.command == "destroy":
        provisioner.destroy(args.instance_id)


if __name__ == "__main__":
    main()
