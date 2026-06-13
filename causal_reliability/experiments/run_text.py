import argparse

from causal_reliability.data.text_shortcuts import make_text_task
from causal_reliability.experiments.common import run_task
from causal_reliability.utils.config import load_config
from causal_reliability.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/text.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    bundle = make_text_task(**cfg.get("data", {}))
    print(run_task("text", bundle, cfg))


if __name__ == "__main__":
    main()
