from causal_reliability.experiments.run_synthetic import main as synthetic_main
from causal_reliability.experiments.run_tabular import main as tabular_main
from causal_reliability.experiments.run_text import main as text_main
from causal_reliability.experiments.run_vision import main as vision_main


def main() -> None:
    synthetic_main()
    vision_main()
    text_main()
    tabular_main()


if __name__ == "__main__":
    main()
