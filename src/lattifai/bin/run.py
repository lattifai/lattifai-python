from dataclasses import dataclass

import nemo_run as run


@dataclass
class ModelConfig:
    size: int = 128
    layers: int = 2


@run.cli.entrypoint
def train(model_config: ModelConfig):
    print(f"Model size: {model_config.size}, layers: {model_config.layers}")


def main():
    run.cli.main(train)


if __name__ == "__main__":
    main()
