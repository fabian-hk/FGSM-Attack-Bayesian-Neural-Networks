from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml


@dataclass
class Configuration:
    id: int
    epsilons: List[float]
    bnn_adversary_samples: int
    bnn_training_epochs: int
    nn_training_epochs: int
    threshold_std: float
    correct_std: float
    wrong_std: float

    def __init__(self):
        self.id = 1
        self.epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.bnn_adversary_samples = 40
        self.bnn_training_epochs = 5
        self.nn_training_epochs = 2

    def save(self):
        path = Path(f"data/{self.id:02}_config.yml")
        yaml.dump(asdict(self), path.open("w"), sort_keys=False)
