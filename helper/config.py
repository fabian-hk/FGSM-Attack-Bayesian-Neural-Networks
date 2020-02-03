from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml


@dataclass
class Configuration:
    epsilons: List[float]
    id: int
    correct_std: float
    wrong_std: float
    threshold_std: float

    def __init__(self):
        self.id = 0
        self.epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def save(self):
        path = Path(f"data/{self.id}_config.yml")
        yaml.dump(asdict(self), path.open("w"))
