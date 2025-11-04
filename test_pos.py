from HMM import HMM
from pathlib import Path

base = Path(__file__).parent
tags_path = Path(base / "inputs" / "brown-train-tags.txt")
sentences_path = Path(base / "inputs" / "brown-train-sentences.txt")

# Here, we infer POS tags (states) from words (observations)
model = HMM(tags_path, sentences_path, "#")
model.train()
model.training_check()