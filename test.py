from HMM import HMM
from pathlib import Path

base = Path(__file__).parent
tags_path = Path(base / "data" / "inputs" / "simple-train-tags.txt")
sentences_path = Path(base / "data" / "inputs" / "simple-train-sentences.txt")
test_tags_path = Path(base / "data" / "inputs" / "simple-test-tags.txt")
test_sentences_path = Path(base / "data" / "inputs" / "simple-test-sentences.txt")

# train model on pos tags
model = HMM(tags_path, sentences_path)
model.train()
# model.training_check()

print("---")
model.cli()
print("---")
model.test_on(test_sentences_path)