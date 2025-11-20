from pathlib import Path
from HMM import HMM

base = Path(__file__).parent
tags_path = Path(base / "inputs" / "brown-train-tags.txt")
sentences_path = Path(base / "inputs" / "brown-train-sentences.txt")
test_tags_path = Path(base / "inputs" / "brown-test-tags.txt")
test_sentences_path = Path(base / "inputs" / "brown-test-sentences.txt")

# train model on pos tags
model = HMM(tags_path, sentences_path)
model.train()

print("---")
model.cli()
print("---")
model.test_on(test_sentences_path, test_tags_path)