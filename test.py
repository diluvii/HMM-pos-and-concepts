from pathlib import Path
from HMM import HMM

base = Path(__file__).parent

################################################################################
### POS TAGGING
################################################################################
def test_pos():
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
    model.test_on(test_sentences_path, key_file=test_tags_path)
    print()
    print()

################################################################################
### CATEGORICAL LEARNING
################################################################################
def test_cat(v=1):
    if v == 1:
        cat_path = Path(base / "inputs" / "ms-train-obs.txt")
        obs_path = Path(base / "inputs" / "ms-train-st.txt")
        test_st_path = Path(base / "inputs" / "ms-test-obs.txt")
    
    if v == 2:
        cat_path = Path(base / "inputs" / "ms-train-obs-2.txt")
        obs_path = Path(base / "inputs" / "ms-train-st-2.txt")
        test_st_path = Path(base / "inputs" / "ms-test-obs-2.txt")

    # train model on observations & categories
    model = HMM(obs_path, cat_path)
    model.train()
    model.training_check()

    print("---")
    model.test_on(test_st_path)

    print()
    print()

################################################################################
### ACTUALLY RUNNING THE FUNCTION WE DEFINED FOR TESTING
################################################################################
# test_pos()      # POS tagging
test_cat(1)     # categorical learning tagged (1001 | A)
# test_cat(2)     # categorical learning tagged (1 0 0 1 | A A A A)