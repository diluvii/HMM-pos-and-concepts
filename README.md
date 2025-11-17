# Hidden Markov model for POS tagging & for categorization
By Yawen Xue. Final project for COGS 21: Computational Neuroscience!

**Algorithm:** HMM

**Datasets:** 
- Brown Corpus (present-day American English)
- Experimental stimuli from Medin & Schaffer (1978): 16 items that vary along 4 binary features

## On Medin & Schaffer (1978)
[landmark study in categorical learning; summarize XXX]

## Report

### POS tagging

### Categorical learning

## Running instructions
Clone from GitHub, create a virtual environment, install dependencies, run.
```
git clone https://github.com/yawenx2004/HMM-pos-and-concepts
cd hmm_pos_and_concepts

python3 -m venv venv
source venv/bin/activate
pip install numpy

python3 test.py
```

## Build process
Building the model has 3 main steps: training, decoding, and testing. Here's how I did them—

### Training
Training an HMM consists of building **transition probabilities** (probability each state goes to any next state) and **emission probabilities** (probabililty each state is emitted by any observation). I used bigrams for transition probabilities.

For both I first made a counts matrix. In the transition counts matrix ```tcounts```, the rows and columns both represent states. Whenever state *i* is followed by state *j*, ```tcounts[i][j]``` is incremented. In the emission counts matrix ```ecounts```, the rows represent states and the columns represent observations. Whenever state *i* is emitted by observation *j*, ```ecounts[i][j]``` is incremented.

Then my helper function ```to_log()``` applies smoothing to the matrices by incrementing each entry by 0.1 (so that we don't run into the problem of having to take the log of 0), sums the total for each row, and derives the **log likelihood** of each entry by taking the log of each entry divided by the total.

## Resources
- Context theory of classification learning ([Medin & Schaffer, 1978](https://www.semanticscholar.org/paper/Context-theory-of-classification-learning.-Medin-Schaffer/2cd7154ef2e19d4733bfac30f04ed708f01b42d1))
- I referred to the [CS10 instructions](https://www.cs.dartmouth.edu/~cs10/PS-5.html) for implementing HMM & Viterbi, and in particular for POS tagging downloaded the test files from there. Please note, however, that I did not implement the lab following these instructions, and instead used them as a refresher on what the transition and emission matrices should contain.
