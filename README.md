# HMM
Final project for COGS 21—I implemented a Hidden Markov Model (HMM) and used it on POS tagging, which yielded decent enough results. I then ran the HMM on the conceptual learning dataset from Medin & Schafer (1978) to see what it would do.

An HMM infers hidden states from a series of observations.

(Author: Yawen Xue)

## Running instructions
After you clone the repository and `cd` into it, here is how you run the code—
```bash
echo "first make a virtual environment & install dependencies"
python3 -m ven venv
source venv/bin/activate
pip install -r requirements

echo "then run test.py to see how the HMM performs"
python3 test.py
```

`test.py` contains, for both POS tagging and categorical learning, the following test(s)—
- A function that takes a test observations file as a parameter and returns a test states file (e.g. for POS tagging, it takes a sentences file and returns a tags file). This function also checks the output states file against a given states file and reports the accuracy of the HMM predictions.
- For POS tagging only, a command line interface (CLI) that prompts you to enter via terminal an observation to be parsed into states (e.g. for POS tagging, enter a sentence and see how the HMM parses it).

### Directory structure
```
HMM/
├── img/                    # screenshots for documentation
├── inputs/                 # train & test data
├── HMM.py                  # contains HMM implementation
├── Spinner.py              # spinner for command line UX; displayed when in progress
├── test.py                 # tests for model
├── README.md
├── requirements.txt
└── .gitignore
```

## Report
According to the specs in the instructions here is my report—

(Please note that there are no success metrics for Medin & Schafer 1978 since it was testing transfer! More on this later)

- **Data:** For training, a pair of files (one with observations and one with corresponding states); for testing, also a pair of files (one with observations and one with corresponding states), although if you don't need to verify accuracy you wouldn't need the testing states file because this one the HMM will write
    - **POS tagging:** I used `.txt` files from the **Brown Corpus**. Here is an example sentence: 
    ````
    meanwhile , it was learned the state highway department is very near being ready to issue the first $30 million worth of highway reconstruction bonds .
    ````
    And here is an example corresponding tags set: 

    ````
    `` ADV DET ADJ N P ADJ N V VN '' , DET N VD , `` P DET ADJ N P DET N , DET N P N CNJ DET N P DET N '' .`
    ````
    The testing files are similarly formatted. I made no adjustments to formatting, nor processing of data.
    - **Categorical learning:** Experimental stimuli from Medin & Schafer (1978), a categorical learning with stimuli that vary binarily along 4 dimensions. I represented observations and states in two ways—1) as `1000` and `A`, 2) as `1 0 0 0` and `A A A A`. In both representations, 1 in each vector represents true for the corresponding binary feature and 0 represents false. I separated the features in the second type of formatting.
- **Objective:** Given each observation, accurately predict the states they are emitted by. (For Medin & Schafer, just to see what it does.)
- **Model description:** An HMM is a supervised learning algorithm that takes a sequence of observations and predicts the states that emits each observation. For example, it can take a sentence and predict what part of speech each constituent word is.
    - **Training:** The model is trained on a set of observations and a corresponding set of states. It calculates two things—**transition probabilities** (log likelihood `∀ A ∈ STATES, B ∈ STATES: B follows A`) and **emission probabilities** (log likelihood `∀ A ∈ STATES, B ∈ OBSERVATIONS: B is emitted by A` (i.e. observation B should be tagged with state A)). For both probabilities, I first made a count matrix, then applied smoothing with constant 0.01, then finally took the logs of each entry over the sum of the row.
    - **Decoding:** Probability matrices are not enough to infer hidden states. To do so, I used the **Viterbi algorithm**, which is a dynamic programming algorithm that returns the state for each observation with the highest likelihood. First I filled in a `delta` table of dimension number of states by length of given observation and a `psi` table which holds backpointers for the most likely previous state for each state. Then I used backtracking to find the states sequence.
- **Metric:** What percent of the inferred states are accurate? (Again, only applicable for POS tagging.)
- **Discussion:** The HMM is decent at POS tagging, reaching 94.3342% accuracy. See following sections for more details.

### POS tagging
I trained the model on `/inputs/brown-train-tags.txt` and `/inputs/brown-train-sentences.txt` and tested it on `/inputs/brown-test-sentences.txt`. The results are stored in `/inputs/brown-test-tags-output.txt` and compared against `/inputs/brown-test-tags.txt`

Here's a screenshot of the terminal—
![](/img/pos-terminal.png)

The command line sentence is parsed accurately for the most part. Please note, however, that the `!` is incorrectly parsed as a `.`.

Some of the inaccuracies in tagging are results of some punctuation being tagged as other punctuation (see above). This should not be happening, as exclaimation points and periods should not be treated as states and are in fact punctuation instead of parts of speech. I had to treat punctuation as parts of speech given the way some punctuation in the training sentences are actually given POS tags in the training tags. 

However, one way I could improve model performance is finding some way to treat punctuation separately from tags. This could be as easy as in the training state separating punctuation from words, which I did try to implement but wasn't able to debug before the deadline.

### Categorical learning
Medin & Schafer (1978) (henceforth abbreviated MS78) is a categorical learning study where participants are presented with 9 stimuli that vary binary along 4 binary features (in my study encoded as either a `0` or a `1`), each stimulus labeled either category A or category B. They are then presented with 7 stimuli and tested on how they classify the stimuli. Most notably it shows **exemplar-based** instead of **rule-based** learning (kind of in line with Hopfield networks!—wonder how *that* would perform on this study), wherein participants have an exemplar for each category and assign category to new stimuli based on similarity to exemplars instead of going by a set of rules.

Technically this study isn't the best candidate for an HMM, which requires transition probabilities whereas the order of stimuli presented in this study don't matter. However, how my model performs when fed non-feature separated data (e.g. `1000`, `A` vs. feature-separated data (e.g. `1 0 0 0`, `A A A A`) does demonstrate properties of HMMs.)

**Training:**
| Stimulus | Category |
| ---      | ---      |
| 1000     | A        |
| 1010     | A        |
| 0100     | A        |
| 0010     | A        |
| 0001     | A        |
| 1100     | B        |
| 1001     | B        |
| 0111     | B        |
| 1111     | B        |

**Transfer:** Note that I included a table with human performances, where the category is the one most participants chose and the number indicates the percentage of participants who chose that category.
| Stimulus | No feature separation | Feature separation | Human |
| ---      | ---      | ---      | ---      |
| 0110     | A        | A        | A (0.59)       |
| 1110     | A        | B        | A (0.94)       |
| 0000     | A        | A        | ? (0.5)       |
| 1101     | A        | B        | A (0.62)        |
| 0101     | A        | A        | B (0.69)       |
| 0011     | A        | A        | B (0.66)       |
| 1011     | A        | B        | B (0.84)       |

It is in hindsight unsurprising that no feature separation should lead to A—the HMM would just consistently assign a random state for a before-unseen stimulus, and given the absence of transition probabilities the same random state (A or B) will always receive the best score. I find it really interesting how separating the features leads to results that at first glance look similar ish to the human ones. However, this proved untrue—the HMM assigned the second transfer stimulus, which 94% of human participates assigned to category A, to category B. The surface-level similarity is just *not* assigning the same category to each stimulus. Ultimately, the HMM looks at the sequence of features (e.g. how likely each feature is to follow the previous) whereas the human looks at how the combination of these specific features in these specific slots resembles category exemplars.

It's interesting how the model predicts *both* A and B as category labels in the feature-separated portion. This, though, can be explained through how now all the observations are either `1` and `0` which have been seen before, instead of `0000` which is a previously unseen sequence. Therefore the HMM need not stick to an arbitrary pick between A and B, and can instead assign probabilities based on sequences and transition probabilities. Entirely different method of "thinking" than human cognition!

## Future direction
- Look over the papers Eun Tack sent me & implement these models and test them on categorical learning, possibly over winter break. I'm especially interested in how discrimination-based vs. distanced-based models perform on various categorical learning tasks, and how each compares to human categorical learning.
- Will implement the HMM in Rust because I want to learn Rust.