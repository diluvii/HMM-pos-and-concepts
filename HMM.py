# Author: Yawen Xue
# Date: 01 Nov. 2025
# Hidden Markov Model (HMM) class for COGS 21 final project
import numpy as np


# Feed it a file of observations and a file of corresponding states!
class HMM:
    def __init__(self, states_file, observations_file, start_state=None, smoothing=0.01):
        self.states_file = states_file
        self.observations_file = observations_file
        self.states = {}
        self.observations = {}

        self.start_state = start_state
        self.smoothing = smoothing

        # During training we'll build transition & emission probabilities from states & observations
        ## Transition probability: Probability that state A is followed by each of the states
        ## Emission probability: Probability that observation A is emitted by each of the states
        self.transition_prob = []
        self.emission_prob = []
    
    def train(self):
        # First build list of states & observations
        states_reader = open(self.states_file, "r")
        observations_reader = open(self.observations_file, "r")

        # Create set of states & map each onto a number for the matrix
        states = set()
        for line in states_reader:
            curr_states = line.split()
            for st in curr_states:
                states.add(st)
        states = list(states)                   
        states.sort()

        # Make start state the first state
        if self.start_state:
            states.insert(0, self.start_state)
        
        states = list(enumerate(states))
        for st in states:
            self.states[st[1]] = st[0]

        # Do the same for observations
        observations = set()
        for line in observations_reader:
            curr_observations = line.split()
            for obs in curr_observations:
                observations.add(obs.lower().strip())
        observations = list(observations)
        observations.sort()
        observations = list(enumerate(observations))
        for obs in observations:
            self.observations[obs[1]] = obs[0]

        # Then build transition & emission probabilities
        self.transition_prob = self.build_tprob()
        self.emission_prob = self.build_eprob()

        states_reader.close()
        observations_reader.close()

    '''
    Build transition probabilities
        Which are the probabilities of moving between pairs of states
    This will return a dictionary of dictionaries like so—
        Each state is mapped to a dictionary
        Each dictionary contains a mapping of all states to transition probabilities
    '''
    def build_tprob(self):
        tprob_counts = np.zeros((len(self.states), len(self.states)))

        # Count transitions per line in states file
        states_reader = open(self.states_file, "r")
        for line in states_reader:
            curr_states = line.split()
            i = 0
            while i < len(curr_states) - 1:
                curr = curr_states[i]
                next = curr_states[i + 1]

                # Increment counts in matrix
                tprob_counts[self.states[curr]][self.states[next]] += 1
                i += 1

        states_reader.close()
        return self.to_log(tprob_counts)
    
    '''
    Build emission probabilities
        That is, for example, each time you see "will" how likely is it that it's a noun/verb/etc etc?
    '''
    def build_eprob(self):
        eprob_counts = np.zeros((len(self.states), len(self.observations)))

        # Count occurences of each state for each observation
        with open(self.states_file, "r") as states_reader, open(self.observations_file, "r") as observations_reader:
            for line_st, line_obs in zip(states_reader, observations_reader):
                curr_st = line_st.split()
                curr_obs = line_obs.split()

                # Sanity check they're actually paired
                if len(curr_st) != len(curr_obs):
                    raise ValueError("Mismatched length between state and observations files")
                
                # Now count!
                for st, obs in zip(curr_st, curr_obs):
                    obs = obs.lower().strip()
                    eprob_counts[self.states[st]][self.observations[obs]] += 1

        return self.to_log(eprob_counts)
    
    '''
    Helper function to actually make transition probabilities properly formatted
        (Currently it consists of counts instead of actual probabilities)
    Takes in transition counts (np array) and spits out transition log probabilities
    '''
    def to_log(self, counts):
        n = len(counts)
        m = len(counts[0])
        row_sums = []

        # Apply smoothing (we don't want to log(0) to each count), & sum counts from each row
        for i in range(n):
            row_sum = 0
            for j in range(m):
                counts[i][j] += self.smoothing
                row_sum += counts[i][j]
            row_sums.append(row_sum)
        
        # Now do a second pass over the counts matrix & turn each into log(count/row sum)
        for i in range(n):
            row_sum = row_sums[i]
            for j in range(m):
                counts[i][j] = np.log(counts[i][j] / row_sum)

        return counts
    
    '''
    Sanity check for training
    '''
    def training_check(self, print_matrix=False):
        print("States:", self.states)
        print("Observations:", self.observations)
        print("Transition matrix:", self.transition_prob.shape)
        if print_matrix: print("\t", self.transition_prob)
        print("Emission matrix:", self.emission_prob.shape)
        if print_matrix: print("\t", self.emission_prob)
