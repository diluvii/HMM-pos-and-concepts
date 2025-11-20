# Author: Yawen Xue
# Date: 01 Nov. 2025 - 20 Nov. 2025
# Hidden Markov Model (HMM) class for COGS 21 final project
import numpy as np
from Spinner import Spinner


class HMM:
    def __init__(self, states_file, observations_file, smoothing=0.01):
        self.states_file = states_file
        self.observations_file = observations_file
        self.states = {}
        self.observations = {}
        self.smoothing = smoothing

        # build log likelihood probability matrices during training
        self.transition_prob = None
        self.emission_prob = None
    
    def train(self):
        # get list of states
        spinner = Spinner()
        spinner.start()

        states = {'#'}
        with open(self.states_file, 'r') as st_reader:
            for line in st_reader:
                st = line.split()
                for s in st:
                    if s.isalpha():
                        states.add(s)
        states = list(states)
        states.sort()
        states = enumerate(states)
        for s in states:
            self.states[s[1]] = s[0]

        # get list of observations
        observations = set()
        with open(self.observations_file, 'r') as obs_reader:
            for line in obs_reader:
                obs = line.split()
                for o in obs:
                    if o.isalpha():
                        observations.add(o.lower())
        observations = list(observations)
        observations.sort()
        observations = enumerate(observations)
        for o in observations:
            self.observations[o[1]] = o[0]

        self.build_tprob()
        self.build_eprob()

        spinner.stop()
    
    def build_tprob(self):
        tcounts = np.zeros((len(self.states), len(self.states)))

        # loop through lines in states file
        with open(self.states_file, 'r') as st_reader:
            for line in st_reader:
                st = line.split()
                st = ['#'] + [s for s in st if s.isalpha()]

                # count transitions
                for i in range(0, len(st) - 1):
                    curr = self.states[st[i]]      # chase down index
                    next = self.states[st[i + 1]]  # chase down index
                    tcounts[curr][next] += 1        # increment count for this transition
        
        self.transition_prob = self.to_log(tcounts)

    def build_eprob(self):
        ecounts = np.zeros((len(self.states), len(self.observations)))
        
        # loop through /pairs/ in states & observations file
        with open(self.states_file, 'r') as st_reader, open(self.observations_file, 'r') as obs_reader:
            for st_line, obs_line in zip(st_reader, obs_reader):
                st = st_line.split()
                st = [s for s in st if s.isalpha()]
                obs = obs_line.split()
                obs = [o for o in obs if o.isalpha()]

                # count emissions
                for i in range(len(st)):
                    ecounts[self.states[st[i]]][self.observations[obs[i]]] += 1

        self.emission_prob = self.to_log(ecounts)

    def to_log(self, counts):
        n = len(counts)
        m = len(counts[0])
        row_sums = []

        # apply smoothing (we don't want to log(0) to each count), & sum counts from each row
        for i in range(n):
            row_sum = 0
            for j in range(m):
                counts[i][j] += self.smoothing
                row_sum += counts[i][j]
            row_sums.append(row_sum)
        
        # now do a second pass over the counts matrix & turn each into log(count/row sum)
        for i in range(n):
            row_sum = row_sums[i]
            for j in range(m):
                counts[i][j] = np.log(counts[i][j] / row_sum)

        return counts
    
    def training_check(self):
        # sanity checks!
        print(self.states)
        print(self.observations)
        print(self.transition_prob)
        print(self.emission_prob)
    
    ################################################################################
    ## now training is done, & we have emission & transition probability matrices ##
    ## next steps are for viterbi decoding & testing                              ##
    ################################################################################
    def viterbi(self, obs):
        # 0) format obs data
        obs = obs.split()
        obs = [o.lower() for o in obs if o.isalpha()]

        # 1) init tables for storing best candidates & backpointers
        return obs
    
    def test_on(self, file):
        # run viterbi on every line
        with open(file, 'r') as reader:
            for line in reader:
                print(self.viterbi(line))
    
    def cli(self):
        print("enter sentences here to test POS tagging accuracy!\n\tplease make sure to leave a space between punctuations and words.\n\tpress q to exit.")
        while True:
            print()
            sentence = input('your sentence here:\n')

            # quit on 'q'
            if sentence == 'q':
                break

            # check if sentence is valid; if not, ask for next input
            if not self.valid_input(sentence):
                print("\033[35minvalid input format. please make sure to leave spaces between words and punctuation!\033[0m")
                continue
            print(self.viterbi(sentence))
    
    def valid_input(self, sentence):
        for word in sentence.split():
            
            # determine if this word starts with letters
            alpha = word[0].isalpha()
            for s in word:

                # if word starts with letters but ends with non-letters... invalid
                if s.isalpha() != alpha:
                    return False
                
        return True
