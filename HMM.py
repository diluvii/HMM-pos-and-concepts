from pathlib import Path
import numpy as np
from Spinner import Spinner


class HMM:
    def __init__(self, states_file, observations_file, smoothing=0.01):
        self.states_file = states_file
        self.observations_file = observations_file
        self.states = {}         # name -> index
        self.index2state = []    # index -> name (filled in train)
        self.observations = {}   # obs -> index
        self.smoothing = smoothing

        # build log likelihood probability matrices during training
        self.transition_prob = None
        self.emission_prob = None
        self.unk_emissions = None

    def train(self):
        spinner = Spinner('training')
        spinner.start()

        # 1) collect states (keep everything as states as you requested)
        states = {'#'}
        with open(self.states_file, 'r') as st_reader:
            for line in st_reader:
                for s in line.split():
                    states.add(s)
        states = sorted(states)
        self.states = {name: idx for idx, name in enumerate(states)}
        # build reverse mapping list for stable index -> name access
        self.index2state = [None] * len(self.states)
        for name, idx in self.states.items():
            self.index2state[idx] = name

        # 2) collect observations (everything, lowercase)
        observations = set()
        with open(self.observations_file, 'r') as obs_reader:
            for line in obs_reader:
                for o in line.split():
                    observations.add(o.lower())
        observations = sorted(observations)
        self.observations = {o: i for i, o in enumerate(observations)}
        # add unknown observation column
        if "<UNK>" not in self.observations:
            self.observations["<UNK>"] = len(self.observations)

        # 3) build matrices
        self.build_tprob()
        self.build_eprob()

        spinner.stop()

    def build_tprob(self):
        S = len(self.states)
        tcounts = np.zeros((S, S), dtype=float)

        with open(self.states_file, 'r') as st_reader:
            for line in st_reader:
                st = ['#'] + line.split()
                for i in range(len(st) - 1):
                    curr_idx = self.states[st[i]]
                    next_idx = self.states[st[i + 1]]
                    tcounts[curr_idx, next_idx] += 1

        self.transition_prob = self.to_log(tcounts)

    def build_eprob(self):
        S = len(self.states)
        O = len(self.observations)
        ecounts = np.zeros((S, O), dtype=float)

        with open(self.states_file, 'r') as st_reader, open(self.observations_file, 'r') as obs_reader:
            for st_line, obs_line in zip(st_reader, obs_reader):
                st_tokens = st_line.split()
                obs_tokens = obs_line.split()

                # if lengths mismatch, still iterate up to min — safer than crashing
                L = min(len(st_tokens), len(obs_tokens))
                for i in range(L):
                    sname = st_tokens[i]
                    oname = obs_tokens[i].lower()
                    sidx = self.states[sname]
                    oidx = self.observations.get(oname, self.observations["<UNK>"])
                    ecounts[sidx, oidx] += 1

        # give the <UNK> column a small count for every state (so log-probs are finite)
        ecounts[:, self.observations["<UNK>"]] += self.smoothing

        self.emission_prob = self.to_log(ecounts)

        # store per-state log-prob for unknown emission lookups (fast)
        # compute as log of the (smoothed) '<UNK>' column
        self.unk_emissions = self.emission_prob[:, self.observations["<UNK>"]]

    def to_log(self, counts):
        # convert counts -> probabilities with smoothing already applied
        counts = counts + self.smoothing
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / row_sums
        return np.log(probs)

    def training_check(self):
        print("states:", self.states)
        print("observations:", list(self.observations.keys()))
        # print(self.transition_prob)
        # print(self.emission_prob)

    ################################################################################
    ## VITERBI
    ################################################################################
    def viterbi(self, obs_input, return_obs=False):
        # 0) format obs tokens (keep tokens as-is, lowercase for lookups)
        raw_tokens = obs_input.split()
        tokens = [t.lower() for t in raw_tokens]  # we store observations in lowercase

        T = len(tokens)
        S = len(self.states)

        # DP tables (log-space)
        delta = np.full((T, S), -np.inf, dtype=float)
        psi = np.full((T, S), -1, dtype=int)

        start_idx = self.states["#"]

        # init first step (do not set entry for '#' itself)
        for next_idx in range(S):
            if self.index2state[next_idx] == "#":
                continue
            obs_idx = self.observations.get(tokens[0], self.observations["<UNK>"])
            tprob = self.transition_prob[start_idx, next_idx]
            eprob = self.emission_prob[next_idx, obs_idx]
            delta[0, next_idx] = tprob + eprob
            psi[0, next_idx] = start_idx

        # recursion
        for t in range(1, T):
            obs_idx = self.observations.get(tokens[t], self.observations["<UNK>"])
            for next_idx in range(S):
                if self.index2state[next_idx] == "#":
                    continue

                # find best previous state index (skip '#')
                best_score = -np.inf
                best_prev = -1
                for curr_idx in range(S):
                    if self.index2state[curr_idx] == "#":
                        continue
                    # transition from curr -> next
                    score = delta[t-1, curr_idx] + self.transition_prob[curr_idx, next_idx]
                    if score > best_score:
                        best_score = score
                        best_prev = curr_idx

                if best_prev >= 0:
                    delta[t, next_idx] = best_score + self.emission_prob[next_idx, obs_idx]
                    psi[t, next_idx] = best_prev
                else:
                    # stays -inf if no valid prev
                    delta[t, next_idx] = -np.inf
                    psi[t, next_idx] = -1

        # backtrack
        last = int(np.argmax(delta[T-1]))
        seq_idx = [last]
        curr = last
        for t in range(T-1, 0, -1):
            curr = int(psi[t, curr])
            if curr == -1:
                # if something went wrong, fill rest with '#' as fallback
                curr = start_idx
            seq_idx.append(curr)
        seq_idx.reverse()

        # decode to state names
        decoded = [self.index2state[i] for i in seq_idx]

        if return_obs:
            # return decoded tags and the original raw input (so format_return can use it)
            return decoded, obs_input

        return decoded

    def test_on(self, file, key_file):
        base = Path(__file__).parent
        output_name = str(key_file)[:-4] + '-output'
        output_name += '.txt'
        output_file = Path(base / "inputs" / output_name)
        # run viterbi on every line
        spinner = Spinner(f'testing on given file')
        spinner.start()
        with open(file, 'r') as reader, open(output_file, 'w') as writer:
            for line in reader:
                st, obs = self.viterbi(line, return_obs=True)
                writer.write(self.format_return(obs, st))
                writer.write('\n')
        spinner.stop()
        # compare against key file
        spinner = Spinner('comparing against key file')
        spinner.start()
        total = 0
        wrong = 0
        with open(output_file, 'r') as output_reader, open(key_file, 'r') as key_reader:
            # check each line
            for output_line, key_line in zip(output_reader, key_reader):
                output = output_line.split()
                key = key_line.split()

                # check each state/observation pair
                for i in range(len(output)):
                    total += 1
                    if output[i] != key[i]:
                        wrong += 1
        
        spinner.stop()
        
        accuracy = ((total - wrong) / total) * 100
        print(f'accuracy: {accuracy:.4f}')

    def cli(self):
        print("enter sentences here to test POS tagging accuracy!\n\tplease make sure to leave a space between punctuations and words.\n\tpress q to exit.")
        while True:
            print()
            sentence = input('your sentence here:\n')

            # quit on 'q'
            if sentence == 'q':
                break

            # don't process '' or '\n'
            if sentence == '' or sentence == '\n':
                continue

            st, obs = self.viterbi(sentence, return_obs=True)
            print(self.format_return(obs, st))

    def valid_input(self, sentence):
        # TODO: keep as you like; here we accept anything
        return True

    def format_return(self, obs, st):
        # st is a list of tags aligned to tokens in obs
        return " ".join(st)
