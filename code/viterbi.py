"""
@author: Clive Gomes <cliveg@andrew.cmu.edu>
@title: 11-611 Natural Language Processing Homework #4
@description: Code to Perform POS Tagging using the Viterbi Algorithm

Base Code Provided by Instructors Alan W. Black <http://www.cs.cmu.edu/~awb/>
and David R. Mortensen <http://www.cs.cmu.edu/~dmortens/> for the Intro to NLP
(11-611) course at CMU

"""
import math
import sys
import time
import numpy as np

from collections import defaultdict

# Magic strings and numbers
HMM_FILE = sys.argv[1]
TEXT_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]
TRANSITION_TAG = "trans"
EMISSION_TAG = "emit"
OOV_WORD = "OOV" 
INIT_STATE = "init"
FINAL_STATE = "final"


class Viterbi():
    def __init__(self):
        # Transition and emission probabilities (in log scale).
        # default value = 1.0 (which would not be possible for any probability)
        self.transition = defaultdict(lambda: defaultdict(lambda: 1.0))
        self.emission = defaultdict(lambda: defaultdict(lambda: 1.0))
        
        # States to iterate over 
        self.states = set()
        self.POSStates = set()
        
        # Vocab to check for OOV words
        self.vocab = set()

        # Text to run viterbi with
        self.text_file_lines = []
        with open(TEXT_FILE, "r") as f:
            self.text_file_lines = f.readlines()

    def readModel(self):
        # Read HMM transition and emission probabilities
        # Probabilities are converted into Log Space
        with open(HMM_FILE, "r") as f:
            for line in f:
                line = line.split()

                # Read transition
                # Example line: trans NN NNPS 9.026968067100463e-05
                # Read in states as prev_state -> state
                if line[0] == TRANSITION_TAG:
                    (prev_state, state, trans_prob) = line[1:4]
                    self.transition[prev_state][state] = math.log(float(trans_prob))
                    self.states.add(prev_state)
                    self.states.add(state)

                # Read in states as state -> word
                elif line[0] == EMISSION_TAG:
                    (state, word, emit_prob) = line[1:4]
                    self.emission[state][word] = math.log(float(emit_prob))
                    self.states.add(state)
                    self.vocab.add(word)

        # Keep track of the non-initial and non-final states
        self.POSStates = self.states.copy()
        self.POSStates.remove(INIT_STATE)
        self.POSStates.remove(FINAL_STATE)


    # Run Viterbi algorithm and write the output to the output file
    def runViterbi(self):
        result = []
        
        for line in self.text_file_lines:
            result.append(self.viterbiLine(line))

        # Print output to file
        with open(OUTPUT_FILE, "w") as f:
            for line in result:
                f.write(line)
                f.write("\n")

    ## Perform Viterbi Algorithm on one line
    ## (Implemented by the Student)
    #
    # 1. If transition or emmition probability does not exist, the weight is 
    #    set to -inf; similarly, backpointer is set to index -1.
    # 2. If all previous states are "unseen" (-inf/-1), return empty string
    # 
    # Formulae:
    #     i --> word index
    #     j --> current tag
    #     k --> previous tag
    #     A --> transition matrix (2-level dictionary)
    #     B --> emission matrix (2-level dictionary)
    #     w --> Viterbi weights (log scale)
    #     b --> Viterbi backpointers (tag index)
    #     x --> word
    #
    #     w[i, j] = max(log(B[j, x[i]]) + log(A[k, j]) + w[i-1, k])
    #     b[i, j] = argmax(log(B[j, x[i]]) + log(A[k, j]) + w[i-1, k])
    ########################################################################
    def viterbiLine(self, line):
        words = line.split()
        
        # Initialize DP matrix for Viterbi here
        tags = list(self.emission.keys())
        w = np.zeros([len(words)+2, len(tags)], dtype=float)
        b = np.zeros([len(words)+2, len(tags)], dtype=int)
        best_path = []
            
        for (i, word) in enumerate(words):
            
            # Replace unseen words with OOV
            if word not in self.vocab:
                word = OOV_WORD

            # Fill weights matrix
            for (j, tag) in enumerate(tags):
                probs = []
                tagIDs = []
                for (k, prev_tag) in enumerate(tags):
                    if self.emission[tag][word] != 1.0 and self.transition[prev_tag][tag] != 1.0:
                        probs.append(self.emission[tag][word] + self.transition[prev_tag][tag] + w[i, k])
                        tagIDs.append(k)
                 
                if len(probs) == 0:
                    w[i+1, j] = -np.inf
                    b[i+1, j] = -1
                else: 
                    w[i+1, j] = np.max(probs)
                    b[i+1, j] = tagIDs[np.argmax(probs)]
            
        # Final state
        for (j, tag) in enumerate(tags):
            probs = np.zeros(len(tags))
            for (k, prev_tag) in enumerate(tags):
                probs[k] += self.emission[tag][FINAL_STATE] + self.transition[prev_tag][tag] + w[-2, k]
            w[-1, j] = probs.max()
            b[-1, j] = probs.argmax()

        
        # Backtrack & find best path
        for x in range(len(w)-1, 1, -1):
            tagID = w[x].argmax()
            if tagID == -1:
                return ''
            
            best_path.append(tags[b[x, tagID]])

        # Return string of best tag sequence
        best_path.reverse()
        return ' '.join(best_path)

# Main function        
if __name__ == "__main__":
    # Mark start time
    t0 = time.time()
    viterbi = Viterbi()
    viterbi.readModel()
    viterbi.runViterbi()
    # Mark end time
    t1 = time.time()
    print("Time taken to run: {}".format(t1 - t0))

