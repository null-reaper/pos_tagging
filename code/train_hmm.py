"""
@author: Clive Gomes <cliveg@andrew.cmu.edu>
@title: 11-611 Natural Language Processing Homework #4
@description: Code to Train the HMM Model for POS Tagging

Base Code Provided by Instructors Alan W. Black <http://www.cs.cmu.edu/~awb/>
and David R. Mortensen <http://www.cs.cmu.edu/~dmortens/> for the Intro to NLP
(11-611) course at CMU

"""
import sys
import re

from collections import defaultdict

class HMMTrain():
    def __init__(self, TAG_FILE, TOKEN_FILE, OUTPUT_FILE):
        self.TAG_FILE = TAG_FILE
        self.TOKEN_FILE = TOKEN_FILE 
        self.OUTPUT_FILE = OUTPUT_FILE
        
        # Vocabulary
        self.vocab = {}
        self.OOV_WORD = "OOV"
        self.INIT_STATE = "init"
        self.FINAL_STATE = "final"
        
        # Transition and emission probabilities
        self.emissions = {}
        self.transitions = {}
        self.transitions_total = defaultdict(lambda: 0)
        self.emissions_total = defaultdict(lambda: 0)



    # Train the model
    def train(self):
        # Read from tag file and token file. 
        with open(self.TAG_FILE) as tag_file, open(self.TOKEN_FILE) as token_file:
            for tag_string, token_string in zip(tag_file, token_file):
                tags = re.split("\s+", tag_string.rstrip())
                tokens = re.split("\s+", token_string.rstrip())
                pairs = zip(tags, tokens)

                # Starts off with initial state
                prevtag = self.INIT_STATE

                for (tag, token) in pairs:

                    # This block is a little trick to help with out-of-vocabulary (OOV)
                    # words.  the first time we see *any* word token, we pretend it
                    # is an OOV.  this lets our model decide the rate at which new
                    # words of each POS-type should be expected (e.g., high for nouns,
                    # low for determiners).

                    if token not in self.vocab:
                        self.vocab[token] = 1
                        token = self.OOV_WORD

                    # Store occurrence counts for each dictionary 
                    # (Implememted by the student)
                    
                    if prevtag not in self.transitions:
                        self.transitions[prevtag] = defaultdict(lambda: 0)
        
                    self.transitions[prevtag][tag] += 1   
                    self.transitions_total[prevtag] += 1
                    
                    if tag not in self.emissions:
                        self.emissions[tag] = defaultdict(lambda: 0)
        
                    self.emissions[tag][token] += 1        
                    self.emissions_total[tag] += 1
                    
                    prevtag = tag


                # Stop probability for each sentence
                if prevtag not in self.transitions:
                    self.transitions[prevtag] = defaultdict(lambda: 0)

                self.transitions[prevtag][self.FINAL_STATE] += 1
                self.transitions_total[prevtag] += 1

    # Calculate the transition probability prevtag -> tag
    def calculate_transition_prob(self, prevtag, tag):
        # Get transition probabilities from counts (Implememted by the student)
        return self.transitions[prevtag][tag]/self.transitions_total[prevtag]

    # Calculate the probability of emitting token given tag
    def calculate_emission_prob(self, tag, token):
        # Get emission probabilities from counts (Implememted by the student)
        return self.emissions[tag][token]/self.emissions_total[tag]

    # Write the model to an output file.
    def writeResult(self):
        with open(self.OUTPUT_FILE, "w") as f:
            for prevtag in self.transitions:
                for tag in self.transitions[prevtag]:
                    f.write("trans {} {} {}\n"
                        .format(prevtag, tag, self.calculate_transition_prob(prevtag, tag)))

            for tag in self.emissions:
                for token in self.emissions[tag]:
                    f.write("emit {} {} {}\n"
                        .format(tag, token, self.calculate_emission_prob(tag, token)))


# Main function
if __name__ == "__main__":
    # Files
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]
    OUTPUT_FILE = sys.argv[3]

    model = HMMTrain(TAG_FILE, TOKEN_FILE, OUTPUT_FILE)
    model.train()
    model.writeResult()



