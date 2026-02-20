import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p = prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        n_states = len(self.hidden_states)
        assert self.prior_p.shape == (n_states,)
        assert self.transition_p.shape == (n_states, n_states)
        assert self.emission_p.shape == (len(self.observation_states), n_states)


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence
        """


        # Initialize variables
        state_dict = self.observation_states_dict

        A = self.transition_p

        # Edge case check for if the input states are of length 0
        if len(input_observation_states) == 0:
            return 1.0

        #initialize probability variable
        probability_val = None

        #Enumerate through the states
        for i, entry in enumerate(input_observation_states):
            idx_occurrence = state_dict[entry]
            #Find the probabilities associated with the current state.
            probabilities = self.emission_p[idx_occurrence, :]
            if i == 0:
                probability_val = self.prior_p * probabilities
            else:
                probability_val = (probability_val @ A) * probabilities

        return float(np.sum(probability_val))

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """

        state_dict = self.observation_states_dict
        A = self.transition_p

        # Edge case check for if the input states are of length 0
        if len(decode_observation_states) == 0:
            return []

        #initialize probability variable and backpointers
        probability_val = None
        backpointers = []

        #Enumerate through the states
        for i, entry in enumerate(decode_observation_states):
            idx_occurrence = state_dict[entry]
            #Find the probabilities associated with the current state
            probabilities = self.emission_p[idx_occurrence, :]
            if i == 0:
                probability_val = self.prior_p * probabilities
                backpointers.append(np.zeros_like(probability_val, dtype=int))
            else:
                scores = probability_val[:, None] * A
                backpointers.append(np.argmax(scores, axis=0))
                probability_val = np.max(scores, axis=0) * probabilities

        #Pick best final state then backtrack
        last_state = int(np.argmax(probability_val))
        best_path = [last_state]
        for i in range(len(decode_observation_states) - 1, 0, -1):
            last_state = int(backpointers[i][last_state])
            best_path.append(last_state)

        best_path.reverse()
        return best_path