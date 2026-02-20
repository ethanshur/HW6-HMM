import pytest
from hmm import HiddenMarkovModel
import numpy as np
import os


def test_mini_weather():
    """
    Create an instance of your HMM class using the "small_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    base = os.path.dirname(os.path.dirname(__file__))
    mini_hmm = np.load(os.path.join(base, "data", "mini_weather_hmm.npz"))
    mini_input = np.load(os.path.join(base, "data", "mini_weather_sequences.npz"))

    hmm = HiddenMarkovModel(
        observation_states=mini_hmm["observation_states"],
        hidden_states=mini_hmm["hidden_states"],
        prior_p=mini_hmm["prior_p"],
        transition_p=mini_hmm["transition_p"],
        emission_p=mini_hmm["emission_p"].T,
    )

    obs = mini_input["observation_state_sequence"]
    gt = list(mini_input["best_hidden_state_sequence"])

    # Check for correctness
    path_idx = hmm.viterbi(obs)
    path_labels = [hmm.hidden_states_dict[int(i)] for i in path_idx]
    assert path_labels == gt

    # Check for empty sequences
    assert hmm.forward(np.array([], dtype=obs.dtype)) == 1.0
    assert hmm.viterbi(np.array([], dtype=obs.dtype)) == []

    pass


def test_full_weather():
    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file.
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    base = os.path.dirname(os.path.dirname(__file__))
    hmm_npz = np.load(os.path.join(base, "data", "full_weather_hmm.npz"))
    seq_npz = np.load(os.path.join(base, "data", "full_weather_sequences.npz"))

    hmm = HiddenMarkovModel(
        observation_states=hmm_npz["observation_states"],
        hidden_states=hmm_npz["hidden_states"],
        prior_p=hmm_npz["prior_p"],
        transition_p=hmm_npz["transition_p"],
        emission_p=hmm_npz["emission_p"].T,
    )

    obs = seq_npz["observation_state_sequence"]
    gt = list(seq_npz["best_hidden_state_sequence"])

    path_idx = hmm.viterbi(obs)
    path_labels = [hmm.hidden_states_dict[int(i)] for i in path_idx]

    assert path_labels == gt
