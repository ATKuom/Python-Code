#
from collections import Counter
from designs import arr_expert
from str_creation_equalprob import BASIC_LIST
import numpy as np
import pandas as pd


def basic_structure(sequence):
    """
    Checking if all the units of a Brayton Cycle present and properly sequenced.
    T->A->C->H
    Input: String representation of a layout
    Return: True or False
    """
    starting_index = 0
    for elements in BASIC_LIST:
        starting_index = sequence.find(elements, starting_index)
        if starting_index == -1:
            return False

    return True


def HX_restrictions(sequence, char_occur_dict):
    """
    Checking the naming and ordering of CO2-CO2 heat exchangers
    Input: String representation of a layout
    Return: True or False
    """
    if "a" in char_occur_dict and char_occur_dict["a"] != 2:
        return False

    if "a" not in char_occur_dict and "b" in char_occur_dict:
        return False

    if "b" in char_occur_dict and char_occur_dict["b"] != 2:
        return False

    if sequence.count("aa") or sequence.count("bb"):
        return False

    if (
        sequence.find("-1a") != -1
        and sequence.find("1a", sequence.find("-1a") + 2) != -1
    ):
        return False
    if (
        sequence.find("-1b") != -1
        and sequence.find("1a", sequence.find("-1b") + 2) != -1
    ):
        return False
    if (
        sequence.find("-2a") != -1
        and sequence.find("2a", sequence.find("-2a") + 2) != -1
    ):
        return False
    if (
        sequence.find("-2b") != -1
        and sequence.find("2b", sequence.find("-2b") + 2) != -1
    ):
        return False

    return True


def splitter_restrictions(sequence, char_occur_dict):
    if "1" in char_occur_dict and char_occur_dict["1"] != 3:
        return False

    if "1" not in char_occur_dict and "2" in char_occur_dict:
        return False

    if "2" in char_occur_dict and char_occur_dict["2"] != 3:
        return False

    if (
        sequence.count("11")
        or sequence.count("1-1")
        or sequence.count("22")
        or sequence.count("2-2")
    ):
        return False

    if sequence.count("-1") == 2 or sequence.count("-2") == 2:
        return False

    return True


def validity(datalist):
    valid_strings = []
    for sequence in datalist:
        char_occur_dict = Counter(sequence)
        if (
            basic_structure(sequence)
            and HX_restrictions(sequence, char_occur_dict)
            and splitter_restrictions(sequence, char_occur_dict)
        ):
            valid_strings.append(sequence)

    return valid_strings


if __name__ == "__main__":
    # datalist = expert_designs
    # datalist = np.load("randomgstrings.npy")
    datalist = np.load("semirandomgstrings.npy")
    valid_strings = np.unique(np.array(validity(datalist)))
    print(len(valid_strings))
    DF = pd.DataFrame(valid_strings)
    DF.to_csv("valid_semirandom_strings.csv")
    # valid_strings = np.append(valid_strings, arr_expert)
    # print(valid_strings)
    # print(len(valid_strings))
    # np.save('D0test.npy',valid_strings)
