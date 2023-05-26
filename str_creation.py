# Random string creation as a representation of process
# There are some restrictions on the generation of words such as you must include as pairs for heat exc and mixer/splitter

import numpy as np
import random as random
from itertools import repeat


RESTRICTED_EQUIP = ["a", "a", "b", "b", "1", "2", "-1", "-2"]
BASIC_LIST = ["T", "A", "C", "H"]


def brayton_cycle(word_length, new_word, indexes_new_word):
    """
    Initialize the standard brayton cycle for the new string as inputting TACH successively. Put the the equipment in random positions while staying true to the succession.
    (Check the word_creation doc for general information)
    Return:
    the modified new word
    available indexes of the word for further mutation
    """
    starting_i = 0
    for elements in BASIC_LIST:
        i = random.randint(
            starting_i, (word_length - (len(BASIC_LIST) - BASIC_LIST.index(elements)))
        )
        new_word[i] = elements
        indexes_new_word.remove(i)
        starting_i = i + 1

    return (new_word, indexes_new_word)


def splitter_restriction():
    pass


def restriction_rules(equip_str, new_word, indexes_new_word, i):
    """
    Applying restriction rules of Splitter, Mixer and Heat Exchangers
    The numerical limitation for these equipment comes from the number of apperances in the restricted equips list
    1 branch mixer requires 1 branch splitter or vice versa (1, -1) and (2, -2)
    Heat exchangers requires one input and one output (a, a) and (b, b)
    (Check the word_creation doc for general information)
    Return:
    Modified new_word
    Modified available indexes for indexes_new_word
    """
    equip_str.remove(new_word[i])
    if new_word[i] == "1":
        i = random.choice(indexes_new_word)
        new_word[i] = "-1"
        indexes_new_word.remove(i)
        equip_str.remove(new_word[i])
    elif new_word[i] == "-1":
        i = random.choice(indexes_new_word)
        new_word[i] = "1"
        indexes_new_word.remove(i)
        equip_str.remove(new_word[i])
    elif new_word[i] == "2":
        i = random.choice(indexes_new_word)
        new_word[i] = "-2"
        indexes_new_word.remove(i)
        equip_str.remove(new_word[i])
    elif new_word[i] == "-2":
        i = random.choice(indexes_new_word)
        new_word[i] = "2"
        indexes_new_word.remove(i)
        equip_str.remove(new_word[i])
    elif new_word[i] == "a":
        i = random.choice(indexes_new_word)
        new_word[i] = "a"
        indexes_new_word.remove(i)
        equip_str.remove(new_word[i])
    else:
        if "a" in new_word:
            i = random.choice(indexes_new_word)
            new_word[i] = "b"
            indexes_new_word.remove(i)
            equip_str.remove(new_word[i])
        else:
            equip_str.append(new_word[i])
            new_word[i] = "a"
            equip_str.remove(new_word[i])
            i = random.choice(indexes_new_word)
            new_word[i] = "a"
            indexes_new_word.remove(i)
            equip_str.remove(new_word[i])

    return (new_word, indexes_new_word)


def word_creation():
    """
    Process equipments are represented by respective string piece
    T = Turbine A = Cooler (Utility) C = Compressor H = Heater(Utility)
    a,a = First Heat-Exchanger input/output b,b = Second Heat exchanger input/output
    1,2 = 2 two-branch mixer
    -1,-2 = 2 two-branc splitter
    Every process layout string must include brayton cycle (TACH). They can be up to 20 equipment. Random equipment puts into random positions to generate random process layouts
    Return:
    Created random process layout as string
    """
    equip_str = RESTRICTED_EQUIP + BASIC_LIST
    word_length = np.random.randint(4, 21)
    indexes_new_word = list(range(word_length))
    new_word = [""] * word_length
    brayton_cycle(word_length, new_word, indexes_new_word)
    while new_word.count("") > 0:
        i = random.choice(indexes_new_word)
        new_word[i] = random.choice(equip_str)
        if new_word[i] not in RESTRICTED_EQUIP:
            indexes_new_word.remove(i)
        else:
            if new_word.count("") > 0:
                indexes_new_word.remove(i)
                restriction_rules(equip_str, new_word, indexes_new_word, i)
            else:
                new_word[i] = ""
    return "".join(new_word)


def numberofstrings(N):
    """
    Creating the desired number of randomly generated strings.
    Return:
    List of all generated strings
    """
    Listofstrings = []
    for _ in repeat(None, N):
        Listofstrings.append(word_creation())
    return Listofstrings


if __name__ == "__main__":
    semirandomly_generated_strings = np.array(numberofstrings(100000))
    np.save("semirandomgstrings.npy", semirandomly_generated_strings)
