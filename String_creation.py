# Random string creation as a representation of process
# No restrictions just pick a letter for each avaiable space from the alphabet
import numpy as np
import random as random
from itertools import repeat
import config


RESTRICTED_EQUIP = [
    "a",
    "b",
    "1",
    # "2",
    "-1",
    # "-2",
    # "3", "4", "5", "-3","-4", "-5"
    # "c", "d"
]
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
        indexes_new_word.remove(i)
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
    randomly_generated_strings = np.array(numberofstrings(100), dtype=object)
    print("Longest string character length:", len(np.max(randomly_generated_strings)))
    print(len(randomly_generated_strings))
    np.save(config.DATA_DIRECTORY / "r_g_strings.npy", randomly_generated_strings)
