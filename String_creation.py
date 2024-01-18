# Random string creation as a representation of process
# No restrictions just pick a letter for each avaiable space from the alphabet
import numpy as np
import random as random
from itertools import repeat
import config
from thermo_validity import validity

RESTRICTED_EQUIP = [
    "a",
    "a",
    # "b",
    "1",
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
        if new_word[i] in RESTRICTED_EQUIP:
            equip_str.remove(new_word[i])
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
    # randomly_generated_strings = np.array(numberofstrings(20000), dtype=object)
    # print(
    #     "Longest string character length:",
    #     len(max(randomly_generated_strings, key=len)),
    #     max(randomly_generated_strings, key=len),
    # )
    # randomly_generated_strings = [
    #     "G" + string + "E" for string in randomly_generated_strings
    # ]
    # print(len(validity(randomly_generated_strings)))
    N = 50000
    randomly_generated_strings = []
    i = 0
    while len(randomly_generated_strings) < N:
        string = "G" + word_creation() + "E"
        i += 1
        if validity([string]):
            randomly_generated_strings.append(string)
        randomly_generated_strings = np.unique(
            np.array(randomly_generated_strings, dtype=object)
        )
        randomly_generated_strings = list(randomly_generated_strings)
    print(i, len(randomly_generated_strings), randomly_generated_strings[:10])
    randomly_generated_strings = np.array(randomly_generated_strings, dtype=object)
    print(
        "Longest string character length:",
        len(max(randomly_generated_strings, key=len)),
        max(randomly_generated_strings, key=len),
    )
    j, k, l = 0, 0, 0
    for string in randomly_generated_strings:
        if "a" in string:
            j += 1
            if "-1" in string:
                l += 1
        if "-1" in string:
            k += 1
    print(j, k, l)
    np.save(config.DATA_DIRECTORY / "v8D0_m1.npy", randomly_generated_strings)
