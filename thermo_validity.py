#
from collections import Counter
from designs import goeos_expert
import numpy as np
import config

BASIC_LIST = ["T", "A", "C", "H"]
classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
char_to_int = dict((c, i) for i, c in enumerate(classes))


def string_to_equipment(sequence, char_to_int=char_to_int, classes=classes):
    equipment = []
    splitter = False
    for i, char in list(enumerate(sequence)):
        try:
            if splitter == True:
                splitter = False
                continue
            equipment.append(char_to_int[char])
        except:
            equipment.append(char_to_int[char + sequence[i + 1]])
            splitter = True
    return np.array(equipment)


def basic_structure(sequence, char_occur_dict):
    """
    Checking if all the units of a Brayton Cycle present and properly sequenced.
    T->A->C->H
    Input: String representation of a layout
    Return: True or False
    """
    if "G" in char_occur_dict.keys():
        if char_occur_dict["G"] != 1:
            return False
    if "E" in char_occur_dict.keys():
        if char_occur_dict["E"] != 1:
            return False
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

    if sequence[-2] == "a" or sequence[-2] == "b":
        if sequence[1] == "a" or sequence[1] == "b":
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
    if "1" in char_occur_dict and sequence.count("-1") != 1:
        return False

    if "1" not in char_occur_dict and "2" in char_occur_dict:
        return False

    if "2" in char_occur_dict and char_occur_dict["2"] != 3:
        return False
    if "2" in char_occur_dict and sequence.count("-2") != 1:
        return False

    if (
        sequence.count("11")
        or sequence.count("-11")
        or sequence.count("22")
        or sequence.count("-22")
    ):
        return False

    if sequence.count("-1") > 1 or sequence.count("-2") > 1:
        return False
    return True


def special_rules(sequence):
    equipment = string_to_equipment(sequence)
    if len(equipment) > 22:
        return False
    for current, next in zip(equipment, np.roll(equipment, -1)):
        if current == next:
            return False
    if 9 in equipment:
        splitter = np.where(equipment == 9)[0][0]
        equipment = np.roll(equipment, -splitter)
        splitter = 0
        m1, m2 = np.where(equipment == 7)[0]
        turbine_positions = np.where(equipment == 1)[0]
        compressor_positions = np.where(equipment == 3)[0]
        if m1 == len(equipment) - 1 or m2 == len(equipment) - 1:
            return False
        if np.any(turbine_positions < splitter) or np.any(turbine_positions > m2):
            return True
        if np.any(compressor_positions < splitter) or np.any(compressor_positions > m2):
            return True
        if np.any(turbine_positions > m1) or np.any(compressor_positions > m1):
            if np.any(turbine_positions < m1) or np.any(compressor_positions < m1):
                return True
        return False
    return True


def validity(datalist):
    valid_strings = []
    for sequence in datalist:
        char_occur_dict = Counter(sequence)
        if (
            basic_structure(sequence, char_occur_dict)
            and HX_restrictions(sequence, char_occur_dict)
            and splitter_restrictions(sequence, char_occur_dict)
            and special_rules(sequence)
        ):
            valid_strings.append(sequence)

    return valid_strings


if __name__ == "__main__":
    # from empty import layout_to_string
    # layouts = np.load(config.DATA_DIRECTORY / "broken_layouts.npy", allow_pickle=True)
    # datalist = layout_to_string(layouts)
    # print(len(datalist))
    datalist = np.load(config.DATA_DIRECTORY / "v810k.npy", allow_pickle=True)
    # datalist = ["GTACHE", "GTCAHE", "GTTCAHE", "GHTCAHE", "GaTCAHaE"]
    # datalist = goeos_expert
    print(len(datalist), len(validity(datalist)))
    valid_strings = np.unique(np.array(validity(datalist), dtype=object))
    print(len(valid_strings), valid_strings)
    # np.save(config.DATA_DIRECTORY / "v5m2D0_candidates.npy", valid_strings)
    p_datalist = np.load(config.DATA_DIRECTORY / "v8D0_m1.npy", allow_pickle=True)
    # print(len(p_datalist))
    n_datalist = np.concatenate((p_datalist, valid_strings), axis=0)
    n_valid_strings = np.unique(n_datalist)
    print(len(n_valid_strings))
    # # np.save(config.DATA_DIRECTORY / "v3DF_m1.npy", n_valid_strings)
    index = np.where(np.isin(n_valid_strings, p_datalist, invert=True))[0]
    new_ones = n_valid_strings[index]
    print(len(new_ones))
    # np.save(config.DATA_DIRECTORY / "v81k_QA.npy", new_ones)
