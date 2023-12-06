#
from collections import Counter
from designs import arr_expert
from String_creation import BASIC_LIST
import numpy as np
import config

classes = ["G", "T", "A", "C", "H", "a", "b", "1", "2", "-1", "-2", "E"]
char_to_int = dict((c, i) for i, c in enumerate(classes))


def string_to_equipment(sequence, char_to_int=char_to_int, classes=classes):
    equipment = []
    splitter = False
    for char in sequence:
        try:
            equipment.append(char_to_int[char])
        except:
            equipment.append(char_to_int["-1"])
            splitter = True
    if splitter == True:
        equipment.pop(equipment.index(9) + 1)
        splitter = False
    return np.array(equipment)


def length(sequence):
    length = 0
    for char in sequence:
        if char == "-":
            continue
        else:
            length += 1
    if length > 22:
        return False
    else:
        return True


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

    for elements in BASIC_LIST:
        index = sequence.find(elements)
        if index == -1:
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
    if sequence[-1] == "a" or sequence[-1] == "b":
        if sequence[0] == "a" or sequence[0] == "b":
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
        or sequence.count("2-2")
    ):
        return False

    if sequence.count("-1") > 1 or sequence.count("-2") > 1:
        return False
    return True


def special_rules(sequence):
    equipment = string_to_equipment(sequence)
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
            and length(sequence)
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
    datalist = np.load(
        config.DATA_DIRECTORY / "v3D0_m2_generated.npy", allow_pickle=True
    )
    print(len(datalist), len(validity(datalist)))
    valid_strings = np.unique(np.array(validity(datalist), dtype=object))
    print(len(valid_strings))
    # np.save(config.DATA_DIRECTORY / "v3m2D0_candidates.npy", valid_strings)
    p_datalist = np.load(config.DATA_DIRECTORY / "v3D0_m2.npy", allow_pickle=True)
    print(len(p_datalist))
    n_datalist = np.concatenate((p_datalist, valid_strings), axis=0)
    n_valid_strings = np.unique(n_datalist)
    print(len(n_valid_strings))
    # np.save(config.DATA_DIRECTORY / "v3DF_m1.npy", n_valid_strings)
    index = np.where(np.isin(n_valid_strings, p_datalist, invert=True))[0]
    new_ones = n_valid_strings[index]
    print(new_ones, len(new_ones))
    np.save(config.DATA_DIRECTORY / "v3D1_m2_candidates.npy", new_ones)
