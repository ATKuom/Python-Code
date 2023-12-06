import numpy as np
import config
from designs import expert_designs


def splthxnmbrcheck(datalist):
    splitter_number = 0
    HX_number = 0
    splt_hx_number = 0
    expertdesign_number = 0
    present_ed = list()
    for seq in datalist:
        if "-1" in seq:
            splitter_number += 1
        if "a" in seq:
            HX_number += 1
        if "a" in seq and "-1" in seq:
            splt_hx_number += 1
        for expert in expert_designs:
            if seq == expert:
                expertdesign_number += 1
                present_ed.append(seq)

    return (
        len(datalist),
        splitter_number,
        HX_number,
        splt_hx_number,
        expertdesign_number,
        present_ed,
    )


if __name__ == "__main__":
    # datalist = arr_expert
    datalist = np.load(config.DATA_DIRECTORY / "len20d10.npy", allow_pickle=True)
    total, splitter, hx, both, ed, edlist = splthxnmbrcheck(datalist)
    print(total, splitter, hx, both, ed, edlist)
