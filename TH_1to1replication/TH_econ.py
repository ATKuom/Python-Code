def economics(pec, prod_capacity):
    ##Parameters
    cap_factor = 0.85
    T = 8760
    hourtosec = 3600
    costofNG = 0.00285  # $/MJ
    lvh = 50.01  # MJ/kgNG
    rnf = 0.06
    int_rate = 0.03
    equity_frac = 0.4
    debt_frac = 0.6
    equity_roi = 0.2
    debt_roi = 0.085
    wacc = equity_frac * equity_roi + debt_frac * debt_roi
    tax_rate = 0.38
    mfuel = 1.26  ##22.3  # kg/s PhDthes_103
    total_pec = sum(
        pec
    )  # + 9.721e6 + 1e5 * prod_capacity  # 9.721e6 for GT set, 1M $ for 10 MWe for piping and stuff
    annual_om = 9.1111  # $/MWhe in 2023$
    tpc = 1.242 * total_pec
    land = 0.014 * total_pec
    fci = tpc + land  # 3.5575 * total_pec

    fc = costofNG * lvh * T * cap_factor * hourtosec * mfuel
    oandm = prod_capacity * annual_om * T * cap_factor
    suc = oandm / 12 + fc / 52 + tpc * 0.02
    wc = 0.005 * tpc + oandm / 44 + fc / 12
    fnc = 0.027 * tpc
    ooc = 0.15 * tpc

    ##AFUDC calculation
    year1 = 0.4 * tpc * (1 + int_rate) ** 3
    y1_equity_afudc = equity_frac * equity_roi * year1 * 1.5
    y1_debt_afudc = debt_frac * debt_roi * year1 * 1.5
    year2 = 0.6 * tpc * (1 + int_rate) ** 4
    y2_equity_afudc = equity_frac * equity_roi * year2 * 0.5
    y2_debt_afudc = debt_frac * debt_roi * year2 * 0.5
    land_equity = equity_frac * equity_roi * land * 2
    land_debt = debt_frac * debt_roi * land * 2
    startup_equity = suc * (1 + int_rate) ** 2 * equity_frac * equity_roi * 0.5
    startup_debt = suc * (1 + int_rate) ** 2 * debt_frac * debt_roi * 0.5
    equity_afudc = y1_equity_afudc + y2_equity_afudc + land_equity + startup_equity
    debt_afudc = y1_debt_afudc + y2_debt_afudc + land_debt + startup_debt
    total_afudc = equity_afudc + debt_afudc

    pfi = year1 + year2
    toc = land + tpc + suc + wc + fnc + ooc
    tasc = toc + total_afudc
    # Escalated Amounts
    esc_land = land * (1 + int_rate) ** 2
    esc_tpc = year1 + year2
    esc_suc = suc * (1 + int_rate) ** 3
    esc_wc = wc * (1 + int_rate) ** 3
    esc_fnc = fnc * (1 + int_rate) ** 2.5
    esc_ooc = ooc * (1 + int_rate) ** 3
    esc_toc = esc_land + esc_tpc + esc_suc + esc_wc + esc_fnc + esc_ooc
    tci = esc_toc + total_afudc
    ndi = esc_land + esc_wc + equity_afudc
    tdi = tci - ndi

    macrs = [
        5,
        9.5,
        8.55,
        7.7,
        6.93,
        6.23,
        5.9,
        5.9,
        5.91,
        5.9,
        5.91,
        5.9,
        5.91,
        5.9,
        5.91,
        2.95,
        0,
        0,
        0,
        0,
    ]
    txd = [tdi * mcr / 100 for mcr in macrs]

    annual_bd = tdi / 20
    ditx = [(txdj - annual_bd) * tax_rate for txdj in txd]
    rceaf = equity_afudc / 20
    tcrj = [annual_bd + ditxj + rceaf for ditxj in ditx]
    tcr = sum(tcrj)
    total_investment = tcr + esc_land + esc_wc

    bby_equity_tot = 0.4 * total_investment
    adj_equity = [0.4 * ditxj + rceaf for ditxj in ditx]
    bd_equity = (bby_equity_tot - wc - land - sum(adj_equity)) / 20
    bby_equity = [bby_equity_tot]
    for i in range(len(adj_equity)):
        bby_equity.append(bby_equity[i] - adj_equity[i] - bd_equity)

    bby_debt_tot = 0.6 * total_investment
    bd_debt = bby_debt_tot / 20
    adj_debt = [0.6 * ditxj for ditxj in ditx]
    bby_debt = [bby_debt_tot]
    for i in range(len(adj_debt)):
        bby_debt.append(bby_debt[i] - adj_debt[i] - bd_debt)

    tcr_yearly = [
        bd_debt + adj_debt[i] + bd_equity + adj_equity[i]
        for i in range(len(adj_equity))
    ]

    otxi = 0.02 * pfi
    itx = [
        (tax_rate / (1 - tax_rate)) * (tcrj[i] + equity_roi * bby_equity[i] - txd[i])
        for i in range(20)
    ]
    fuel_cost = [fc * (1 + rnf) ** i for i in range(20)]
    oandm_cost = [oandm * (1 + int_rate) ** i for i in range(20)]
    trr_current = [
        tcrj[i]
        + equity_roi * bby_equity[i]
        + debt_roi * bby_debt[i]
        + itx[i]
        + otxi
        + fuel_cost[i]
        + oandm_cost[i]
        for i in range(20)
    ]

    trr_pvalue_current = [trr_current[i] / ((1 + wacc) ** (i + 1)) for i in range(20)]
    crf = (wacc * (1 + wacc) ** 20) / ((1 + wacc) ** 20 - 1)
    kfuel = (1 + rnf) / (1 + wacc)
    celffuel = (kfuel * (1 - kfuel**20)) / (1 - kfuel) * crf
    trrl_current = sum(trr_pvalue_current) * crf
    fcl_current = fc * celffuel
    first_term = (trrl_current - fcl_current) / (T * cap_factor)
    zk = list()
    for equip in pec:
        zk.append(first_term * equip / total_pec)
    lcoe = trrl_current / (prod_capacity * T * cap_factor)
    return zk, fcl_current / (T), lcoe


if __name__ == "__main__":
    pec = [
        2.523e6,
        2.734e6,
        1.231e6,
        1.327e6,
        1.939e6,
        0.821e6,
        0.478e6,
        0.690e6,
        9.721e6,
    ]
    prod_capacity = 29.83
    economics(pec, prod_capacity)
