


e6 = m * (h6 - h0 - T0 * (s6 - s0))

    e1 = m * (h1 - h0 - T0 * (s1 - s0))
fuel_tur = e6 - e1
    prod_tur = w_tur
 if t6 > 550:
        ft_tur = 1 + 1.106e-4 * (t6 - 550) ** 2
    elif t1 > 550:
        ft_tur = 1 + 1.106e-4 * (t1 - 550) ** 2
    else:
        ft_tur = 1
    cost_tur = 182600 * (w_tur**0.5561) * ft_tur
cost_prod_execo_tur = (fuel_tur + cost_tur) / w_tur
e2 = m * (h2 - h0 - T0 * (s2 - s0))
 fuel_HX = e1 - e2
e3 = m * (h3 - h0 - T0 * (s3 - s0))
fuel_cooler = q_c
    prod_cooler = e2 - e3
dt1_cooler = t2 - cw_temp
    dt2_cooler = t3 - cw_temp
    A_cooler = q_c / (U_c * lmtd(dt1_cooler, dt2_cooler))
    cost_cooler = 32.88 * U_c * A_cooler**0.75
e4 = m * (h4 - h0 - T0 * (s4 - s0))
fuel_comp = w_comp
    prod_comp = e4 - e3
    cost_comp = 1230000 * w_comp**0.3992
e5 = m * (h5 - h0 - T0 * (s5 - s0))
    dt1_hx = t1 - t5
    dt2_hx = t2 - t4
    A_hx = q_hx / (U_hx * lmtd(dt1_hx, dt2_hx))

    (h5, s5) = enthalpy_entropy(t5, p5)
if t1 > 550:
        ft_hx = 1 + 0.02141 * (t1 - 550)
    elif t5 > 550:
        ft_hx = 1 + 0.02141 * (t5 - 550)
    else:
        ft_hx = 1
    cost_HX = 49.45 * U_hx * A_hx**0.7544 * ft_hx
 if t6 > 550:
        ft_heater = 1 + 5.4e-5 * (t6 - 550) ** 2
    elif t5 > 550:
        ft_heater = 1 + 5.4e-5 * (t5 - 550) ** 2
    else:
        ft_heater = 1
    cost_heater = 820800 * q_heater**0.7327 * ft_heater

##Economic Analysis
    pec.append(cost_tur)
    pec.append(cost_HX)
    pec.append(cost_cooler)
    pec.append(cost_comp)
    pec.append(cost_heater)
    prod_capacity = (w_tur - w_comp) / 1e6
    zk,cftot = economics(pec, prod_capacity)
    ##Exergoeconomic Analysis
    # [c1,c2,c3,c4,c5,c6,cw_tur,cw_comp,cf]
    m1 = np.array(
        [
            [e1 - e6, 0, 0, 0, 0, 0, w_tur, 0, 0],
            [-e1 + e2, 0, 0, -e4, e5, 0, 0, 0, 0],
            [0, e2, -e3, 0, 0, 0, 0, 0, 0],
            [0, 0, -e3, e4, 0, 0, 0, w_comp, 0],
            [0, 0, 0, 0, -e5, e6, 0, 0, q_heater],
            [1,0,0,0,0,-1,0,0,0],
            [1,-1,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1]

        ]
    )
    m2 = np.asarray(zk).reshape(5, 1)
    np.append(m2,[0,0,0,])
    costs = np.linalg.solve(m1, m2)
    breakpoint()
    z = cost_comp