
 
   
    

    h5 = h4 + q_hx
    t5 = temperature(h5, p5)

    t2 = [t2 for t2 in range(int(t4) + 5, int(t1))]
    if len(t2) == 0:
        return float("inf")
    h2 = list()
    for temp in t2:
        a, _ = enthalpy_entropy(temp, p2)
        h2.append(a)
    h2 = np.asarray(h2)
    q_hx1 = m * h1 - m * h2
    t5 = [t2 for t2 in range(int(t4), int(t1) - 5)]
    if len(t5) == 0:
        return float("inf")
    h5 = list()
    for temp in t5:
        a, _ = enthalpy_entropy(temp, p5)
        h5.append(a)
    h5 = np.asarray(h5)
    q_hx2 = m * h5 - m * h4
    q_hx = q_hx1 - q_hx2
    idx1 = np.where(q_hx[:-1] * q_hx[1:] < 0)[0]
    
  

    


    
    

    
    


    
fuel_tur = e6 - e1
    prod_tur = w_tur
 
cost_prod_execo_tur = (fuel_tur + cost_tur) / w_tur

 fuel_HX = e1 - e2

fuel_cooler = q_c
    prod_cooler = e2 - e3

    
    

fuel_comp = w_comp
    prod_comp = e4 - e3
    
##Cooler
    t3 = t2 - q_cool / (m * cp)
    t2 = t1 - q_hx / (m * cp)
    t5 = t4 + q_hx / (m * cp)
    t6 = t5 + q_heater / (m * cp)
    

 

##Economic Analysis
    
    ##Exergoeconomic Analysis
    
    
    breakpoint()
    z = cost_comp
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

        ])