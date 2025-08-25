# x_range = np.linspace(0.00001, 0.99999, 500) # Gerando o space_plot da fracao liquida
    # T0 = antoine_calc.get_vapor_temperature(P=P)[1] # Temperatura de vapor do agua, quando x1 = 0!
    # print(T0)
    # x_plot = x_range
    # y_plot = []
    # T_plot = []
    # for x in x_range:
    #     x_ = np.array([x, 1 - x])
    #     args = (x_, P, antoine_calc, pr_calc, mixture, nrtl_calc, )
    #     T = minimize(fun=bolha_T_FO, 
    #              x0=T0,
    #              args=args,
    #              method='NELDER-MEAD'
    #              )
    #     T0 = T.x[0]
    #     print(T0)
    #     y = bolha_T(T=T0, x=x_, P_sis=P, antoine_calc=antoine_calc, pr_calc=pr_calc, mixture=mixture, nrtl_calc=nrtl_calc)
    #     T_plot.append(T0)
    #     y_plot.append(y[0])


def bolha_T_FO(T: float, 
            x: np.ndarray, 
            P_sis: float, 
            antoine_calc: ModeloAntoine,
            pr_calc: ModeloPengRobinson,
            mixture: Mixture,
            nrtl_calc: GammaNRTL,):
    
    P_sat = antoine_calc.get_vapor_pressure(T=T)
    gamma = np.exp(nrtl_calc.get_lnGamma(T=T, fraction=x))

    # Faz uma estimativa inicial de y (Gas ideal)
    y_guess = (x * gamma * P_sat) / P_sis
    y_guess /= np.sum(y_guess)

    # Cria um estado
    local_state = State(mixture=mixture, T=T, P=P_sis, z=None, is_vapor=True)
    for _ in range(20):
        # Calcula o estado termodinâmico do gas (Peng-Robinson)
        local_state.z = np.array(y_guess)
        local_state.params = pr_calc.get_params(state=local_state)
        local_state.Z = max(pr_calc.get_Z(state=local_state))
        phi = pr_calc.get_phi(state=local_state)

        y_old = (x * gamma * P_sat) / (P_sis * phi)
        y_new = y_old / np.sum(y_old)

        if np.allclose(y_old, y_new, atol=1e-8):
            break

        y_guess = y_new


    soma_y = np.sum(y_old)
    err = (1.0 - soma_y)**2 * 10**5
    return err

    

def bolha_T(T: float, 
            x: np.ndarray, 
            P_sis: float, 
            antoine_calc: ModeloAntoine,
            pr_calc: ModeloPengRobinson,
            mixture: Mixture,
            nrtl_calc: GammaNRTL):
    
    P_sat = antoine_calc.get_vapor_pressure(T=T)
    gamma = np.exp(nrtl_calc.get_lnGamma(T=T, fraction=x))

    # Faz uma estimativa inicial de y (Gas ideal)
    y_guess = (x * gamma * P_sat) / P_sis
    y_guess /= np.sum(y_guess)

    # Cria um estado
    local_state = State(mixture=mixture, T=T, P=P_sis, z=None, is_vapor=True)
    for _ in range(20):
        # Calcula o estado termodinâmico do gas (Peng-Robinson)
        local_state.z = np.array(y_guess)
        local_state.params = pr_calc.get_params(state=local_state)
        local_state.Z = max(pr_calc.get_Z(state=local_state))
        phi = pr_calc.get_phi(state=local_state)

        y_old = (x * gamma * P_sat) / (P_sis * phi)
        y_new = y_old / np.sum(y_old)

        if np.allclose(y_old, y_new, atol=1e-8):
            break

        y_guess = y_new

    return y_guess

def get_equilibrium_point(x, P, T0, antoine_calc, pr_calc, mixture, nrtl_calc):
    """
    Calcula a composição do vapor (y) em equilíbrio com um líquido (x) a uma pressão P.
    """
    args = (x, P, antoine_calc, pr_calc, mixture, nrtl_calc)
    
    # Encontra a temperatura de bolha
    T = minimize(fun=bolha_T_FO, x0=T0, args=args, method='NELDER-MEAD').x[0]
    
    # Encontra a composição do vapor
    y1 = bolha_T(T=T, x=x, P_sis=P, antoine_calc=antoine_calc, pr_calc=pr_calc, mixture=mixture, nrtl_calc=nrtl_calc)[0]
    
    return y1


def alimentacao_equilibrio(x1, q, xf, P, antoine_calc, pr_calc, mixture, nrtl_calc) :
    x1 = x1[0]
    y_alimentacao = q * x1 / (q - 1) + xf / (1 - q)

    x = np.array([x1, 1 - x1])
    T0 = np.array([300.0]) #K
    args = (P, T0, antoine_calc, pr_calc, mixture, nrtl_calc, )
    y_equilibrio = get_equilibrium_point(x, *args)
    # T = minimize(fun=bolha_T_FO, 
    #              x0=T0,
    #              args=args,
    #              method='NELDER-MEAD'
    #              ).x[0]
    # y_equilibrio = bolha_T(T=T, x=x, P_sis=P, antoine_calc=antoine_calc, pr_calc=pr_calc, mixture=mixture, nrtl_calc=nrtl_calc)[0]

    err = (y_alimentacao - y_equilibrio)**2 * 10**4
    return err


def pinch_point_FO(xp1, xd1, P, antoine_calc, pr_calc, mixture, nrtl_calc):
    xp1 = xp1[0]
    xp = np.array([xp1, 1 - xp1])
    T0 = np.array([300.0]) #K
    # Obtem o ponto na curva de equilibrio
    args = (P, T0, antoine_calc, pr_calc, mixture, nrtl_calc, )
    
    yp = get_equilibrium_point(xp, *args)

    eps = 0.0001
    xp_pos = np.array([xp1 + eps, 1 - (xp1 + eps)])
    yp_pos = get_equilibrium_point(xp_pos, *args)
    xp_neg = np.array([xp1 - eps, 1 - (xp1 - eps)])
    yp_neg = get_equilibrium_point(xp_neg, *args)

    dyp_dxp = (yp_pos - yp_neg) / (2 * eps)

    err = (yp - xd1 - dyp_dxp * (xp1 - xd1))
    return err