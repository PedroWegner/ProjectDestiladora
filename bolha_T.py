import numpy as np
from gamma_method import *
from peng_robinson_method import * 
from antoine_method import *
from scipy.optimize import minimize

def bolha_T_FO(T: float, 
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

    soma_y = np.sum(y_old)
    print(y_old)
    err = (1.0 - soma_y)**2 * 10**5
    return err
    pass

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

if __name__ == '__main__':
    # ----- TEMPERATURA E PRESSAO
    T = 352.135 # K
    P = 97270.99 # Pa
    x = np.array([0.45, 0.55])

    # ----- PARAMETROS NRTL
    a12 = -0.8009
    a21 = 3.4578
    b12 = 246.2 
    b21 = -586.1
    alpha = 0.3

    a_ij = np.array([
        [0, a12],
        [a21, 0]
    ])

    b_ij = np.array([
        [0, b12],
        [b21, 0]
    ])

    nrtl_calculator = GammaNRTL(list_aij=a_ij, list_bij=b_ij, alpha=alpha) # Fixa os parametros do modelo

    # ----- PARAMETROS ANTOINE
    water_ant = np.array([5.40221, 1838.675, -31.73])
    ethanol_ant = np.array([5.24677, 1598.673, -46.424])
    params = [ethanol_ant, water_ant]
    antoine_calc = ModeloAntoine(params=params) # Fixa os parametros do modelo

    # ----- PARAMETROS PENG-ROBINSON
    water_pr = Component(name='H2O', Tc=647.10, Pc=220.55e5, omega=0.345) 
    ethanol_pr = Component(name='C2H6O', Tc=513.90, Pc=61.48e5, omega=0.645)
    kij = 0.0
    k_ij = np.array([[0, kij],[kij,0]])
    mixture = Mixture([ethanol_pr, water_pr], k_ij=k_ij, l_ij=0.0)

    y = np.array([0.64, 0.36])
    trial_state = State(mixture=mixture, T=T, P=P, z=y, is_vapor=True) # Isso aqui pode e vai ser contornado
    
    calc_peng_robinson = ModeloPengRobinson()

    args = (x, P, antoine_calc, calc_peng_robinson, mixture, nrtl_calculator, )
    T0 = np.array([350.0])

    T = minimize(fun=bolha_T_FO, 
                 x0=T0,
                 args=args,
                 method='NELDER-MEAD'
                 )
    print(T)