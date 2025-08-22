import numpy as np
from gamma_method import *
from peng_robinson_method import * 
from antoine_method import *
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
import openpyxl as pyxl
import os
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

    nrtl_calc = GammaNRTL(list_aij=a_ij, list_bij=b_ij, alpha=alpha) # Fixa os parametros do modelo

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
    pr_calc = ModeloPengRobinson()


    # PEGA DADOS JA SALVOS PARA O EQUILIBRIO
    name = 'water_ethanol_equilibri_0.9727099bar'
    wb = pyxl.load_workbook((os.path.dirname(os.path.abspath(__file__)) + f'\\data\\{name}.xlsx'))
    sheet = wb.active
    x_plot = []
    y_plot = []
    T_plot = []
    for l in filter(None, sheet.iter_rows(min_row=2, values_only=True)):
        x_plot.append(l[0])
        y_plot.append(l[1])
        T_plot.append(l[2])
    
    plt.plot([0,1],[0,1], color='k', linewidth=0.75)
    plt.plot(x_plot, y_plot, color='slategray', linewidth=1.25)
    plt.xlim(left=0, right=1)
    plt.ylim(bottom=0, top=1)


    # Tentativa de Mc-Thiele
    Cpl = np.array([111283.3087, 75433.29859])
    deltaH = np.array([42738814.21, 44095448.08])
    x1_F = 0.17
    x_F = np.array([x1_F, 1 - x1_F])
    x1_D = 0.80
    args = (x_F, P, antoine_calc, pr_calc, mixture, nrtl_calc, )

    T0 = np.array([300.0]) #K
    T_bolha = minimize(fun=bolha_T_FO, 
                 x0=T0,
                 args=args,
                 method='NELDER-MEAD'
                 ).x[0]
    print(T_bolha)

    Cpl_mix = np.sum(x_F * Cpl)
    deltaH_mix = np.sum(x_F * deltaH)
    T_F = 22.22 + 273.15
    q = 1 + Cpl_mix * (T_bolha - T_F) / deltaH_mix
    print(q)

    x_al_eq0 = np.array([x1_F])
    args = (q, x1_F, P, antoine_calc, pr_calc, mixture, nrtl_calc)
    x_al_eq = minimize(fun=alimentacao_equilibrio, 
                 x0=x_al_eq0,
                 args=args,
                 method='NELDER-MEAD'
                 ).x[0]
    y_al_eq = q * x_al_eq / (q - 1) + x1_F / (1 - q)
    plt.plot(np.array([x1_F, x_al_eq]), np.array([x1_F, y_al_eq]))

    Rmin = (x1_D - y_al_eq) / (y_al_eq - x_al_eq)
    print(Rmin) # ERRADO PARA O CASO ATUAL!


    args = (x1_D, P, antoine_calc, pr_calc, mixture, nrtl_calc)
    xp1 = fsolve(pinch_point_FO, 
                x0=np.array([0.6]), 
                args=args)[0]
    print(xp1)
    xp = np.array([xp1, 1 - xp1])
    args_helper = (P, T0, antoine_calc, pr_calc, mixture, nrtl_calc)

    yp1 = get_equilibrium_point(xp, *args_helper)
    print(yp1)
    m_min = (yp1 - x1_D) / (xp1 - x1_D)
    R_min = m_min / (1 - m_min)
    print(R_min, Rmin)
    intercepto_y = yp1 - m_min * xp1
    x_reta = np.linspace(0, x1_D, 10)
    y_reta = m_min * x_reta + intercepto_y
    plt.plot(x_reta, y_reta, label=f'Reta de Operação (R_min={R_min:.2f})', color='green')


    plt.show()
