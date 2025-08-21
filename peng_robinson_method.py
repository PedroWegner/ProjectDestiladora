from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.linalg import eigh

RGAS_SI = 8.314 # constantes dos gases J mol-1 K-1
@dataclass
class Component:
    name: str
    Tc: float
    Pc: float
    omega: float

@dataclass
class Mixture:
    components: list[Component]
    k_ij: np.ndarray
    l_ij: np.ndarray

@dataclass
class State:
    mixture: Mixture
    z: np.ndarray
    is_vapor: bool
    T: float
    P: Optional[float] = None
    Z: Optional[float] = None
    Vm: Optional[float] = None
    V: Optional[float] = None
    n: float = 100
    helmholtz_derivatives: Optional[Dict[str, any]] = None
    P_derivatives: Optional[Dict[str, any]] = None
    fugacity_dict: Optional[Dict[str, any]] = None
    residual_props: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, any]] = None
    

class CubicParametersWorker:
    def __init__(self, omega1: float, omega2: float, m):
        self.omega1 = omega1
        self.omega2 = omega2
        self.m = m

        self.mixture = None
        self.components = None
        self.z = None
        self.n = None
        self.Tc = None
        self.Pc = None
        self.omega = None
        self.T = None
        self.Tr = None
        self.params_dict = {}

    def _calculate_pure_params(self) -> None:
        m = self.m(self.omega)
        alpha = (1 + m * (1 - np.sqrt(self.Tr)))**2
        ac = self.omega1 * (RGAS_SI * self.Tc)**2 / self.Pc
        ai = ac * alpha
        bi = self.omega2 * (RGAS_SI * self.Tc) / self.Pc 
        
        self.params_dict['m']= m
        self.params_dict['alpha']= alpha
        self.params_dict['ac']= ac
        self.params_dict['ai']= ai
        self.params_dict['bi']= bi

    def _calculate_binary_mixture_params(self):
        ai = self.params_dict['ai']
        bi = self.params_dict['bi']
        aij_matrix = (np.sqrt(np.outer(ai, ai))) * (1 - self.mixture.k_ij)
        bij_matrix = 0.5 * (np.add.outer(bi,bi)) * (1 - self.mixture.l_ij)
        a_mix = self.z @ aij_matrix @ self.z
        b_mix = self.z @ bij_matrix @ self.z

        self.params_dict['aij_matrix'] =  aij_matrix
        self.params_dict['bij_matrix'] =  bij_matrix
        self.params_dict['a_mix'] =  a_mix
        self.params_dict['b_mix'] =  b_mix

    def _calculate_B_and_derivatives(self):
        ni = self.z * self.n
        bij_matrix = self.params_dict['bij_matrix']
        b_mix = self.params_dict['b_mix']
        B = self.n * b_mix
        Bi = np.array((2 * bij_matrix @ ni - B) / self.n)
        soma_BiBj = Bi.reshape(-1, 1) + Bi.reshape(1, -1)
        Bij = (2 * bij_matrix - soma_BiBj) / self.n

        self.params_dict['B'] =  B
        self.params_dict['Bi'] =  Bi
        self.params_dict['Bij'] =  Bij
        
    def _calculate_D_and_derivatives(self):
        ni = np.array(self.z * self.n)
        ai = self.params_dict['ai']
        aij_matrix = self.params_dict['aij_matrix']
        alpha = self.params_dict['alpha']
        m = self.params_dict['m']
        ac = self.params_dict['ac']
        a_mix = self.params_dict['a_mix']
        D = self.n**2 * a_mix
        Di = 2 * (ni @ aij_matrix)
        Dij = 2 * aij_matrix

        alphaij_T = ac * (- m * (alpha * self.Tr)**0.5) / self.T
        aii_ajj = np.outer(ai, ai)
        aii_dajj = np.outer(ai, alphaij_T)
        ajj_daii = np.outer(alphaij_T, ai)
        aij_T = (1 - self.mixture.k_ij) *(aii_dajj + ajj_daii) / (2 * aii_ajj**0.5)

        DiT = 2 * ni @ aij_T
        DT = (1/2) * ni @ DiT

        alphaii_TT = ac * m * (1 + m) * self.Tr**0.5 / (2 * self.T**2)
        # Eq. 105
        delh_delT = - (1 / (2 * (aii_ajj)**(3 / 2))) * (aii_dajj + ajj_daii)**2
        daii_dajj = np.outer(alphaij_T, alphaij_T)
        aii_ddajj = np.outer(ai, alphaii_TT)
        ajj_ddaii = np.outer(alphaii_TT, ai)
        delg_delT = (2 * daii_dajj + aii_ddajj + ajj_ddaii) * (1 / aii_ajj**0.5)
        daij_TT = ((1 - self.mixture.k_ij) / 2) * (delh_delT + delg_delT)
        DTT = ni @ daij_TT @ ni

        self.params_dict['D'] =  D
        self.params_dict['Di'] =  Di
        self.params_dict['DiT'] =  DiT
        self.params_dict['Dij'] =  Dij
        self.params_dict['DT'] =  DT
        self.params_dict['DTT'] =  DTT

    def _allocate_variables(self, state: State) -> None:
        self.mixture = state.mixture
        self.components = state.mixture.components
        self.z = state.z
        self.n = state.n
        self.Tc = np.array([c.Tc for c in self.components])
        self.Pc = np.array([c.Pc for c in self.components])
        self.omega = np.array([c.omega for c in self.components])
        self.T = state.T 
        self.Tr = self.T / self.Tc

    def _deallocate_variables(self) -> None:
        self.mixture = None
        self.components = None
        self.z = None
        self.n = None
        self.Tc = None
        self.Pc = None
        self.omega = None
        self.T = None
        self.Tr = None
        self.params_dict = {}

    def params_to_dict(self, state: State):
        # Alloca variaveis necessaria para os calculos
        self._deallocate_variables()
        self._allocate_variables(state=state)
        
        self._calculate_pure_params()
        self._calculate_binary_mixture_params()
        self._calculate_B_and_derivatives()
        self._calculate_D_and_derivatives()
        """Empacota todos os calculos do worker para enviar para o strategy"""
        return self.params_dict

class SolveZWorker:
    def __init__(self, delta1: float, delta2: float):
        self.delta1 = delta1
        self.delta2 = delta2
        
    def _solver_Z(self, A: float, B: float):
        # 1. Parametros do modelos
        delta = self.delta1 + self.delta2
        delta_inv = self.delta1 * self.delta2
        # 2. Solucao analitica da cubica
        a1 = B * (delta - 1) - 1
        a2 = B**2 * (delta_inv - delta) - B * delta + A
        a3 = - (B**2 * delta_inv * (B + 1) + A * B)
        _Q = (3 * a2 - a1**2) / 9
        _R = (9 * a1 * a2 - 27 * a3 -2 *a1**3)/54
        _D = _Q**3 + _R**2
        if _D < 0:
            theta = np.arccos(_R / np.sqrt(-_Q**3))
            x1 = 2 * np.sqrt(-_Q) * np.cos(theta / 3)  - a1 /3
            x2 = 2 * np.sqrt(-_Q) * np.cos((theta + 2 * np.pi) / 3) - a1 /3
            x3 = 2 * np.sqrt(-_Q) * np.cos((theta + 4 * np.pi) / 3) - a1 /3
        else:
            _S = np.cbrt(_R + np.sqrt(_D))
            _T = np.cbrt(_R - np.sqrt(_D))
            x1 = _S + _T - (1/3) * a1
            x2 = (-1/2)*(_S + _T) - (1/3) * a1 + (1/2) * 1j * np.sqrt(3) * (_S - _T)
            x3 = (-1/2)*(_S + _T) - (1/3) * a1 - (1/2) * 1j * np.sqrt(3) * (_S - _T)
        # 3. Limpeza das raizes obtidas
        Z = [x1, x2, x3]
        Z = [r.real for r in Z if np.isclose(r.imag, 0) and r.real > 0]
        return sorted(Z)

    def get_Z_(self, state: State) -> tuple:
        A = state.params['a_mix'] * state.P / (RGAS_SI * state.T)**2
        B = state.params['b_mix'] * state.P / (RGAS_SI * state.T)
        Z = self._solver_Z(A=A, B=B)
        return Z

    def get_Z(self, state: State, params) -> tuple:
        state = state
        # print(state.params)
        A = params['a_mix'] * state.P / (RGAS_SI * state.T)**2
        B = params['b_mix'] * state.P / (RGAS_SI * state.T)
        Z = self._solver_Z(A=A, B=B)
        is_vapor = None
        if len(Z) == 1:
            # print("O sistema só tem uma fase possível")
            if Z[0] < 0.5:
                # print("O estado só pode ser liquido")
                is_vapor = False
            else: 
                # print("O sistema só pode ser vapor")
                is_vapor = True
        else: 
            is_vapor = state.is_vapor
        return (Z, is_vapor)

class CubicCoreModelWorker:
    def __init__(self, delta1: float, delta2: float):
        self.delta1 = delta1
        self.delta2 = delta2

        self._core_model = {}
        

    def _calculate_F(self, state: State, D: float) -> None:
        f =self._core_model['f'] 
        fV =self._core_model['fV']
        fB = self._core_model['fB']
        fVV =self._core_model['fVV']
        fBV = self._core_model['fBV'] 
        fBB=self._core_model['fBB']

        g =self._core_model['g'] 
        gV =self._core_model['gV']
        gB = self._core_model['gB']
        gVV =self._core_model['gVV']
        gBV= self._core_model['gBV'] 
        gBB =self._core_model['gBB']
        
        #Fn FB e FD
        self._core_model['F'] = - state.n * g - D * f / state.T
        self._core_model['Fn'] = -g
        self._core_model['FT'] = D * f / state.T**2
        self._core_model['FV'] = - state.n * gV - D * fV / state.T
        self._core_model['FB'] =  - state.n * gB - D * fB / state.T
        self._core_model['FD'] = - f / state.T
        self._core_model['FnV'] = - gV
        self._core_model['FnB'] = -gB
        self._core_model['FTT'] = - 2 * self._core_model['FT'] / state.T
        self._core_model["FBT"] = D * fB / state.T**2
        self._core_model['FDT'] = f / state.T**2
        self._core_model['FBV'] = - state.n * gBV - D * fBV / state.T
        self._core_model['FBB'] = - state.n * gBB - D *fBB / state.T
        self._core_model['FDV'] = - fV / state.T
        self._core_model['FBD'] = - fB / state.T
        self._core_model['FTV'] = D * fV / state.T**2
        self._core_model['FVV'] = - state.n * gVV - D * fVV / state.T
    
    def _calculate_f_functions(self, state: State, B: float) -> None:
        f = 1 / (RGAS_SI * B * (self.delta1 - self.delta2)) * np.log((state.V + self.delta1 * B) / (state.V + self.delta2 * B))
        fV = (1 / (RGAS_SI * B * (self.delta1 - self.delta2))) * (1 /(state.V + self.delta1 * B) - 1 /(state.V + self.delta2 * B))
        fB = - (f + fV * state.V) / B
        fVV = (1 / (RGAS_SI * B * (self.delta1 - self.delta2))) * (-1 /(state.V + self.delta1 * B)**2 + 1 /(state.V + self.delta2 * B)**2)
        fBV = - (2 * fV + state.V * fVV) / B
        fBB = - (2 * fB + state.V * fBV) / B

        self._core_model['f'] = f
        self._core_model['fV'] = fV
        self._core_model['fB'] = fB
        self._core_model['fVV'] = fVV
        self._core_model['fBV'] = fBV
        self._core_model['fBB'] = fBB

    def _calculate_g_functions(self, state: State, B: float) -> None:
        self._core_model['g'] = np.log(1 - B / state.V)
        self._core_model['gV'] = B / (state.V * (state.V - B))
        self._core_model['gB'] = - 1 / (state.V - B)
        self._core_model['gVV'] = - 1 / (state.V - B)**2 + 1 / state.V**2
        self._core_model['gBV'] = 1 / (state.V - B)**2
        self._core_model['gBB'] = - 1 / (state.V - B)**2

    def core_model_to_dict(self, state: State, params: Dict[str, any]) -> Dict[str, any]:
        self._core_model = {}
        B = params['b_mix'] * state.n
        D = params['a_mix'] * state.n**2
        self._calculate_f_functions(state=state, B=B)
        self._calculate_g_functions(state=state, B=B)
        self._calculate_F(state=state, D=D)
        return self._core_model

class CubicHelmholtzDerivativesWorker:
    def __init__(self):
        self.derivatives = {}

    def _calculate_F_parcial_derivatives(self, params: Dict[str, any], core_model: Dict[str, float]) -> None:
        Bi = np.array(params['Bi'])
        Di = np.array(params['Di'])
        DiT = params['DiT']
        DT = params['DT']
        DTT = params['DTT']
        Bij = np.array(params['Bij'])
        Dij = np.array(params['Dij'])
        Fn = core_model['Fn']
        FB = core_model['FB']
        FD = core_model['FD']
        FT = core_model['FT']
        FV = core_model['FV']
        FnB = core_model['FnB']
        FBD = core_model['FBD']
        FBB = core_model['FBB']
        FBT = core_model['FBT']
        FDT = core_model['FDT']
        FnV = core_model['FnV']
        FBV = core_model['FBV']
        FDV = core_model['FDV']
        FTT = core_model['FTT']
        FTV = core_model['FTV']
        FVV = core_model['FVV']


        self.derivatives['dF_dni'] = Fn + FB * Bi + FD * Di
        self.derivatives['dF_dT'] = FT + FD * DT
        self.derivatives['dF_dV'] = FV

        t_FnB = FnB * (np.add.outer(Bi, Bi))
        t_FBD = FBD * (np.outer(Bi, Di) + np.outer(Di, Bi))
        self.derivatives['dF_dninj'] = t_FnB + t_FBD + FB * Bij + FBB * np.outer(Bi, Bi) + FD * Dij
        self.derivatives['dF_dniT'] = (FBT + FBD * DT) * Bi + FDT * Di + FD * DiT
        self.derivatives['dF_dniV'] = FnV + FBV * Bi + FDV * Di 
        self.derivatives['dF_dTT'] = FTT + 2 * FDT * DT + FD * DTT
        self.derivatives['dF_dTV'] = FTV + FDV * DT
        self.derivatives['dF_dVV'] = FVV
    
    def helmholtz_derivatives_to_dict(self, params: Dict[str, any], core_model: Dict[str, float]) -> Dict[str, any]:
        self.derivatives = {}
        self._calculate_F_parcial_derivatives(params=params, core_model=core_model)
        return self.derivatives

class PressionDerivativesWorker:
    def __init__(self):
        self.derivatives = {}

    def _calculate_P_derivatives(self, state: State) -> None:
        dF_dVV = state.helmholtz_derivatives['dF_dVV']
        dF_dTV = state.helmholtz_derivatives['dF_dTV']
        dF_dniV = state.helmholtz_derivatives['dF_dniV']

        self.derivatives['dP_dV'] = - RGAS_SI * state.T * dF_dVV - state.n * RGAS_SI * state.T / (state.V**2)
        self.derivatives['dP_dT'] = - RGAS_SI * state.T * dF_dTV + state.P / state.T
        self.derivatives['dP_dni'] = - RGAS_SI * state.T * dF_dniV + RGAS_SI * state.T / state.V
        
    def P_derivatives_to_dict(self, state: State):
        self.derivatives = {}
        self._calculate_P_derivatives(state=state)
        return self.derivatives
    
class FugacityWorker:
    def __init__(self):
        self.fugacity_dict = {}

    def _calculate_fugacity(self, state: State) -> None:
        dF_dni = np.array(state.helmholtz_derivatives['dF_dni'])
        self.fugacity_dict['lnphi'] = dF_dni.reshape(-1) - np.log(state.Z)
        self.fugacity_dict['phi'] = np.exp(self.fugacity_dict['lnphi'])

    def _caculate_fugacity_derivatives(self, state: State) -> None:
        dF_dniT = np.array(state.helmholtz_derivatives['dF_dniT'])
        dF_dninj = state.helmholtz_derivatives['dF_dninj']
        dP_dV = state.P_derivatives['dP_dV']
        dP_dT = state.P_derivatives['dP_dT']
        dP_dni = np.array(state.P_derivatives['dP_dni'])
        # Volume parcial molar
        Vi = np.array(- dP_dni / dP_dV).reshape(-1)
        n_dlnphi_dni = state.n * dF_dninj + 1 + (state.n * np.outer(dP_dni, dP_dni)) / (RGAS_SI * state.T * dP_dV)

        self.fugacity_dict['dlnphi_dT'] = dF_dniT + 1 / state.T - Vi * dP_dT / (RGAS_SI * state.T)    
        self.fugacity_dict['dlnphi_dP'] = Vi / (RGAS_SI * state.T) - 1 / state.P
        self.fugacity_dict['n_dlnphi_dni'] = n_dlnphi_dni
        self.fugacity_dict['dlnphi_dni'] = n_dlnphi_dni / state.n

    def fugacity_to_dict(self, state: State) -> Dict[str, any]:
        self.fugacity_dict = {}
        self._calculate_fugacity(state=state)
        self._caculate_fugacity_derivatives(state=state)
        return self.fugacity_dict

class ResidualPropertiesWorker:
    def __init__(self):
        self.residual_dict = {}

    def _calculate_residual_properties(self, state: State, core_model: Dict[str, any]) -> None:
        Sr_TVn = (- state.T * state.helmholtz_derivatives['dF_dT'] - core_model['F']) * RGAS_SI
        Ar = core_model['F'] * state.T * RGAS_SI

        self.residual_dict['Sr'] = Sr_TVn + state.n * RGAS_SI * np.log(state.Z)  
        self.residual_dict['Hr'] = Ar + state.T * Sr_TVn + state.P * state.V - state.n * RGAS_SI * state.T
        self.residual_dict['Gr'] = Ar + state.P * state.V - state.n * RGAS_SI * state.T * (1 + np.log(state.Z))

        self.residual_dict['F'] = core_model['F']

    def residual_props_to_dict(self, state: State, core_model: Dict[str, any]) -> Dict[str, float]:
        self.residual_dict = {}
        self._calculate_residual_properties(state=state, core_model=core_model)
        return self.residual_dict

class TestDerivativesEngine:
    def __init__(self):
        pass

    def _test_residual_gibbs(self, state: State) -> None:
        ni = state.n * state.z
        lnphi = state.fugacity_dict['lnphi']
        termo_1 = np.sum(ni * lnphi)
        termo_2 = state.residual_props['Gr'] / (RGAS_SI * state.T)
        test = np.allclose(termo_1, termo_2, 1e-6)
        if test:
            print("Teste de Gibbs residual adimensional passou (Eq. 31, cap.2)")
        else:
            print("Falha no teste do gibbs residual....")

    def _test_gibbs_duhem(self, state: State) -> None:
        ni = state.n * state.z
        dlnphi_dni = state.fugacity_dict['dlnphi_dni']
        res = ni @ dlnphi_dni
        _test = np.allclose(res, 0, atol=1e-6)
        if _test:
            print("Teste de Gibbs-Duhem passou (Eq. 34, cap .2)")
        else:
            print("Gibbs-Duhem é uma fraude aqui")

    def _test_pressure_derivatives(self, state: State) -> None:
        ni = state.n * state.z
        dlnphi_dP = state.fugacity_dict['dlnphi_dP']
        n_dlnphi_dP= np.sum(ni * dlnphi_dP)
        Z_nP = (state.Z - 1) * state.n / state.P
        _test = np.allclose(n_dlnphi_dP, Z_nP, 1e-6)
        
        if _test:
            print("Teste da derivada da pressão passou (Eq. 36, cap. 2)")
        else:
            print("Teste da pressão é uma fraude")

    def _test_temperature_derivatives(self, state: State) -> None:
        ni = state.n * state.z
        dlnphi_dT = state.fugacity_dict['dlnphi_dT']
        Hr = state.residual_props['Hr']
        n_dlnphi_dT = np.sum(ni * dlnphi_dT)
        Hr_RT = - Hr / (RGAS_SI * state.T**2)
        _test = np.allclose(Hr_RT, n_dlnphi_dT, atol=1e-6)
        if _test:
            print('Teste de Entalpia Residual passou (Eq. 37, cap.2)')
        else:
            print('Teste de entalpia residual é uma fraude!!!!!')

    def _second_derivatives(self, state: State) -> None:
            dF_dVV = state.helmholtz_derivatives['dF_dVV']
            dF_dniV = state.helmholtz_derivatives['dF_dniV']
            dF_dninj = state.helmholtz_derivatives['dF_dninj']
            n = state.n
            n_array = state.z * n
            V = state.Vm * n

            first_value = V * dF_dniV + n_array @ dF_dninj
            first_test = np.allclose(first_value, 0, 1e-6)
            second_value = V * dF_dVV + n_array @ dF_dniV
            second_test = np.isclose(second_value, 0, 1e-6)
            _test = first_test and second_test
            if _test:
                print('O teste de criticalidade passaram (Eq. 60 e 61, cap.2)')
            else: 
                print("As segunda derivadas não passaram")

    def tests(self, state: State) -> None:
        self._test_residual_gibbs(state=state)
        self._test_gibbs_duhem(state=state)
        self._test_pressure_derivatives(state=state)
        self._test_temperature_derivatives(state=state)
        self._second_derivatives(state=state)

class ModeloPengRobinson: #essa classe pode ser quebrada para adotar outras EoS cubicas!!!!
    def __init__(self):
        # 1. Parametros universais do modelo de Peng-Robinson
        self.delta1 = 1 + np.sqrt(2)
        self.delta2 = 1 - np.sqrt(2)
        self.omega1 = 0.45724
        self.omega2 = 0.07780
        self.m_func = lambda omega: 0.37464 + 1.54226 * omega - 0.26992 * omega**2

        # Inicializando os workers da classe
        self.params_worker = CubicParametersWorker(omega1=self.omega1, omega2=self.omega2, m=self.m_func)
        self.solver_Z_worker = SolveZWorker(delta1=self.delta1, delta2=self.delta2)
        self.core_model_worker =  CubicCoreModelWorker(delta1=self.delta1, delta2=self.delta2)
        self.helmholtz_derivatives_worker = CubicHelmholtzDerivativesWorker()
        self.pression_derivatives_worker = PressionDerivativesWorker()
        self.fugacity_worker = FugacityWorker()
        self.residual_props_worker = ResidualPropertiesWorker()

    def calculate_state(self, state: State) -> None:
        # Worker especifico da Strategy
        params = self.params_worker.params_to_dict(state=state)
        state.params = params

        Z, is_vapor = self.solver_Z_worker.get_Z(state=state, params=params) # ponto de apoio
        state.is_vapor = is_vapor
        if state.is_vapor:
            state.Z = max(Z)
        else:
            state.Z = min(Z)
        state.Vm = state.Z * RGAS_SI * state.T / state.P
        state.V = state.Vm * state.n

        core_model = self.core_model_worker.core_model_to_dict(state=state, params=params) # ponto de apoio

        state.helmholtz_derivatives = self.helmholtz_derivatives_worker.helmholtz_derivatives_to_dict(params=params, core_model=core_model)
        state.P_derivatives = self.pression_derivatives_worker.P_derivatives_to_dict(state=state)
        state.fugacity_dict = self.fugacity_worker.fugacity_to_dict(state=state)
        state.residual_props = self.residual_props_worker.residual_props_to_dict(state=state, core_model=core_model)

    def calculate_params(self, state: State) -> None:
        state.params = self.params_worker.params_to_dict(state=state)

    def calculate_state_2(self, state: State) -> None:
        # params = self.params_worker.params_to_dict(state=state)
        # state.params
        # state.Vm = 4 * params['b_mix']
        Vm = state.Vm
        T = state.T
        b = state.params['b_mix']
        a = state.params['a_mix']

        P = RGAS_SI * T / (Vm - b) - a / ((Vm + self.delta1 * b) * (Vm + self.delta2 * b))
        state.P = P
        Z = P * Vm / (RGAS_SI * T)
        state.Z = Z
        state.V = state.Vm * state.n
        core_model = self.core_model_worker.core_model_to_dict(state=state, params=state.params)

        state.helmholtz_derivatives = self.helmholtz_derivatives_worker.helmholtz_derivatives_to_dict(params=state.params, core_model=core_model)
        state.P_derivatives = self.pression_derivatives_worker.P_derivatives_to_dict(state=state)
        state.fugacity_dict = self.fugacity_worker.fugacity_to_dict(state=state)
        state.residual_props = self.residual_props_worker.residual_props_to_dict(state=state, core_model=core_model)



    def get_Z(self, state: State) -> np.ndarray:
        Z = self.solver_Z_worker.get_Z_(state=state) # ponto de apoio
        return Z

    def get_params(self, state: State) -> np.ndarray:
        params = self.params_worker.params_to_dict(state=state)
        return params

    def get_phi(self, state: State) -> np.ndarray:
        state.Vm = state.Z * RGAS_SI * state.T / state.P
        state.V = state.Vm * state.n

        core_model = self.core_model_worker.core_model_to_dict(state=state, params=state.params) # ponto de apoio

        state.helmholtz_derivatives = self.helmholtz_derivatives_worker.helmholtz_derivatives_to_dict(params=state.params, core_model=core_model)
        state.P_derivatives = self.pression_derivatives_worker.P_derivatives_to_dict(state=state)
        state.fugacity_dict = self.fugacity_worker.fugacity_to_dict(state=state)

        return state.fugacity_dict['phi']


    def calculate_state_3(self, state: State, Z: np.ndarray) -> None:
        state.Z = Z
        # if state.is_vapor:
        #     state.Z = max(Z)
        # else:
        #     state.Z = min(Z)
        state.Vm = state.Z * RGAS_SI * state.T / state.P
        state.V = state.Vm * state.n

        core_model = self.core_model_worker.core_model_to_dict(state=state, params=state.params) # ponto de apoio

        state.helmholtz_derivatives = self.helmholtz_derivatives_worker.helmholtz_derivatives_to_dict(params=state.params, core_model=core_model)
        state.P_derivatives = self.pression_derivatives_worker.P_derivatives_to_dict(state=state)
        state.fugacity_dict = self.fugacity_worker.fugacity_to_dict(state=state)
        state.residual_props = self.residual_props_worker.residual_props_to_dict(state=state, core_model=core_model)
        

if __name__ == '__main__':
    water = Component(name='H2O', Tc=647.10, Pc=220.55e5, omega=0.345) 
    ethanol = Component(name='C2H6O', Tc=513.90, Pc=61.48e5, omega=0.645)
    kij = 0.0
    k_ij = np.array([[0, kij],[kij,0]])
    mixture = Mixture([ethanol, water], k_ij=k_ij, l_ij=0.0)


    # processo
    T = 352.135 # K
    P = 97270.99 # Pa
    y = np.array([0.64, 0.36])
    trial_state = State(mixture=mixture, T=T, P=P, z=y, is_vapor=True)
    
    calc_peng_robinson = ModeloPengRobinson()
    trial_state.params = calc_peng_robinson.get_params(state=trial_state)
    trial_state.Z = max(calc_peng_robinson.get_Z(state=trial_state))
    phi = calc_peng_robinson.get_phi(state=trial_state)
    print(phi)
    print(trial_state.Z)