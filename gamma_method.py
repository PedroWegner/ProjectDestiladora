import numpy as np
from abc import ABC, abstractmethod

class GammaModel(ABC):
    def __init__(self, R: float=8.314):
        self.R = R
        self.ncomps = None
        super().__init__()
    
    @abstractmethod
    def get_lnGamma(self, fraction: list[float]) -> list[float]:
        pass


class GammaNRTL(GammaModel):
    def __init__(self, list_aij: list[float], alpha: float=0.20, R: float=8.314, list_bij: list[float]=None):
        super().__init__(R=R)
        print('Modelo NRTL iniciado')
        # 1. Converter entradas para arrays NumPy para operações vetorizadas
        self.aij = np.array(list_aij)
        self.bij = None if list_bij is None else np.array(list_bij)
        self.alpha = alpha
        self.ncomps = len(list_aij)


    def get_lnGamma(self, T: float, fraction: list[float]) -> list[float]:
        """
        Calcula todos os parâmetros do modelo NRTL de forma vetorizada.
        """
        # Cálculo de tau (τ) e G (matrizes n x n)
        # Operações elemento a elemento, sem a necessidade de loops.
        if self.bij is not None:
            tau = self.aij + self.bij / T
        else:
            tau = self.aij / (self.R * T)
        G = np.exp(-self.alpha * tau)

        # Cálculo de phi (φ) e theta (θ) (vetores de tamanho n)
        # O operador @ realiza a multiplicação de matrizes.
        # G.T é a matriz transposta de G.
        phi = G.T @ fraction
        theta = (tau * G).T @ fraction

        # Cálculo do coeficiente de atividade (γ) (vetor de tamanho n)
        # As operações são aplicadas a arrays inteiros, aproveitando o "broadcasting" do NumPy.
        theta_div_phi = theta / phi
        
        # O termo da soma é calculado com multiplicação de matrizes
        sum_term_matrix = G * (tau - theta_div_phi)
        sum_term_vector = sum_term_matrix @ (fraction / phi)

        ln_gamma = theta_div_phi + sum_term_vector
        return ln_gamma


if __name__ == '__main__':
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

    
    # Processo
    T = 352.135 # K
    P = 97270.99 # Pa

    nrtl_calculator = GammaNRTL(list_aij=a_ij, list_bij=b_ij, alpha=0.3)
    x = np.array([0.45, 0.55])
    gamma = nrtl_calculator.get_lnGamma(T=T, fraction=x)
    print(np.exp(gamma))