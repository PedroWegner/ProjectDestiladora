import numpy as np

class ModeloAntoine:
    def __init__(self, params: np.ndarray):
        self.params = np.asarray(params)

    def get_vapor_pressure(self, T: float) -> np.ndarray:
        A = self.params.T[0]
        B = self.params.T[1]
        C = self.params.T[2]
        P_sat = 10**(A - B / (T + C)) # bar
        P_sat = 10**5 * P_sat
        return np.array(P_sat)
    
if __name__ == '__main__':
    water_ant = np.array([5.40221, 1838.675, -31.73])
    ethanol_ant = np.array([5.24677, 1598.673, -46.424])
    T = 352.135 # K

    params = [ethanol_ant, water_ant]

    antoine_calc = ModeloAntoine(params=params)
    x = antoine_calc.get_vapor_pressure(T=T)