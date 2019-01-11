import pickle
import numpy as np
import time

def compute_ccd_energy(h, u, t, n):
    rhs = 0
    term = 1 * (0.5) * np.einsum("lklk -> ", u[:n, :n, :n, :n])
    rhs += term
    term = 1 * (0.25) * np.einsum("dclk, lkdc -> ", t, u[:n, :n, n:, n:])
    rhs += term
    term = 1 * np.einsum("kk -> ", h[:n, :n])
    rhs += term

    return rhs
    
def g(h, u, t, n):
    rhs = np.zeros(t.shape)
    term = 1 * (0.5) * np.einsum("balk, lkji -> abij", t, u[:n, :n, :n, :n])
    rhs += term
    term = 1 * (0.5) * np.einsum("dcji, badc -> abij", t, u[n:, n:, n:, n:])
    rhs += term
    
    one_dca = np.ones((len(h) - n, len(h) - n)) - np.eye(len(h) - n)
    term = 1 * np.einsum("ac, ac, bcji -> abij", one_dca, h[n:, n:], t)
    rhs += (term - term.swapaxes(0, 1))

    one_dki = np.ones((n, n)) - np.eye(n)
    term = 1 * (-1.0) * np.einsum("ki, ki, bajk -> abij", one_dki, h[:n, :n], t)
    rhs += (term - term.swapaxes(2, 3))

    term = 1 * (-1.0) * np.einsum("acji, bkck -> abij", t, u[n:, :n, n:, :n])
    rhs += (term - term.swapaxes(0, 1))
    term = 1 * (-1.0) * np.einsum("baik, lkjl -> abij", t, u[:n, :n, :n, :n])
    rhs += (term - term.swapaxes(2, 3))
    term = 1 * (0.25) * np.einsum("balk, dcji, lkdc -> abij", t, t, u[:n, :n, n:, n:])
    rhs += term
    term = 1 * np.einsum("acik, bdjl, lkdc -> abij", t, t, u[:n, :n, n:, n:])
    rhs += (term - term.swapaxes(2, 3))
    term = 1 * np.einsum("acik, bkjc -> abij", t, u[n:, :n, :n, n:])
    rhs += ((term - term.swapaxes(0, 1)) - (term - term.swapaxes(0, 1)).swapaxes(2, 3))
    term = 1 * (0.5) * np.einsum("bail, dcjk, lkdc -> abij", t, t, u[:n, :n, n:, n:])
    rhs += (term - term.swapaxes(2, 3))
    term = 1 * (-0.5) * np.einsum("acji, bdlk, lkdc -> abij", t, t, u[:n, :n, n:, n:])
    rhs += (term - term.swapaxes(0, 1))
    rhs += u[n:, n:, :n, :n]

    return rhs
    
    


def _diag_compute_amplitudes(h, u, n, tol=1e-4):
    # Here d_{ij}^{ab} = d[a][b][i][j]
    d = np.zeros((len(h) - n, len(h) - n, n, n))

    for a in range(len(h) - n):
        for b in range(len(h) - n):
            for i in range(n):
                for j in range(n):
                    d[a][b][i][j] = h[i][i] + h[j][j] - h[n + a][n + a] - h[n + b][n + b]

    tau = np.divide(u[n:, n:, :n, :n], d, where=(d != 0))

    e_ccd_prev = 0
    e_ccd = compute_ccd_energy(h, u, tau, n)
    diff = abs(e_ccd_prev - e_ccd)

    while diff > tol:
        tau = np.divide(g(h, u, tau, n), d)

        e_ccd_prev = e_ccd
        e_ccd = compute_ccd_energy(h, u, tau, n)

        diff = abs(e_ccd_prev - e_ccd)

    return e_ccd
    


def compute_amplitudes(h, u, n, diagonal_h=True):
    if not diagonal_h:
        raise NotImplementedError("This function currently only supports a diagonal h-matrix")
    else:
        return _diag_compute_amplitudes(h, u, n)
        
def OBME(Z):
    h = np.zeros(3,3)
    for i in range(3):
        h[i,i] = Z**2/(2*i**2)
        
    return h

if __name__ == "__main__":
    n = 2
    Z = 2
    
    from TBME import TBME
    u = TBME(Z)
    h = OBME(Z)

    t1 = time.time()
    e_ccd = compute_amplitudes(h, u, 2)
    t2 = time.time()

    print ("Time spent: {0}".format(t2 - t1))
    print ("Energy: {0}".format(e_ccd))


