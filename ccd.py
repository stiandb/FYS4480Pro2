from Quantum import Quantum
from TBME import TBME
import numpy as np


quantum = Quantum(2,TBME)
quantum.set_Z(2)
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
quantum.set_states(states)
L = 2*quantum.u.shape[0] #Toal number of spin single particle functions

h   = np.zeros((L,L))
uAS = np.zeros((L,L,L,L))

for p in range(L):
    for q in range(L):
        h[p,q] = quantum.ihj(p,q)
        for r in range(L):
            for s in range(L):
                uAS[p,q,r,s] = quantum.ijvklAS(p,q,r,s)


n_fermi = quantum.n_fermi
n_basis = quantum.n_basis
hole = slice(0,2)
part = slice(n_fermi,L)

Eref = np.einsum('ii->', h[hole, hole]) + 0.5 * np.einsum('ijij->', uAS[hole, hole, hole, hole])
print("Eref: %g" % Eref)

f_pq = h + np.einsum('pmqm->pq', uAS[:, hole, :, hole])



Dijab = np.zeros((2,2,L-n_fermi,L-n_fermi))
for i in range(n_fermi):
    for j in range(n_fermi):
        for a in range(n_fermi,L):
            for b in range(n_fermi,L):
                #Definition of Dijab in part c
                Dijab[i,j,a-n_fermi,b-n_fermi] = f_pq[i,i]+f_pq[j,j]-f_pq[a,a]-f_pq[b,b]

#Set diagonal elements of f_pq to zero for use in the iterative scheme
f_zeros = f_pq.copy()
f_zeros[np.diag_indices_from(f_zeros)] = 0

#Define amplitudes
t = np.zeros((2,2,L-n_fermi,L-n_fermi))
t = uAS[hole,hole,part,part]/Dijab #Initial guess, see Eq. 28

#CCD energy in first iteration
Ecorr = 0.25*np.einsum('ijab,ijab->',uAS[hole,hole,part,part],t)
ECCD  = Eref+Ecorr
print("ECCD (initial guess): %g" % ECCD)



def CCD_rhs(f,uAS,t):
    #Eq 23
    rhs = uAS[hole,hole,part,part].copy() #u[a,b,i,j] = u[i,j,a,b]
    ijvklmat = uAS
    f_zeros = f
    o = hole
    v = part
    rhs = ijvklmat[o,o,v,v]
    term = np.einsum('ki,jkab->ijab',f_zeros[o,o],t)
    term -= term.swapaxes(0,1)
    rhs += term
    term = -np.einsum('ac,ijbc->ijab',f_zeros[v,v],t)
    term -= term.swapaxes(2,3)
    rhs += term
    term = 0.5*np.einsum('klij,klab->ijab',ijvklmat[o,o,o,o],t)
    rhs += term
    term = np.einsum('akic,jkbc->ijab',ijvklmat[v,o,o,v],t)
    term -= term.swapaxes(0,1)
    term -= term.swapaxes(2,3)
    rhs += term
    term = 0.5*np.einsum('abcd,ijcd->ijab',ijvklmat[v,v,v,v],t)
    rhs += term
    term = 0.25*np.einsum('klcd,klab,ijcd->ijab',ijvklmat[o,o,v,v],t,t,optimize=True)
    rhs += term
    term = -0.5*np.einsum('klcd,ilab,kjcd->ijab',ijvklmat[o,o,v,v],t,t,optimize=True)
    term -= term.swapaxes(0,1)
    rhs += term
    term = -np.einsum('klcd,ikac,ljbd->ijab',ijvklmat[o,o,v,v],t,t,optimize=True)
    term -= term.swapaxes(2,3)
    rhs += term
    term = -0.5*np.einsum('klcd,ijac,klbd->ijab',ijvklmat[o,o,v,v],t,t,optimize=True)
    term -= term.swapaxes(2,3)
    rhs += term
    

    return rhs

#Iterate
max_iters = 500
for i in range(0,max_iters):
    rhs_new = CCD_rhs(f_zeros,uAS,t)
    t = rhs_new/Dijab
    Ecorr = 0.25*np.einsum('ijab,ijab->',uAS[hole,hole,part,part],t)
    ECCD  = Eref+Ecorr
    print("ECCD (initial guess):", ECCD)