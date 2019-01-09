from Quantum import *
from matplotlib.pylab import*
float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return('\n'.join(rv))




def TBME(Z):

	'''
	Hardcoded two-body matrix elements (TBME) <pq|V|rs> for project1 in Fys4480.
	Note that these are the radial integrals and do NOT inculde spin, thus you have 
	to properly anti-symmetrize the TBME's yourself.
	'''

	u 		   = np.zeros((3,3,3,3))
	u[0,0,0,0] = (5*Z)/8.0
	u[0,0,0,1] = (4096*np.sqrt(2)*Z)/64827.0
	u[0,0,0,2] = (1269*np.sqrt(3)*Z)/50000.0
	u[0,0,1,0] = (4096*np.sqrt(2)*Z)/64827.0
	u[0,0,1,1] = (16*Z)/729.0
	u[0,0,1,2] = (110592*np.sqrt(6)*Z)/24137569.0
	u[0,0,2,0] = (1269*np.sqrt(3)*Z)/50000.0
	u[0,0,2,1] = (110592*np.sqrt(6)*Z)/24137569.0
	u[0,0,2,2] = (189*Z)/32768.0
	u[0,1,0,0] = (4096*np.sqrt(2)*Z)/64827.0
	u[0,1,0,1] = (17*Z)/81.0
	u[0,1,0,2] = (1555918848*np.sqrt(6)*Z)/75429903125.0
	u[0,1,1,0] = (16*Z)/729.0
	u[0,1,1,1] = (512*np.sqrt(2)*Z)/84375.0
	u[0,1,1,2] = (2160*np.sqrt(3)*Z)/823543.0
	u[0,1,2,0] = (110592*np.sqrt(6)*Z)/24137569.0
	u[0,1,2,1] = (29943*np.sqrt(3)*Z)/13176688.0
	u[0,1,2,2] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[0,2,0,0] = (1269*np.sqrt(3)*Z)/50000.0
	u[0,2,0,1] = (1555918848*np.sqrt(6)*Z)/75429903125.0
	u[0,2,0,2] = (815*Z)/8192.0
	u[0,2,1,0] = (110592*np.sqrt(6)*Z)/24137569.0
	u[0,2,1,1] = (2160*np.sqrt(3)*Z)/823543.0
	u[0,2,1,2] = (37826560*np.sqrt(2)*Z)/22024729467.0
	u[0,2,2,0] = (189*Z)/32768.0
	u[0,2,2,1] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[0,2,2,2] = (617*Z)/(314928.0*np.sqrt(3))
	u[1,0,0,0] = (4096*np.sqrt(2)*Z)/64827.0
	u[1,0,0,1] = (16*Z)/729.0
	u[1,0,0,2] = (110592*np.sqrt(6)*Z)/24137569.0
	u[1,0,1,0] = (17*Z)/81.0
	u[1,0,1,1] = (512*np.sqrt(2)*Z)/84375.0
	u[1,0,1,2] = (29943*np.sqrt(3)*Z)/13176688.0
	u[1,0,2,0] = (1555918848*np.sqrt(6)*Z)/75429903125.0
	u[1,0,2,1] = (2160*np.sqrt(3)*Z)/823543.0
	u[1,0,2,2] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[1,1,0,0] = (16*Z)/729.0
	u[1,1,0,1] = (512*np.sqrt(2)*Z)/84375.0
	u[1,1,0,2] = (2160*np.sqrt(3)*Z)/823543.0
	u[1,1,1,0] = (512*np.sqrt(2)*Z)/84375.0
	u[1,1,1,1] = (77*Z)/512.0
	u[1,1,1,2] = (5870679552*np.sqrt(6)*Z)/669871503125.0
	u[1,1,2,0] = (2160*np.sqrt(3)*Z)/823543.0
	u[1,1,2,1] = (5870679552*np.sqrt(6)*Z)/669871503125.0
	u[1,1,2,2] = (73008*Z)/9765625.0
	u[1,2,0,0] = (110592*np.sqrt(6)*Z)/24137569.0
	u[1,2,0,1] = (2160*np.sqrt(3)*Z)/823543.0
	u[1,2,0,2] = (37826560*np.sqrt(2)*Z)/22024729467.0
	u[1,2,1,0] = (29943*np.sqrt(3)*Z)/13176688.0
	u[1,2,1,1] = (5870679552*np.sqrt(6)*Z)/669871503125.0
	u[1,2,1,2] = (32857*Z)/390625.0
	u[1,2,2,0] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[1,2,2,1] = (73008*Z)/9765625.0
	u[1,2,2,2] = (6890942464*np.sqrt(2/3)*Z)/1210689028125.0
	u[2,0,0,0] = (1269*np.sqrt(3)*Z)/50000.0
	u[2,0,0,1] = (110592*np.sqrt(6)*Z)/24137569.0
	u[2,0,0,2] = (189*Z)/32768.0
	u[2,0,1,0] = (1555918848*np.sqrt(6)*Z)/75429903125.0
	u[2,0,1,1] = (2160*np.sqrt(3)*Z)/823543.0
	u[2,0,1,2] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[2,0,2,0] = (815*Z)/8192.0
	u[2,0,2,1] = (37826560*np.sqrt(2)*Z)/22024729467.0
	u[2,0,2,2] = (617*Z)/(314928.0*np.sqrt(3))
	u[2,1,0,0] = (110592*np.sqrt(6)*Z)/24137569.0
	u[2,1,0,1] = (29943*np.sqrt(3)*Z)/13176688.0
	u[2,1,0,2] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[2,1,1,0] = (2160*np.sqrt(3)*Z)/823543.0
	u[2,1,1,1] = (5870679552*np.sqrt(6)*Z)/669871503125.0
	u[2,1,1,2] = (73008*Z)/9765625.0
	u[2,1,2,0] = (37826560*np.sqrt(2)*Z)/22024729467.0
	u[2,1,2,1] = (32857*Z)/390625.0
	u[2,1,2,2] = (6890942464*np.sqrt(2/3)*Z)/1210689028125.0
	u[2,2,0,0] = (189*Z)/32768.0
	u[2,2,0,1] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[2,2,0,2] = (617*Z)/(314928.0*np.sqrt(3))
	u[2,2,1,0] = (1216512*np.sqrt(2)*Z)/815730721.0
	u[2,2,1,1] = (73008*Z)/9765625.0
	u[2,2,1,2] = (6890942464*np.sqrt(2/3)*Z)/1210689028125.0
	u[2,2,2,0] = (617*Z)/(314928.0*np.sqrt(3))
	u[2,2,2,1] = (6890942464*np.sqrt(2/3)*Z)/1210689028125.0
	u[2,2,2,2] = (17*Z)/256.0
	return(u)
"""
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
Z = 2
HeliumCI = CI(2,TBME)
HeliumCI.set_states(states)
HeliumCI.set_Z(Z)
e,u,H = HeliumCI.solve()
print("CI Helium")
print(e[0])
print(e)
print(bmatrix(H) + '\n')
print('-------------------------')
HeliumHF = HF(2,TBME)
HeliumHF.set_states(states)
HeliumHF.set_Z(Z)
e, C, H, E = HeliumHF.solve(max_iter=1)
print('HF Helium')
print('One iteration')
print(E)
print(bmatrix(H) + '\n')
e,C,H,E = HeliumHF.solve()
print('Convergence')
print(E)
print(bmatrix(H) + '\n')


BerylliumCI = CI(4,TBME)
BerylliumCI.set_states(states)
Z = 4
BerylliumCI.set_Z(Z)
e,u,H = BerylliumCI.solve()
print('CI Beryllium')
print(e[0])
print(e)
print(bmatrix(H) + '\n')
BerylliumHF = HF(4,TBME)
BerylliumHF.set_states(states)
BerylliumHF.set_Z(Z)
e,C,H,E = BerylliumHF.solve(max_iter=1)
print('HF Beryllium')
print('One iteration')
print(E)
print(bmatrix(H) + '\n')
e,C,H,E = BerylliumHF.solve()
print('Convergence')
print(E)
print(bmatrix(H) + '\n')


def HE_ref_energy(Z):
	return((5/8 - Z)*Z)

def BE_ref_energy(Z):
	return((586373/373248 - 5/4*Z)*Z)

Z = np.linspace(0,2,50)

EHE = HE_ref_energy(Z)
EBE = BE_ref_energy(Z)

CIEHE = np.zeros(len(Z))
HFEHE = np.zeros(len(Z))
CIEBE = np.zeros(len(Z))
HFEBE = np.zeros(len(Z))
real_energy_HE = -2.9037*np.ones(len(Z))
real_energy_BE = -14.6674*np.ones(len(Z))

for i,Zz in enumerate(Z):
	HeliumCI = CI(2,TBME)
	HeliumCI.set_states(states)
	HeliumCI.set_Z(Zz)
	e,u,H = HeliumCI.solve()
	CIEHE[i] = e[0]

	HeliumHF = HF(2,TBME)
	HeliumHF.set_states(states)
	HeliumHF.set_Z(Zz)
	e, C, H, E = HeliumHF.solve()
	HFEHE[i] = E

	BerylliumCI = CI(4,TBME)
	BerylliumCI.set_states(states)
	BerylliumCI.set_Z(Zz)
	e,u,H = BerylliumCI.solve()
	CIEBE[i] = e[0]

	BerylliumHF = HF(4,TBME)
	BerylliumHF.set_states(states)
	BerylliumHF.set_Z(Zz)
	e,C,H,E = BerylliumHF.solve()
	HFEBE[i] = E
	


plot(Z,EHE,label='Reference Energy')
plot(Z,CIEHE,label='CI')
plot(Z,HFEHE,label='HF')
plot(Z,real_energy_HE,'--k')
legend()
xlabel('Z - Number of protons')
ylabel('Energy [a.u]')
title("Two electrons")
show()

plot(Z,EBE,label='Reference Energy')
plot(Z,CIEBE,label='CI')
plot(Z,HFEBE,label='HF')
plot(Z,real_energy_BE,'--k')
legend()
xlabel('Z - Number of protons')
ylabel('Energy [a.u]')
title("Four electrons")
show()

print('Reference energy HE:', HE_ref_energy(2))
print('Reference energy BE:', BE_ref_energy(4))
"""



CCDsolve = CCD(2,TBME)
CCDsolve.set_Z(2)
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
CCDsolve.set_states(states)
t,E = CCDsolve.solve_Energy(max_iter=50)
print(E)
print(t)


