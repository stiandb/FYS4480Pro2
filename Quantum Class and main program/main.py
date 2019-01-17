from Quantum import *
from matplotlib.pylab import*


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



CCDsolve = CCD(2,TBME)
CCDsolve.set_Z(2)
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
CCDsolve.set_states(states)
t,E,E_MBPT = CCDsolve.solve_Energy(max_iter=1000)

print('CCD Helium Energy = ', E)
print('MBPT Helium Energy', E_MBPT)

CCDsolve = CCD(2,TBME)
CCDsolve.set_Z(2)
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
CCDsolve.set_states(states)
CCDsolve.HF_basis()
t,E,E_MBPT = CCDsolve.solve_Energy(max_iter=1000)

print('CCD Helium Energy (HF) = ', E)

CCDsolve = CCD(4,TBME)
CCDsolve.set_Z(4)
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
CCDsolve.set_states(states)
t,E,E_MBPT = CCDsolve.solve_Energy(max_iter=1000)

print('CCD BE Energy = ', E)
print('MBPT BE Energy', E_MBPT)

CCDsolve = CCD(4,TBME)
CCDsolve.set_Z(4)
states = np.array([[1,1],[1,0],[2,1],[2,0],[3,1],[3,0]])
CCDsolve.set_states(states)
CCDsolve.HF_basis()
t,E,E_MBPT = CCDsolve.solve_Energy(max_iter=1000)

print('CCD BE Energy (HF) = ', E)



	




