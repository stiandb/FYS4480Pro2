import numpy as np
import math
from tqdm import tqdm

class Quantum:
	def __init__(self,n_fermi,TBME):
		"""
		n_fermi - Number of electrons (number of states below fermi level)
		TBME - function which returns the two-body matrix elements of the Hamiltonian.
		"""
		self.n_fermi = n_fermi
		self.TBME = TBME
	def set_Z(self,Z):
		"""
		Z - Atomic number / coloumb interaction
		"""
		self.Z = Z
		self.u = self.TBME(Z)
	def set_states(self,states):
		"""
		states - list containing lists [n,m_s] (basis states), where n refers to quantum number n of the state
				 and m_s is the secondary spin number of the state.
		"""
		self.n_basis = len(states)
		self.states = states
	def ihj(self,i,j):
		if self.states[i,1] == self.states[j,1] and self.states[i,0] == self.states[j,0]:
			return(-self.Z**2/(2*self.states[i,0]**2))
		else:
			return(0)
	def ijvkl(self,i,j,k,l):
		if self.states[i,1] == self.states[k,1] and self.states[j,1] == self.states[l,1]:
			if self.states[i,0] == self.states[j,0] and self.states[i,1] == self.states[j,1]:
				return(0)
			elif self.states[k,0] == self.states[l,0] and self.states[k,1] == self.states[l,1]:
				return(0)
			else:
				return(self.u[self.states[i,0]-1,self.states[j,0]-1,\
						self.states[k,0]-1,self.states[l,0]-1])
		else:
			return(0)
	def cHc(self):
		expval = 0
		for i in range(self.n_fermi):
			expval += self.ihj(i,i)
			for j in range(self.n_fermi):
				expval += 0.5*(self.ijvkl(i,j,i,j) - self.ijvkl(i,j,j,i))
		return(expval)
	def cHia(self,i,a):
		expval = self.ihj(i,a)
		for j in range(self.n_fermi):
			expval += self.ijvkl(i,j,a,j) - self.ijvkl(i,j,j,a)
		return(expval)
	def iaHjb(self,i,a,j,b):
		expval = self.ijvkl(j,a,b,i) - self.ijvkl(j,a,i,b)
		if i == j:
			expval += self.ihj(a,b)
			for k in range(self.n_fermi):
				expval += self.ijvkl(a,k,b,k) - self.ijvkl(a,k,k,b)
			if a == b:
				for k in range(self.n_fermi):
					expval += self.ihj(k,k)
					for l in range(self.n_fermi):
						expval += 0.5*(self.ijvkl(k,l,k,l) - self.ijvkl(k,l,l,k))
		if a == b:
			expval -= self.ihj(j,i)
			for k in range(self.n_fermi):
				expval += self.ijvkl(j,k,k,i) - self.ijvkl(j,k,i,k)
		return(expval)
	def pfq(self,p,q):
		sum1 = 0
		for i in range(self.n_fermi):
			sum1 += self.ijvkl(p,i,q,i) - self.ijvkl(p,i,i,q)
		return(self.ihj(p,q) + sum1)

	def ijvklAS(self,i,j,k,l):
		return(self.ijvkl(i,j,k,l) - self.ijvkl(i,j,l,k))

	def ijvklHF(self,p,q,r,s):
		return(self.ijvklHFmat[p,q,r,s])
	def pfqHF(self,p,q):
		if p == q:
			sum1 = 0
			for i in range(self.n_fermi):
				sum1 += self.ijvklHF(p,i,q,i)
			for alpha in range(self.n_basis):
				for beta in range(self.n_basis):
					sum1 += self.C[p,alpha]*self.C[q,beta]*self.ihj(alpha,beta)
			return(sum1)
		else:
			return(0)
	def cHcHF(self):
		expval = 0
		for i in range(self.n_fermi):
			for alpha in range(self.n_basis):
				for beta in range(self.n_basis):
					expval += self.C[i,alpha]*self.C[i,beta]*self.ihj(alpha,beta)
					for j in range(self.n_fermi):
						for gamma in range(self.n_basis):
							for delta in range(self.n_basis):
								expval += 0.5*self.C[i,alpha]*self.C[j,beta]*self.C[i,gamma]*self.C[j,delta]\
										*(self.ijvkl(alpha,beta,gamma,delta) - self.ijvkl(alpha,beta,delta,gamma))
		return(expval)
class CI(Quantum):
	"""
	Find CI (singles) approximation of eigenstates of the Hamiltonian
	"""
	def solve(self):
		n_fermi = self.n_fermi
		n_basis = self.n_basis
		n = n_fermi*(n_basis - n_fermi) + 1
		H = np.zeros((n,n))
		H[0,0] = self.cHc()
		for i in range(n_fermi):
			for j in range(n_fermi,n_basis):
				cHa = self.cHia(i,j)
				H[0,1+i*(n_basis-n_fermi)+j-n_fermi] = cHa
				H[1+i*(n_basis-n_fermi)+j-n_fermi,0] = cHa
				for k in range(n_fermi):
					for l in range(n_fermi,n_basis):
						H[j+1-n_fermi+i*(n_basis-n_fermi),k*(n_basis-n_fermi)+l+1-n_fermi] = self.iaHjb(i,j,k,l)
		self.H_CI = H
		self.e_CI, self.u_CI = np.linalg.eigh(H)
		return(self.e_CI,self.u_CI,self.H_CI)

class HF(Quantum):
	"""
	Find the Hartree-Fock approximation of the ground state energy
	"""
	def iter(self):
		n_basis = self.n_basis
		self.H_hf = np.zeros((n_basis,n_basis))
		for alpha in range(n_basis):
			for beta in range(n_basis):
				self.H_hf[alpha,beta] += self.ihj(alpha,beta)
				for j in range(self.n_fermi):
					for gamma in range(n_basis):
						for delta in range(n_basis):
							self.H_hf[alpha,beta] += self.C[j,gamma]*self.C[j,delta]*\
												(self.ijvkl(alpha,gamma,beta,delta) - self.ijvkl(alpha,gamma,delta,beta))
		self.e_1,C = np.linalg.eigh(self.H_hf)
		self.C = C.T
	def calculate_energies(self):
		n_basis = self.n_basis
		n_fermi = self.n_fermi
		e = 0
		for i in range(n_fermi):
			for alpha in range(n_basis):
				for beta in range(n_basis):
					e += self.C[i,alpha]*self.C[i,beta]*self.ihj(alpha,beta)
					for j in range(n_fermi):
						for gamma in range(n_basis):
							for delta in range(n_basis):
								e += 0.5*self.C[i,alpha]*self.C[j,beta]*self.C[i,gamma]*self.C[j,delta]\
									*(self.ijvkl(alpha,beta,gamma,delta) - self.ijvkl(alpha,beta,delta,gamma))
		return(e)

	def solve(self,tol = 1e-8,max_iter=1000):
		n_basis = self.n_basis
		n_fermi = self.n_fermi
		self.C = np.identity(n_basis)
		e_0 = np.zeros(n_basis)
		err=1
		converge=False
		for i in range(max_iter):
			self.iter()
			err=np.sum(np.abs(self.e_1 - e_0))/n_basis
			e_0 = self.e_1
			if err < tol:
				converge = True
				break
		if converge != True:
			print('HF did not converge')
		E = self.calculate_energies()
		return(self.e_1,self.C,self.H_hf, E)

class CCD(Quantum):
	"""
	Find the Coupled Cluster doubles energy approximation
	"""
	def init(self):
		n_fermi = self.n_fermi
		n_basis = self.n_basis
		self.o = slice(0,n_fermi)
		self.v = slice(n_fermi,n_basis)
		self.t = np.zeros((n_fermi,n_fermi,n_basis - n_fermi,n_basis - n_fermi))
		self.t_new = np.ones((n_fermi,n_fermi,n_basis - n_fermi,n_basis - n_fermi))
		self.f_pq = np.zeros((n_basis,n_basis))
		self.ijvklmat = np.zeros((n_basis,n_basis,n_basis,n_basis))
		self.Dijab = np.zeros((n_fermi,n_fermi,n_basis - n_fermi,n_basis - n_fermi))
		for i in range(n_fermi):
			for j in range(n_fermi):
				for a in range(n_fermi,n_basis):
					for b in range(n_fermi,n_basis):
						self.Dijab[i,j,a-n_fermi,b-n_fermi] = self.pfq(i,i) + self.pfq(j,j) - self.pfq(a,a) - self.pfq(b,b)
						self.t[i,j,a - n_fermi,b - n_fermi] = self.ijvklAS(a,b,i,j)/self.Dijab[i,j,a-n_fermi,b-n_fermi]
		for p in range(n_basis):
			for q in range(n_basis):
				self.f_pq[p,q] = self.pfq(p,q)
				for r in range(n_basis):
					for s in range(n_basis):
						self.ijvklmat[p,q,r,s] = self.ijvklAS(p,q,r,s)
		#Check if Fock matrix is diagonal
		try:
			if self.HFbas:
				f = self.f_pq - np.diag(np.diag(self.f_pq))
				if np.sum(np.fabs(f)) > 1e-16:
					print('Error: fock matrix not diagonal in HF basis')
					print(f)
					exit()
		except AttributeError:
			None

		
	def HF_basis(self,tol = 1e-8,max_iter=1000):
		"""
		Utilizes the HF-solution as basis states
		Call this function before solve_Energy() to compute the energy in HF basis.
		"""
		self.HFbas = True
		n_basis = self.n_basis
		self.ijvklAS = self.ijvklHF
		self.pfq = self.pfqHF
		self.cHc = self.cHcHF
		HFsolve = HF(self.n_fermi, self.TBME)
		HFsolve.set_Z(self.Z)
		HFsolve.set_states(self.states)
		e,C,H,E = HFsolve.solve(tol,max_iter)
		self.C = C
		self.ijvklHFmat = np.zeros((n_basis,n_basis,n_basis,n_basis))
		for p in range(n_basis):
			for q in range(n_basis):
				for r in range(n_basis):
					for s in range(n_basis):
						pqvrs = 0
						for alpha in range(n_basis):
							for beta in range(n_basis):
								for gamma in range(n_basis):
									for delta in range(n_basis):
										pqvrs += self.C[p,alpha]*self.C[q,beta]*self.C[r,gamma]*self.C[s,delta]\
										*(self.ijvkl(alpha,beta,gamma,delta) - self.ijvkl(alpha,beta,delta,gamma))
						self.ijvklHFmat[p,q,r,s] = pqvrs
	
	def t_update(self,f_pq,ijvklmat,t):
		n_fermi = self.n_fermi
		n_basis = self.n_basis
		o = self.o
		v = self.v
		f_zeros = f_pq.copy()
		f_zeros[np.diag_indices_from(f_zeros)] = 0
		rhs = ijvklmat[o,o,v,v].copy()
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
		return(rhs)
	def solve_Amplitude(self,max_iter, eps):
		self.init()
		self.E_MBPT = None
		self.CCD_convergence = False
		try:
			if self.HFbas:
				None
		except AttributeError:
			self.E_MBPT = self.cHc() + 0.25*np.einsum('ijab,ijab->',self.ijvklmat[self.o,self.o,self.v,self.v],self.t)

		for i in range(max_iter):
			self.t_new = self.t_update(self.f_pq,self.ijvklmat,self.t)/self.Dijab
			if np.sum(np.fabs(self.t_new - self.t)) <= eps:
				self.t = self.t_new.copy()
				self.CCD_convergence = True
				break
			self.t = self.t_new.copy()
	def solve_Energy(self,max_iter = 50, eps = 1e-10):
		n_fermi = self.n_fermi
		n_basis = self.n_basis
		self.solve_Amplitude(max_iter = max_iter, eps = eps)
		o = self.o
		v = self.v
		E = self.cHc()
		E += 0.25*np.einsum('ijab,ijab->',self.ijvklmat[o,o,v,v],self.t)
		if not self.CCD_convergence:
			print('CCD did not converge')
		return(self.t,E,self.E_MBPT)


		


						




