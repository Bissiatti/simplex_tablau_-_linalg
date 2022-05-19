import numpy as np 
import sympy
import itertools
from numpy.linalg import matrix_rank


# Implementação do Simplex, duas fases e tableau
# Alunos: Emanuel e Rafael 

def formaPadrao(A,b,c,sinaisDes,xRes):
	"""
	A -> Matriz do problema
	b -> Vetor de restrições
	c -> Função objetivo
	sinaisDes -> sinais das restrições (=,>=,<=)
	xRes -> Restrições dos x's (x>=0,x<=0,real)

	retorna:
		A -> A na forma padrão
		c -> c na forma padrão
	"""

	AResult = np.zeros([
		np.shape(A)[0],
		np.shape(A)[1]+
		(len(sinaisDes)-sinaisDes.count("="))+
		(xRes.count("real"))])

	cResult = np.zeros([
		np.shape(A)[1]+
		(len(sinaisDes)-sinaisDes.count("="))+
		(xRes.count("real"))])

	i = 0
	j = 0
	while i < len(A[0]):
		if xRes[i] == ">=0":
			print("Ok")
			AResult[:,i+j] = A[:,i].T
			cResult[i+j] = c[i]
			print(A[:,i+j],i,j)
		elif xRes[i] == "real":
			AResult[:,i+j] = A[:,i]
			AResult[:,i+j+1] = -A[:,i]
			cResult[i+j] = c[i]
			cResult[i+j+1] = -c[i]
			j+=1
		elif xRes[i] == "<=0":
			AResult[:,i+j] = -A[:,i]
			cResult[i+j] = -c[i]
		i+=1
	k = 0
	i = i+j
	j = 0
	while k < len(sinaisDes):
		if sinaisDes[k] == "<=":
			aux = np.zeros((len(sinaisDes)))
			aux[k] = 1
			AResult[:,i+j] = aux
			j+=1
		elif sinaisDes[k] == ">=":
			aux = np.zeros((len(sinaisDes)))
			aux[k] = -1
			AResult[:,i+j] = aux
			j+=1
		k+=1

	return AResult, cResult

def formaCanonica(A,b,c,B_i):
	"""
	A -> Matriz do problema
	b -> Vetor de restrições
	c -> Função objetivo
	B_i -> Indices da base

	retorna:
		A -> Nova A
		b -> Novo b
		c -> Novo c
	"""
	A_b = A[:,B_i]
	c_b = c[B_i]
	A_bi = np.linalg.inv(A_b)
	yt = np.array([A_bi.T@c_b])
	x_b = np.dot(A_bi, b)
	AResult = A_bi@A
	bResult = A_bi@b
	cResult = yt@b+(c.T-yt@A)
	return AResult, bResult, cResult[0],x_b

def fase1(A,b):
	"""
	A -> Matriz do problema
	b -> Vetor de restrições
	
	retorna:
		os índices básicos.
	"""
	
	M = A.shape[0]
	N = A.shape[1]

	A_positive = np.copy(A)
	b = np.copy(b)
	for i in range(M):
		if b[i] < 0.0:
			b[i] = -1.0 * b[i]
			A_positive[i, :] = -1 * A_positive[i, :]

	A_ = np.concatenate((A_positive, np.eye(M)), axis=1)
	c = np.zeros(shape=(N + M, ))
	for i in range(N, N + M):
		c[i] = 1.0
	indices = list(range(N + M))

	basic_indices = list(range(N, N + M))
	B = A_[:, basic_indices]

	opt_infinity = False

	while True:
		# calcula x_b, c_b, B_inv
		B_inv = np.linalg.inv(B)
		x_b = np.dot(B_inv, b)
		c_b = c[basic_indices]

		reduced_costs = {}
		for j in indices:
			if j not in basic_indices:
				A_j = A_[:, j]
				reduced_costs[j] = c[j] - np.dot(c_b.T, np.dot(B_inv, A_j))

		# Solução ótima
		if (np.array(list(reduced_costs.values())) >= 0.0).all():
			break

		chosen_j = None
		for j in reduced_costs.keys():
			if reduced_costs[j] < 0.0:
				chosen_j = j
				break

		d_b = -1.0 * np.dot(B_inv, A_[:, chosen_j])
		# checa se a solução ótima é infinita
		if (d_b >= 0).all():
			opt_infinity = True
			break

		# Verifica se a base escolhida é L. I.
		l = None
		theta_star = None
		for i, basic_index in enumerate(basic_indices):
			if d_b[i] < 0:
				if l is None:
					l = i
				if -x_b[i]/d_b[i] < -x_b[l]/d_b[l]:
					l = i
					theta_star = -x_b[i]/d_b[i]

		basic_indices[l] = chosen_j
		basic_indices.sort()
		B = A_[:, list(basic_indices)]

	contains_artificial = False
	for x in basic_indices:
		if x >= N:
			contains_artificial = True
			break
	if not contains_artificial:

		x_b = np.dot(np.linalg.inv(B), b)
		basic_indices.sort()
		
		return basic_indices

	basic_indices_no_artificial = []
	for index in basic_indices:
		if index < N:
			basic_indices_no_artificial.append(index)	

	counter = 0
	while len(basic_indices_no_artificial) < M:
		if counter in basic_indices_no_artificial:
			continue
		B_small = A[:, basic_indices_no_artificial]
		B_test = np.concatenate((B_small, A[:, counter]), axis=1)
		if matrix_rank(B_test) == min(B_test.shape[0], B_test.shape[1]):
			basic_indices_no_artificial.append(counter)

		counter += 1

	basic_indices = basic_indices_no_artificial
	basic_indices.sort()

	return basic_indices
	
def simplex(A,b,c,B_i,t="max"):
	"""
	A -> Matriz do problema
	b -> Vetor de restrições
	c -> Função objetivo
	B_i -> Indices da base

	retorna:
		c-> Vetor x ótimo 
	"""

	if t == "min":
		c = -c
	
	z = 0

	while True:

		A,b,c,x_b = formaCanonica(A,b,c,B_i)

		if (c<=0.0001).all():
			print(c)
			x = np.zeros(shape=(A.shape[1], ))
			for i in range(x.shape[0]):
				if i in B_i:
					x[i] = x_b[B_i.index(i)]
			return x
		print(A,b,c)
		j = 0
		while j < len(c):
			maxx = 0
			k = 0
			if c[j] > maxx:
				maxx = c[j]
				k = j
			j+=1
		print(c)
		Ak = A[:,k]
		if (Ak<= 0.001).all():
			return "O problema é ilimitado"
		i = 0
		minn = 1000000
		while i < len(Ak): 
			if (Ak[i]>0):
				if minn >= b[i]/Ak[i]:
					print(Ak[i])
					minn = (b[i]/Ak[i])
					min_i = i
			i+=1
		
		B_i[min_i] = k

def simplexFase1(A,b,c,sinaisDes,xRes):
	A,c = formaPadrao(A,b,c,sinaisDes,xRes)

	n = len(A[0])
	print(n)
	A = np.concatenate([A, np.eye(len(A))],axis=1)

	w = np.zeros(len(A[0]))

	j = 0
	while j <n:
		i = 0
		while i <len(A):
			w[j] += A[i,j]
			i+=1
		j+=1
	
	print(w)
				

	print(A, c)

	v,A = tableau(A,b,w,list(range(n,len(A[0]))))

	print(A)
	

def tableau(A,b,c,B_i,t="max"):
	print(B_i)
	a1 = np.zeros((len(A),1))
	B_i = list(map(lambda x: x+1, B_i))
	b = np.array([b]).T
	A = np.concatenate([a1, A, b], axis=1)
	c = np.append(c,0)
	if t == "max":
		A = np.concatenate([-np.array([np.hstack((-1,c))]),A],axis=0)
	else:
		A = np.concatenate([np.array([np.hstack((-1,c))]),A],axis=0)
	#print(A,B_i)
	iteration = 0
	while True:
		iteration+=1
		ci = A[0] 
        # print("Simplex Tableau: ", iteration)
        # print(A)
		minn = np.min(ci[:len(ci)-1])
		if minn >= 0:
			return ci[len(ci)-1], A
		cond = (ci == minn)
		index1 = np.where(cond)[0][0]
		j = 0
		while index1 in B_i:
			index1 = list(range(1,len(ci)))[j]
			#print(index1)
			j+=1
		j = 1
		b_j = np.zeros(len(A))
		A_index1 = A[:,index1] 
		#print(A_index1)
		# print("foi foi foi\n",index1)
		if (A_index1 <= 0).all():
			print("O problema é ilimitado")
			return 
		while j < len(A):
			if A[j,index1] <= 0:
				b_j[j] = np.inf
			else:
				b_j[j] = A[j,len(ci)-1]/A[j,index1]
			j+=1
		#print(b_j)
		row = np.min(b_j[1:])
		cond = (b_j == row)
		index2 = np.where(cond)[0][0]
		#print(index2)
		A[index2] = A[index2]/A[index2,index1]
		j = 0
		while j < len(A):
			if j == index2 or A[j,index1]==0:
				j+=1
				continue
			A[j] = A[j] - A[j,index1]*A[index2]
			j+=1
		#print(A)
		B_i = []
		j = 1
		while j < len(A[0]):
			if abs(A[0][j])<= 0.000001:
				B_i.append(j)
			j+=1
		# if iteration == 5:
		# 	break	


A = np.array([[1,2,4,7,3],[2,8,9,0,0],[1,1,0,2,6],[-3,4,3,1,-1]])
b = np.array([1,2,3,4])
c = np.array([2,-1,4,2,4])

#A_1, c_1 = formaPadrao(A,b,c,["<=","=",">=",">="],[">=0",">=0","real",">=0","real"])

# print("\nTeste forma padrão")
# print(c_1,"\n",A_1)

# A2 = np.array([[0,2,2,1,0],[2,1,1,1,1]])
# b2 = np.array([1,0])
# c2 = np.array([1,3,2,0,0])

A2 = np.array([[1,2,2,1],[0,2,3,2]])
b2 = np.array([6,10])
c2 = np.array([1,2,3,2])

print(fase1(A2,b2))

#simplexFase1(A,b,c,["<=","=",">=",">="],[">=0",">=0","real",">=0","real"])

# print(A2[:,base])

# print(solve(c2,A2,b2))

#simplex(A2,b2,c2,base)


# print("\nTeste forma Canonica")
# print(formaCanonica(A2,b2,c2,[0,3]))

# print("\nTeste passo simplex")
# x2 = simplex(A2,b2,c2,[0,3])
# print(simplex(A2,b2,c2,[0,3]))
# print(A2@x2)
# print(b2)
# print("\n\n")

#print(A2)
#tableau(A2,b2,c2,[3,2])
