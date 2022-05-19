import numpy as np 
import sympy
import itertools
from numpy.linalg import matrix_rank


# Implementação do Simplex
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
			AResult[:,i+j] = A[:,i]
			cResult[i+j] = c[i]
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


def solve(c, A, b):
	"""
	This method solves the std form LP min (c.T * x) s.t. Ax = b, x >= 0 using simplex algorithm.

	Parameters:
		c, A, b (np arrays): specify the LP in standard form
	
	Returns:
		-1 if LP is infeasible or optimal is (+-) infinity, else
		x (np array): solution to the LP
	"""

	c = -c

	# ensure dimensions are okay
	assert A.shape[0] == b.shape[0], 'first dims of A and b must match, check input!'
	assert A.shape[1] == c.shape[0], 'second dim of A must match first dim of c, check input!'

	# ensure A is full rank, drop redundant rows if not
	if matrix_rank(A) < min(A.shape[0], A.shape[1]):
		print('A is not full rank, dropping redundant rows')
		_, pivots = sympy.Matrix(A).T.rref()
		A = A[list(pivots)]
		print('Shape of A after dropping redundant rows is {}'.format(A.shape))

	# define some frequently used parameters
	indices = list(range(A.shape[1]))

	# get the initial BFS (we will keep around the list of basic indices or basis matrix B as our current solution, instead of x explicitly)
	basic_indices = fase1(A,b)
	B = A[:, basic_indices]
	optimal = False
	opt_infinity = False
	iteration_number = 0
	obj_val = float('inf')

	# main simplex body
	while not optimal:
		# print iteration number
		print('simplex: starting iteration #{}, obj = {}'.format(iteration_number, obj_val))
		iteration_number += 1

		# compute x_b, c_b, B_inv
		print(B)
		B_inv = np.linalg.inv(B)
		x_b = np.dot(B_inv, b)
		if (x_b == 0.0).any():
			print('simplex: alert! this bfs is degenerate')
		c_b = c[basic_indices]

		if iteration_number == 1:
			print('initial x_b = {}, with basic_indices = {}'.format(x_b, basic_indices))

		# compute obj_val just for display purposes
		obj_val = 0.0
		for i, b_i in enumerate(basic_indices):
			obj_val += (c[b_i] * x_b[i])

		# compute reduced cost in each non-basic j-th direction
		reduced_costs = {}
		for j in indices:
			if j not in basic_indices:
				# j is a non-basic index
				A_j = A[:, j]
				reduced_costs[j] = c[j] - np.dot(c_b.T, np.dot(B_inv, A_j))


		# check if this solution is optimal
		if (np.array(list(reduced_costs.values())) >= 0.0).all():
			# all reduced costs are >= 0.0 so this means we are at optimal already
			optimal = True
			break


		# this solution is not optimal, go to a better neighbouring BFS
		chosen_j = None
		for j in reduced_costs.keys():
			if reduced_costs[j] < 0.0:
				chosen_j = j
				break

		d_b = -1.0 * np.dot(B_inv, A[:, chosen_j])
		# check if optimal is infinity
		if (d_b >= 0).all():
			# optimal is -infinity
			opt_infinity = True
			break

		# calculate theta_star and the exiting index l
		l = None
		theta_star = None
		for i, basic_index in enumerate(basic_indices):
			if d_b[i] < 0:
				if l is None:
					l = i
				if -x_b[i]/d_b[i] < -x_b[l]/d_b[l]:
					l = i
					theta_star = -x_b[i]/d_b[i]

		# form new solution by replacing basic_indices[l] with chosen_j
		basic_indices[l] = chosen_j
		basic_indices.sort()
		B = A[:, list(basic_indices)]

	
	if opt_infinity:
		print('Optimal is inifinity')
		return -1

	if not optimal:
		print('optimal not found')
		return -1

	# return solution
	x = np.zeros(shape=(A.shape[1], ))
	for i in range(x.shape[0]):
		if i in basic_indices:
			x[i] = x_b[basic_indices.index(i)]
	return x

	
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

def fase1(A,b):
	"""
	This is a helper method used by solve() method to compute the initial basic feasible solution (BFS) required by simplex algorithm.
	It uses the auxiliary LP technique to compute such a BFS.

	Parameters:
		None

	Returns:
		None if the original LP is infeasible, else
		Tuple (B, basic_indices) where
			-> B 			(np array): basis matrix of the BFS
			-> basic_indices    (list): list of basic indices of the BFS
	"""
	
	M = A.shape[0]
	N = A.shape[1]

	# new constraint matrix A_ and c, b vector (b >= 0 must hold, so multiply by -1 if not already)
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

	# variables: [ x_0, ..., x_N-1, y_0, ..., y_M-1 ]
	# init bfs of aux problem is x = 0, y = b
	# so basic initial indices for aux problem are (N, ..., N + M - 1)
	basic_indices = list(range(N, N + M))
	B = A_[:, basic_indices]

	optimal = False
	opt_infinity = False
	iteration_number = 0
	obj_val = float('inf')

	# main simplex body
	while not optimal:

		# print iteration number
		print('get_init_bfs_aux: starting iteration #{}, obj = {}'.format(iteration_number, obj_val))
		iteration_number += 1

		# compute x_b, c_b, B_inv
		B_inv = np.linalg.inv(B)
		x_b = np.dot(B_inv, b)
		if (x_b == 0.0).any():
			print('get_init_bfs_aux: alert! this bfs is degenerate')
		c_b = c[basic_indices]

		# compute obj_val just for display purposes
		obj_val = 0.0
		for i, b_i in enumerate(basic_indices):
			obj_val += (c[b_i] * x_b[i])

		# compute reduced cost in each non-basic j-th direction
		reduced_costs = {}
		for j in indices:
			if j not in basic_indices:
				# j is a non-basic index
				A_j = A_[:, j]
				reduced_costs[j] = c[j] - np.dot(c_b.T, np.dot(B_inv, A_j))


		# check if this solution is optimal
		if (np.array(list(reduced_costs.values())) >= 0.0).all():
			# all reduced costs are >= 0.0 so this means we are at optimal already
			optimal = True
			break

		# this solution is not optimal, go to a better neighbouring BFS
		chosen_j = None
		for j in reduced_costs.keys():
			if reduced_costs[j] < 0.0:
				chosen_j = j
				break

		d_b = -1.0 * np.dot(B_inv, A_[:, chosen_j])
		# check if optimal is infinity
		if (d_b >= 0).all():
			# optimal is -infinity
			opt_infinity = True
			break

		# calculate theta_star and the exiting index l
		l = None
		theta_star = None
		for i, basic_index in enumerate(basic_indices):
			if d_b[i] < 0:
				if l is None:
					l = i
				if -x_b[i]/d_b[i] < -x_b[l]/d_b[l]:
					l = i
					theta_star = -x_b[i]/d_b[i]

		# form new solution by replacing basic_indices[l] with chosen_j
		basic_indices[l] = chosen_j
		basic_indices.sort()
		B = A_[:, list(basic_indices)]

	
	if obj_val != 0.0:
		print('get_init_bfs_aux: the original problem is infeasible!')
		return None

	# if basic_indices contains no artifical variables, return that
	contains_artifical = False
	for x in basic_indices:
		if x >= N:
			contains_artifical = True
			break
	if not contains_artifical:

		assert len(basic_indices) == M, 'assertion failed, please check this'
		assert matrix_rank(B) == M, 'this should have been equal, assertion failed'
		x_b = np.dot(np.linalg.inv(B), b)
		assert (x_b >= 0.0).all(), 'this does not give a feasible solution, something is wrong, assertion failed'
		print('init_bfs_aux: assertions passed, no artificial vars in basis by chance! found a valid init bfs in {} iterations'.format(iteration_number))
		basic_indices.sort()
		
		return basic_indices

	# basis contains artificial variables
	basic_indices_no_artificial = []
	for index in basic_indices:
		if index < N:
			basic_indices_no_artificial.append(index)	

	# now have to choose columns from A that are linearly independent to the current selection of basis indices
	counter = 0
	while len(basic_indices_no_artificial) < M:
		if counter in basic_indices_no_artificial:
			continue

		# check if counter-th column of A is linearly independent with current selection of indices
		B_small = A[:, basic_indices_no_artificial]
		B_test = np.concatenate((B_small, A[:, counter]), axis=1)
		if matrix_rank(B_test) == min(B_test.shape[0], B_test.shape[1]):
			# is l.i., so take this column
			basic_indices_no_artificial.append(counter)

		counter += 1

	# test if what we got is indeed a BFS to original problem
	basic_indices = basic_indices_no_artificial
	basic_indices.sort()
	B = A[:, basic_indices]

	assert len(basic_indices) == M, 'assertion failed, please check this'
	assert matrix_rank(B) == M, 'this should have been equal, assertion failed'
	x_b = np.dot(np.linalg.inv(B), b)
	assert (x_b >= 0.0).all(), 'this does not give a feasible solution, something is wrong, assertion failed'
	print('init_bfs_aux: assertions passed! found a valid init bfs in {} iterations'.format(iteration_number))

	return basic_indices


A = np.array([[1,2,4,7,3],[2,8,9,0,0],[1,1,0,2,6],[-3,4,3,1,-1]])
b = np.array([1,2,3,4])
c = np.array([2,-1,4,2,4])

A_1, c_1 = formaPadrao(A,b,c,["<=","=",">=",">="],[">=0",">=0","real",">=0","real"])

# print("\nTeste forma padrão")
# print(c_1,"\n",A_1)

# A2 = np.array([[0,2,2,1,0],[2,1,1,1,1]])
# b2 = np.array([1,0])
# c2 = np.array([1,3,2,0,0])

A2 = np.array([[1,2,2,1,1,0],[0,2,3,2,0,1]])
b2 = np.array([6,10])
c2 = np.array([1,2,3,2,0,0])

base = fase1(A2,b2)

print(A2[:,base])

print(solve(c2,A2,b2))

#simplex(A2,b2,c2,base)


# print("\nTeste forma Canonica")
# print(formaCanonica(A2,b2,c2,[0,3]))

# print("\nTeste passo simplex")
# x2 = simplex(A2,b2,c2,[0,3])
# print(simplex(A2,b2,c2,[0,3]))
# print(A2@x2)
# print(b2)
# print("\n\n")

def tableau(A,b,c,B_i,t="max"):
	a1 = np.zeros((len(A),1))
	B_i = list(map(lambda x: x+1, B_i))
	b = np.array([b]).T
	A = np.concatenate([a1, A, b], axis=1)
	c = np.append(c,0)
	A = np.concatenate([-np.array([np.hstack((-1,c))]),A],axis=0)
	print(A,B_i)
	iteration = 0
	while True:
		iteration+=1
		ci = A[0]
		minn = np.min(ci[:len(ci)-1])
		if minn >= 0:
			return ci[len(ci)-1]
		cond = (ci == minn)
		index1 = np.where(cond)[0][0]
		j = 0
		while index1 in B_i:
			index1 = list(range(1,len(ci)))[j]
			print(index1)
			j+=1
		j = 1
		b_j = np.zeros(len(A))
		A_index1 = A[:,index1]
		print(A_index1)
		while (abs(A_index1[1:])<=0.00001).all():
			print("AAAAAAAAAAAAAAAAAAAAAAa\n",A_index1)
			print(len(ci))
			ci2 = ci[list(range(len(ci)))]
			ci2 = np.delete(ci2,index1)
			minn = np.min(ci2)
			cond = (ci2 == minn)
			aux = np.where(cond)[0][0]
			print("kkkkkkkkkkkkkkkkkkkkk",aux)
			if aux >= index1:
				index1 = aux+1
			# index1 = np.random.randint(0,len(ci)-1)
			# while index1 in B_i:
			# 	index1 = np.random.randint(0,len(ci)-1)
			print("-----------------------")
			print(A)
			print(index1)
			print("-----------------------")
			A_index1 = A[:,index1]
			print(A_index1)
			print("#######################")

		# print("foi foi foi\n",index1)
		if (A_index1 <= 0).all():
			print("O problema é ilimitado")
			return 
		while j < len(A):
			if A[j,index1] == 0:
				b_j[j] = np.inf
			else:
				b_j[j] = A[j,len(ci)-1]/A[j,index1]
			j+=1
		print(b_j)
		row = np.min(b_j[1:])
		cond = (b_j == row)
		index2 = np.where(cond)[0][0]
		print(index2)
		A[index2] = A[index2]/A[index2,index1]
		j = 0
		while j < len(A):
			if j == index2 or A[j,index1]==0:
				j+=1
				continue
			A[j] = A[j] - A[j,index1]*A[index2]
			j+=1
		print(A)
		B_i = []
		j = 1
		while j < len(A[0]):
			if abs(A[0][j])<= 0.000001:
				B_i.append(j)
			j+=1
		if iteration == 5:
			break
#print(A2)
#tableau(A2,b2,c2,[3,2])
