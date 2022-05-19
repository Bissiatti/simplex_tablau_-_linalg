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

def simplexFase1(A,b,sinaisDes,xRes):
	artificial_i = []
	i = 0
	while i < len(sinaisDes):
		if sinaisDes[i] == "=":
			artificial_i.append(i)
		if sinaisDes[i] == ">=":
			artificial_i.append(i)
		AResult = np.zeros([
		np.shape(A)[0],
		np.shape(A)[1]+
		(len(sinaisDes)-sinaisDes.count("="))+
		(xRes.count("real"))])
		i+=1

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
			print(A[:,i+j],i,j)
		elif xRes[i] == "real":
			AResult[:,i+j] = A[:,i]
			AResult[:,i+j+1] = -A[:,i]
			j+=1
		elif xRes[i] == "<=0":
			AResult[:,i+j] = -A[:,i]
		i+=1
	k = 0
	i = i+j
	j = 0
	while k < len(sinaisDes):
		if sinaisDes[k] == "<=":
			aux = np.zeros((len(sinaisDes)))
			aux[k] = 1
			AResult[:,i+j] = aux
			cResult[i+j] = -1
			j+=1
		elif sinaisDes[k] == ">=":
			cResult[i+j] = -1
			aux = np.zeros((len(sinaisDes)))
			aux[k] = -1
			AResult[:,i+j] = aux
			j+=1
		k+=1

	print(AResult, cResult)


A = np.array([[1,2,4,7,3],[2,8,9,0,0],[1,1,0,2,6],[-3,4,3,1,-1]])
b = np.array([1,2,3,4])
c = np.array([2,-1,4,2,4])

A_1, c_1 = formaPadrao(A,b,c,["<=","=",">=",">="],[">=0",">=0","real",">=0","real"])

print("\nTeste forma padrão")
print(c_1,"\n",A_1)

# A2 = np.array([[0,2,2,1,0],[2,1,1,1,1]])
# b2 = np.array([1,0])
# c2 = np.array([1,3,2,0,0])

A2 = np.array([[1,2,2,1],[0,2,3,2]])
b2 = np.array([6,10])
c2 = np.array([1,2,3,2])

simplexFase1(A,b,["<=","=",">=",">="],[">=0",">=0","real",">=0","real"])

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
