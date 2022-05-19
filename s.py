import numpy as np

A2 = np.array([[1,2,2,1,1,0],[0,2,3,2,0,1]])
b2 = np.array([6,10])
c2 = np.array([1,2,3,2,0,0])

def tableau(A,b,c,B_i,t="max"):
	a1 = np.zeros((len(A),1))
	B_i = list(map(lambda x: x+1, B_i))
	b = np.array([b]).T
	A = np.concatenate([a1, A, b], axis=1)
	c = np.append(c,0)
	A = np.concatenate([-np.array([np.hstack((-1,c))]),A],axis=0)
	#print(A,B_i)
	iteration = 0
	while True:
		iteration+=1
		ci = A[0] 
        # print("Simplex Tableau: ", iteration)
        # print(A)
		minn = np.min(ci[:len(ci)-1])
		if minn >= 0:
			return ci[len(ci)-1]
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

result = tableau(A2,b2,c2,[3,2])

print(result)