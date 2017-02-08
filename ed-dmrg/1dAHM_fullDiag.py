import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spslin
import time
import argparse

# Define Pauli operators
Id = sps.eye(2)
Sx = sps.csr_matrix(np.array([[0,1.],[1,0]]))
Sz = sps.csr_matrix(np.array([[1,0.],[0,-1]]))
Sy = sps.csr_matrix(np.array([[0.,-1j],[1j,0.]]))
#Sy = -1j * Sz.dot(Sx)

#--------------------------------------------------------------
# Build up interaction between site i and j

def buildInteractionXX(i,j,L):
    OpList = []
    for k in range(L):
	if (k == i):
	    OpList.append(Sx)
	elif (k == j):
	    OpList.append(Sx)
	else:
	    OpList.append(Id)

    return reduce(sps.kron,OpList)



#--------------------------------------------------------------
# Build up interaction between site i and j

def buildInteractionYY(i,j,L):
    OpList = []
    for k in range(L):
	if (k == i):
	    OpList.append(Sy)
	elif (k == j):
	    OpList.append(Sy)
	else:
	    OpList.append(Id)

    return reduce(sps.kron,OpList)



#--------------------------------------------------------------
# Build up interaction between site i and j

def buildInteractionZZ(i,j,L):
    OpList = []
    for k in range(L):
	if (k == i):
	    OpList.append(Sz)
	elif (k == j):
	    OpList.append(Sz)
	else:
	    OpList.append(Id)

    return reduce(sps.kron,OpList)


#--------------------------------------------------------------
# Build transverse-field Ising model

def build1dHeisenbergModel(L):

    Dim = 2**L

    #Ham = sps.csr_matrix(np.zeros((Dim,Dim)))
    Ham = np.zeros((Dim,Dim))	
    for i in range(L-1):
	Ham = Ham + buildInteractionXX(i,i+1,L)
	Ham = Ham + buildInteractionYY(i,i+1,L)
        Ham = Ham + buildInteractionZZ(i,i+1,L)

    # Periodic Boundary Conditions
    Ham = Ham + buildInteractionXX(L-1,0,L)
    Ham = Ham + buildInteractionYY(L-1,0,L)
    Ham = Ham + buildInteractionZZ(L-1,0,L)

    return 0.25*Ham

def tensorProduct(Psi,Phi):

    prod = np.zeros((len(Psi)*len(Phi)))
    k=0
    for i in range(len(Psi)):
        for j in range(len(Phi)):
            prod[k] = Psi[i]*Phi[j] 
            k += 1
    return prod



#--------------------------------------------------------------
# Main function

def main(args):
	


    print ('\n\n---------------------------------\n')
    print (' EXACT DIAGONALIZATION OF HEISENBERG MODEL\n\n')
    
    # Hamiltonian parameters
    N = args.N;
    
    print ('Number of spins   N = %d\n' % N)
    #file = open("energy_ed_1d_TFIM.dat", "w")
    #file.write('L\t\tE\n')
    
    print ('Building the Hamiltonian...'),
    H = build1dHeisenbergModel(N)
    print ('DONE\n')
    #HamDense = np.asarray(Hamiltonian.todense())
    #HamDense = Hamiltonian
    #e = np.linalg.eigvalsh(HamDense)
    #print HamDense
    print ('Solving...'),
    (e,psi) = np.linalg.eigh(H)
    print ('DONE\n\n')

    print ('Ground State Energy: %.10f\n\n' % (e[0]/(1.0*N)))
    ##print 'Ground State Energy = %f\n' % e[0]/N
    #print ('Ground State Wavefunction:\n')
    #gs = psi[:,0]
    #print psi
    #print gs 
    #print e 
    
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--N',type=int)
    parser.add_argument('--h',type=float)
    parser.add_argument('--J',type=float) 
    args = parser.parse_args()

    main(args)

