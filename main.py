import numpy as np
from numpy.core.numeric import identity
from numpy.lib import kron
import matplotlib.pyplot as plt
#######################delete later
class SuperBlock():
    def __init__(self, parentBLock) :
        self.nsites=parentBLock.nsites*2+2
        self.dim=parentBLock.dim**2*2**2
        self.hSB=np.kron(parentBLock.enlargedH(Bx=J,Bz=g),np.identity(2*parentBLock.dim)) +np.kron(np.identity(2*parentBLock.dim),parentBLock.enlargedH(Bx=J,Bz=g))-np.kron(np.identity(parentBLock.dim),np.kron(Block.sigmazloc,np.kron(Block.sigmazloc,np.identity(parentBLock.dim))))-np.kron(parentBLock.initialsigmaz,np.kron(identity(2),np.kron(identity(2),parentBLock.initialsigmaz)))#sum the hamiltonian of each subblock making the superblock  and add the spin interaction term at the boundary
##################################################################################################################################################################        
    def __str__(self) -> str:#if trying to print the superblock, prints the Hamiltonian in numpy matrix format
        return str(self.hSB)     
    def groundstate(self):  #finds the ground state of the superblock hamiltonian .hSB, soon implemented with Lanczos algorithm,
        energy,eigv=np.linalg.eigh(self.hSB) 
                           #momentarily using just standard numpy algorithm for hermitian matrices                            
        eigv=eigv.transpose()
        return (energy[0]/(self.nsites),eigv[0])  #RETURNS a tuble with the energy of the ground state and the corresponding eigenvector

####################

def red_dens_matr(vec): #input:
                        # a vector (class np.ndarray) |v> living in the hilbert space H_1\otimes H_2, 
                        # returns rho = \Tr_2(|v><v|), the reduced density matrix living on H_1 
    dimlocH = int(np.sqrt(vec.size) + 0.5)  #dimlocH is the dimension of the local hilbert space whose tensor square gives the space in which vec lives
                                        #also adds 0.5 to avoid eventual approximation problems
    if (dimlocH**2 !=  vec.size):#checks that the size of the vector is indeed a perfect square (as it should be for a tensor space H\otimes H)
        print("ERROR! initial vector's dimension is not a perfect square")
        return 0 
    else:
        rdm = np.empty((dimlocH,dimlocH)) 
        #creates an empty numpy array to memorize the reduced density matrix
        for i in range(dimlocH):
            for j in range(dimlocH):
                rdm[i][j] = np.sum(np.fromiter((vec[i*dimlocH + k]*vec[j*dimlocH + k] for k in range(dimlocH)),float))           
                #trace out H_2, indexed by k, while i, j are indexes of H_1
        return rdm

class Block():
    #defines two class attribute for the pauli matrices
    sigmaxloc = np.array([[0, -1],[-1,0]])
    sigmazloc = np.array([[1,0],[0,-1]])
    def __init__(self, Bx = 0, Bz = 0,dim =   2  ,hB = 0,sigmaz = 0,initialsigmaz = 0,initiateblock = False, nsites = 1) :
        #Block class, input params:
                # coupling constants--- give the onsite hamiltonian -Bx \sigmax-Bz \sigmaz, needed only when initiating the block 
                # dim--- is the number of the vectors we use to describe our block 
                # hB --- the Hamiltonian of the block in the proper basis (it's a (dim) x (dim) matrix)
                # sigmaz--- the matrix corresponding to the local z pauli operator acting on the last site of the block
                # initialsigmaz--- as before, but acting on the first site of the block
                #  initiateblock: if set to True, initialize the block to start from a single site
        if (initiateblock):
            self.dim = 2#this is the number of the vectors we use to describe our block 
            self.hB = -Bx*Block.sigmaxloc-Bz * Block.sigmazloc
            self.sigmaz = Block.sigmazloc  
            self.initialsigmaz = Block.sigmazloc#this is the operator that respresents the first spin of the block
            self.nsites = 1#2
        else:#here initialize the block manually, with input hamiltonian and everything
            self.dim = dim
            self.hB = np.zeros((dim,dim))  if (isinstance(hB, int)) else hB
            self.sigmaz =  np.zeros((dim,dim)) if (isinstance(sigmaz,int)) else sigmaz  #sigmaz is the form of the \sigma_z local operator acting on rightmost site, 
            self.nsites = nsites   
            self.initialsigmaz = np.zeros((dim,dim)) if (isinstance(initialsigmaz,int)) else initialsigmaz                                                              #needed to link with one extra site in the Block.enlarged method        
    def __str__(self) -> str: #in case it's needed to print the block, prints the bulk hamiltonian instead
        return str(self.hB)     
    def enlargedH(self, Bx, Bz):#takes the coupling constants, and returns the extended hamiltonian of the block with a single site added.
                                # The enlarged hamiltonian is H_{block}\otimes \identity(2) + \identity(block.dim)\otimes H_{single site}-block.sigmaz \otimes sigmaz
        return np.kron(self.hB,np.identity(2)) + np.kron(np.identity(self.dim),-Bx*Block.sigmaxloc-Bz * Block.sigmazloc)-np.kron(self.sigmaz,Block.sigmazloc)# + np.identity(self.dim*2)
########################################################################################################################################################################

def superblock(block,Bx = 0,Bz = 0):#Input: 1-  the block (which also contains the enlarged block as a method)
                                #       2-  the coupling constants to build the interaction between the two enlarged block
                                #Output: tuple containing   1- The energy per site of the ground state
                                #                           2- The eigenvector corresponding to the ground state in the proper basis

    hSB = np.kron(block.enlargedH(Bx = Bx,Bz = Bz),np.identity(2*block.dim))  + np.kron(np.identity(2*block.dim),block.enlargedH(Bx = Bx,Bz = Bz))-np.kron(np.identity(block.dim),np.kron(Block.sigmazloc,np.kron(Block.sigmazloc,np.identity(block.dim))))-np.kron(block.initialsigmaz,np.kron(identity(2),np.kron(identity(2),block.initialsigmaz)))
    #hSB is the hamiltonian of the superblock, it has 4 terms:
    #2 "bulk" hamiltonians for each enlarged block making up the superblock 
    #2 interaction terms for the spins at the boundaries. the interaction terms are constructed using the attributes of the class Block:
    #  In particular, the .sigmaz attribute gives the local spin z operators acting on the last spin (in a certain basis)
    #  and the .initialsigmaz gives  the local operator acting on the first spin 
    energy,eigvec = np.linalg.eigh(hSB)    
    #momentarily using just standard numpy algorithm for hermitian matrices:
    #since I only need the ground state might implement lanczos in the future                      
    eigvec = eigvec.transpose()   #transpose so that, eigvec is now an array whose elements are the eigenvectors of hSB 
                                #(of which i need the first, as the eigenvectors are ordered with decreasing eigenvalues)
    return (energy[0]/(block.nsites*2 + 2),eigvec[0])  #returns a tuble with the energy of the ground state and the corresponding eigenvector

############################################################################################################################################



def infinite_DMRG(J = 0.0,g = 0,iterativetrials = 10,cutoff = 4, precision = 0, energyeverystep = False): 
    #finds the energy of the ground state of the following hamiltonian:
    #H=-(\sum_i sigmaz_i sigmaz_{i+1}+ J sigmax_i+ g sigmaz_i)
    #Inputs: couplings J,g
    #iterativetrails: minimun number of times the algorithm is run
    #precision: if set >0, forces the algoithm to run until the difference of the energy per site(divided by the energy per site) is less than precision
    #cutoff: truncation of the hilbert space at every step, the higher, the most accurate the result. In particular it should increase with the entanglement entropy of the ground state (and be virtually infinite at the critical point)
    #energyeverystep: if set to True, prints at every run of the algorithm the output energy           
    myblock = Block(initiateblock = True)#initiates a block
    energy,grounds = superblock(myblock,Bx = J,Bz = g)#computes the ground state and ground state energy of the corresponding superblock
    i = 0#starts the counter
    while (i < iterativetrials or (gap> precision and precision>0)) :
        rdm = np.array(red_dens_matr(grounds))#finds the reduced density matrix describing the ground state in the first enlarged block
        eig,eigvec = np.linalg.eigh(rdm)#finds the eigenvectors
        eigvec = eigvec.transpose()#transpose the matrix to have an array whose elements are the eigenvectors
        if (len(eig)>cutoff): #if I have more eigenvectors than my cutoff, only keep the dominant ones, else keep them all
            selected_eiv = np.array([eigvec[i] for i in range(len(eig)-1,len(eig)-cutoff-1,-1)])#use the fact they are in growing order   
        else:
            selected_eiv = np.flip(eigvec)#flip to have the selected_eigenvectors in decreasing order of importance
        newhB = np.dot(selected_eiv,np.dot(myblock.enlargedH(Bx = J,Bz = g),selected_eiv.transpose()))  #finds the Hamiltonian of the enlarged block in the basis of the selected_eig, 
                                                                                                        #thus reducing the dimensionality of the Hamiltonian
        newsigmaz = np.dot(selected_eiv,np.dot(np.kron(np.identity(myblock.dim),Block.sigmazloc),selected_eiv.transpose()))    
        #similarly, finds sigmaz for the first and last site in the new basis
        newinitialsigmaz = np.dot(selected_eiv,np.dot(np.kron(myblock.initialsigmaz,np.identity(  2  )),selected_eiv.transpose()))    
        myblock = Block(dim = len(selected_eiv),hB = newhB,sigmaz = newsigmaz,initialsigmaz = newinitialsigmaz,nsites = myblock.nsites + 1)
        #updates the block with the new hamiltonian and local spin operators
        gap=energy
        energy,grounds = superblock(myblock,Bx = J,Bz = g)
        gap=np.abs((energy-gap)/energy)#computes the relative gap with the energy compputed in the previous run
        i  +=  1 #increases the counter
        if (energyeverystep):#with this option on, prints the energy at every run of the algorithm
            print("ENERGY = %f\n gap=%f" %(energy,gap)) 
    return (energy,gap, myblock.nsites)#returns the energy, the gap and the number of runs of the algorithM(=number of sites covered by the simulation, since it adds one at every step and starts with 1)

def analytical_energy(J,N):
    return -np.mean(np.fromiter((np.sqrt(1+J**2-2*J*np.cos(2*np.pi* k/N)) for k in range(N)), float))
N=5
data=[]
oldcutoff=3
for j in range(1,2*N):#create dataset
    cutoff=max(4,min(13, int(-5*np.log(abs(1-j/N+1e-10))))) # chooses a j-dependent cutoff which scales logarithmically near the critical point J=1 and is always 4<J<15
    if (cutoff!= oldcutoff):
        data.append([j/N,infinite_DMRG(J=j/N,  precision=1e-6   , cutoff=cutoff)[0]])
        oldcutoff=cutoff
    else:
        data.append([j/N,infinite_DMRG(J=j/N,  precision=5e-5   , cutoff=cutoff)[0]])

#plot dataset(data), compare with analytical formula (plotted using a number of points  equal to numbersamplepoints)
data=np.array(data)
print(data)
xdata, ydata=data.transpose()
plt.plot(xdata,ydata,'.')
f=open("data.txt", 'w')
f.write("#J             Energy\n")
np.savetxt(f,data,fmt='%f',delimiter='             ')
plt.xlabel("magnetic field x axis")
plt.ylabel("energy density")
numbersamplepoints=100
xdata=np.arange(0,2,2/numbersamplepoints)
ydata=np.array([analytical_energy(xdata[i],100) for i in range(len(xdata))])
plt.plot(xdata,ydata)


plt.show()
f.close

