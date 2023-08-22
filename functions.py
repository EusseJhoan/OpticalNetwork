import numpy as np
from IPython.display import display, Math



def print_matrix(array):
    """Print matrix on a pretty form"""

    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))



def density_reorder(matrix):
    """Create a density matrix from ouput array"""

    stackn=np.hstack((matrix[0]))

    for n in range (matrix.shape[0]-1):
        
        stacknp=np.hstack((matrix[n+1]))
        rho=np.vstack((stackn,stacknp))
        stackn=rho

    return rho





def Bj(matrix):
    """Calculate the matrix associated with the transformation from 
    StrawberyFields to the usual basis (Jonatan(2014)) """

    
    size, _ = matrix.shape

    # Create a sample matrix
    matrix = np.eye(size)

    # Split the matrix vertically into two sub-matrices
    matrix1, matrix2 = np.split(matrix, 2, axis=1)

    
    a0=np.vstack(([row[0] for row in matrix1],[row[0] for row in matrix2]))
    # Create a zeros matriz of shape size x size
    #Bj=np.zeros((size,size))
    for i in range((size//2)-1):
        a=np.vstack(([row[i+1] for row in matrix1],[row[i+1] for row in matrix2]))
        a0=np.vstack((a0,a))
    
    Bj=a0.transpose()
       
    return Bj



def COB_CovMtx(cov_mtx):
    """Transform the Covariance Matrix from the basis used in 
    StrawberyFields to the usual basis (Jonatan(2014)) """
    

    # SF defines vector of quadrature operator as r=(x1,x2,...,xn,p1,p2,...,pn)
    # Jonatan defines vector of quadrature operator as r=(x1,p1,...,xn,pn)
    # Then, we should do a basis transfromation to obtain same Jonatan form of matrices 

  
    #Apply transfromation of basis of Covariance Matrix (V) from SF to Jonatan (Bj^-1*V*Bj)
    cov_mtx2 = np.linalg.inv(Bj(cov_mtx)) @ cov_mtx @ Bj(cov_mtx)

    return cov_mtx2



def COB_Means(means):
    """Transform the means vector from the basis used in StrawberyFields
     to the usual basis (Jonatan(2014)) """
    
    # SF defines vector of quadrature operator as r=(x1,x2,...,xn,p1,p2,...,pn)
    # Jonatan defines vector of quadrature operator as r=(x1,p1,...,xn,pn)
    # Then, we should do a basis transfromation to obtain same Jonatan form of matrices 


    # Create a matrix to use the sizes on Bj() function
    matrix=np.eye(means.size)

    # #Transform vector of means (<r>) to Jonatan basis (Bj^-1*<r>) 
    means2 = np.linalg.inv(Bj(matrix)) @ means

    return  means2  


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


def SimonCriterion(covariance_mtx, hbar):
    """Apply Simon criterion to a two mode system"""
    
    hbar2=hbar**2
    
    A, C, Ct, B =  split(covariance_mtx, 2, 2)
    J=np.array([[0,1],[-1,0]])
    
    left= np.linalg.det(A)*np.linalg.det(B)+ ((hbar2/4)-abs(np.linalg.det(C)))**2 - np.trace(A@J@C@J@B@J@Ct@J)  
    right= (hbar2/4)*(np.linalg.det(A)+np.linalg.det(B)) 
    
    if left >= right:
        print("The bipartite system is separable")
        
    else:
        print("The bipartite system is entangled")

      
def TM_LogNegativity(covariance_mtx):
    """Calculate logarithmic negativity for two mode states"""
    
    A, C, Ct, B = split(covariance_mtx, 2, 2)
    inv_tr = np.linalg.det(A)+np.linalg.det(B)-2*np.linalg.det(C)
    eig_tr2 = 0.5*(inv_tr-np.sqrt(inv_tr**2 -4*np.linalg.det(covariance_mtx)))
    eig_tr=np.sqrt(eig_tr2)
    log_neg=np.max(np.array((0,-np.log2(eig_tr))))
    
    return log_neg 


def PT_CovMtx(covariance_matrix, modes, order):
    """Calculate the partial transposition of the covariance matrix in xp-order or normal order"""
    
    N = covariance_matrix.shape[0]//2
    X = np.ones(N)
    P = np.ones(N)
   
    
    for i in modes:
        P[i] = -1
        T = np.diag(np.concatenate((X, P)))
    
    if order == "xp":
        return T @ covariance_matrix @ T 
    
    elif order == "normal":
        T = COB_CovMtx(T)
        return T @ covariance_matrix @ T 
        
    else:
        print("The order command is wrong you should use xp for xp-ordering of covariance matrix or normal for the usual ordering")
        
   
    
def Omega(modes):
    om = np.array([[0,1],[-1,0]]) 
    OM = om
    for i in range(0,modes-1):
        
        OM = np.block([[OM, np.zeros((OM.shape[0], 2))], [np.zeros((2, OM.shape[0])), om]])
        
    return OM