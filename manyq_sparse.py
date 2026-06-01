# Sparse only, using arrays instead of matrices from scipy
import scipy as sy
import numpy as np
import functools as ft
        
def qm_sparse(n, m):
    '''
    
    Parameters
    ----------
    n : Integer
        number of qubits
    m : array of shape (2,2)
        single qubit density matrix

    Returns
    -------
    m_list : list
        Contains a list of the qubit operators constructed from the single qubit matrix of the type
        m kron I kron I ...., I kron m kron I kron I...., so on where I is a identity(2) matrix

    '''
    if n == 1:
        return [sy.sparse.csr_array(m)]
    
    eye = np.identity(2)
    part = np.zeros((n, 2, 2))
    part[0] = m
    part[1:] = eye
    m_list = [sy.sparse.csr_array(ft.reduce(np.kron, part))]
    for i in range(n - 1):
        part = np.roll(part, shift = 1, axis = 0)
        m_list.append(sy.sparse.csr_array(ft.reduce(np.kron, part)))
    return m_list

def kron_sys_ancilla(operator_sys, ancilla_list):
    '''
    Parameters
    ----------
    operator_sys : Sparse (2 X 2) matrix
        Operator for the system (e.g. identity, sigma_+, sigma_- etc)
    ancilla_list : list of sparse matrices
        Operators for the ancillas (e.g. sigma_z^1, sigma_z^2 for 2 ancillas)

    Returns
    -------
    a list of sparse matrices with operator_sys kronecker producted with ancilla_list
    '''
    return list(map(lambda x: sy.sparse.kron(operator_sys, x), ancilla_list))

def diss_sparse_V(mat_build):
    '''

    Parameters
    ----------
    mat_build : A list of sparse matrices 

    Returns
    -------
    dissipator : list of sparse matrices
        returns the dissipator corresponding to the input sparse matrices in the Vectorized format
     
    The rates corresponding to each dissipator will be operated later
    '''
    n = mat_build[0].shape[0]
    eye_n = sy.sparse.identity(n)
    dissipator = list(map(lambda i: sy.sparse.kron(i, i)  - 0.5 * (sy.sparse.kron(eye_n, i.T @ i) + sy.sparse.kron(i.T @ i, eye_n)), mat_build))
    return dissipator

''' 
The Vectorized implementation ******************************************************************************************************************
Dividing the vectorized Liouvillian into 2 parts : 
    one which takes care of the dissipators 
    the other which takes care of the Hermitian part
'''

def Lind_diss_V(o, diss_matrices, args): # FACTOR OF 2 HASN'T BEEN FIXED FOR THE VECTORIZED VERSIONS
    # FIX THE FACTOR OF 0.5 IN FRONT OF THE DECAY RATES
    '''

    Parameters
    ----------
    o : list of frequencies for one time step
        Data is stored in the form [ancilla_1, ancilla_2 ....]
    diss_matrices : List of sparse matrices
        Contains three different dissipators, sigma_plus, sigma_minus, sigma_z
    args : List
        Contains the arguments required for the calculation of the decay rates 

    Returns
    -------
    diss_all : one sparse matrix
        total dissipator for all the ancilla qubits

    '''
    theta, l, Te = args
    dissp, dissm, dissz = diss_matrices
    diss_all = sy.sparse.csr_array(np.zeros(dissp[0].shape))
    for i in range(len(o)):
        no = (np.exp(o[i]/Te) - 1)**(-1)
        c = np.pi * l * o[i]
        gp_i = np.cos(theta[i])**2 * c * no
        gm_i = np.cos(theta[i])**2 * c * (1 + no)
        #gz_i = np.sin(theta[i])**2 * c * (1/np.tanh(0.5*o[i]/Te))
        diss_all = gp_i * dissp[i] + gm_i * dissm[i] + diss_all #+ gz_i * dissz[i]
    return diss_all

def Lind_herm_V(o, g, os, ham, hint): #ham : 1st entry is for the system, the rest for the ancillas
    n = ham[0].shape[0]
    eye_n = sy.sparse.identity(n)
    ham_total = 0.5 * os * ham[0]
    h_int_p, h_int_m = hint
    for i in range(len(o)): 
        ham_total = 0.5 * o[i] * ham[i + 1] + g[i] * (h_int_p[i] + h_int_m[i]) + ham_total
    return sy.sparse.kron(eye_n, ham_total) - sy.sparse.kron(ham_total.T, eye_n)

def expm_sparse(lindV, dt, rho_t):
    return sy.sparse.linalg.expm(lindV * dt) @ rho_t

'''
The action implementation **********************************************************************************************************************
'''

def diss_action(op_list, rho):
    '''

    Parameters
    ----------
    op_list : list of sparse matrices
        A List of operators for the ancilla qubits because the system qubit is not dissipating
    rho : sparse matrix
        density matrix in the sparse format

    Returns
    -------
    list of sparse matrices after performing the function -- op rho(t) op.T - 0.5 *{op.T * op, rho(t)}

    '''
    disspator = lambda x: x @ rho @ x.T - 0.5 *(x.T @ x @ rho + rho @ x.T @ x)
    return list(map(disspator, op_list))

def commutator(H, rho):
    '''
    

    Parameters
    ----------
    H : a square matrix
    rho : also a square matrix
    
    Returns
    -------
    the commutator of H and rho which is also a square matrix

    '''
    return H @ rho - rho @ H

def Lind_action(o, rho, all_matrices, os, g, rate_args):
    theta, l, Te = rate_args
    sp_ops, sm_ops, sz_ops, h_int_p, h_int_m, ham = all_matrices
    
    sp_rho = diss_action(sp_ops, rho)
    sm_rho = diss_action(sm_ops, rho)
    #sz_rho = diss_action(sz_ops, rho)
    
    ham_total = 0.5 * os * ham[0]
    dissipator_total = sy.sparse.csr_array(np.zeros(sp_ops[0].shape))
    
    for i in range(len(o)):
        no = (np.exp(o[i]/Te) - 1)**(-1) 
        c = np.pi * l * o[i]
        gp_i = np.cos(theta[i])**2 * c * no
        gm_i = np.cos(theta[i])**2 * c * (1 + no)
        #gz_i = np.sin(theta[i])**2 * c * (1/np.tanh(0.5*o[i]/Te))
        #dissipator_total = gp_i * sp_rho[i] + gm_i * sm_rho[i] + gz_i * sz_rho[i] + dissipator_total
        dissipator_total = gp_i * sp_rho[i] + gm_i * sm_rho[i] + dissipator_total
        ham_total = 0.5 * o[i] * ham[i + 1] + g[i] * (h_int_p[i] + h_int_m[i]) + ham_total 
    
    herm_part = commutator(ham_total, rho)

    return -1j * herm_part + dissipator_total

def Lind_action_with_sys_bath(o, rho, all_matrices, os, g, rate_args):
    # Variant of Lind_action where the system qubit also exchanges energy
    # directly with the bath (in addition to colliding with the ancillas),
    # used for the referee-response runs.
    #
    # Differences vs Lind_action:
    #   - rate_args carries an extra entry g_sys = system-bath coupling
    #   - sp_ops/sm_ops/sz_ops are length-n full-Hilbert-space operators
    #     with index 0 = system qubit and indices 1..n-1 = ancillas
    #     (Lind_action expects length-(n-1) ancilla-only operators padded
    #     with kron_sys_ancilla)
    #   - the ancilla dissipator loop reads sp_rho[i+1]/sm_rho[i+1] to skip
    #     the system slot
    #   - an extra bosonic dissipator on the system qubit (sp_rho[0], sm_rho[0])
    #     with rates n_sys*g_sys and (1+n_sys)*g_sys is added to the result
    theta, l, Te, g_sys = rate_args
    sp_ops, sm_ops, sz_ops, h_int_p, h_int_m, ham = all_matrices

    sp_rho = diss_action(sp_ops, rho)
    sm_rho = diss_action(sm_ops, rho)

    ham_total = 0.5 * os * ham[0]
    dissipator_total = sy.sparse.csr_array(np.zeros(sp_ops[0].shape))

    for i in range(len(o)):
        no = (np.exp(o[i]/Te) - 1)**(-1)
        c = np.pi * l * o[i]
        gp_i = np.cos(theta[i])**2 * c * no
        gm_i = np.cos(theta[i])**2 * c * (1 + no)
        dissipator_total = gp_i * sp_rho[i + 1] + gm_i * sm_rho[i + 1] + dissipator_total
        ham_total = 0.5 * o[i] * ham[i + 1] + g[i] * (h_int_p[i] + h_int_m[i]) + ham_total

    n_sys = (np.exp(os/Te) - 1)**(-1)
    diss_sys = g_sys * ((1 + n_sys) * sm_rho[0] + n_sys * sp_rho[0])

    herm_part = commutator(ham_total, rho)

    return -1j * herm_part + dissipator_total + diss_sys
