# All dense matrices
import numpy as np 
import scipy as sy
import functools as ft

def vec(rho): 
    # CONVERTS A MATRIX INTO A VECTOR
    vec_rho = rho.reshape((np.prod(rho.shape),1))
    return vec_rho

def d_vec(n, rho):                                                                       # n - number of qubits, rho = vector to be made into a matrix
    # CONVERTS A (NxN, 1) VECTOR INTO A NxN MATRIX 
    dvec_rho = rho.reshape((2**n,2**n))
    return dvec_rho

def effTemp(efft_args, t) :
    # DEFINES THE EFFECTIVE TEMPERATURE
    T0 = efft_args
    T = T0
    dTdt = 0
    return T, dTdt

def omegasine(o, t, rate_args, efft_args): 
    # COMPUTES THE DERIVATIVE OF THE DRIVING FREQUENCIES TO BE FED INTO THE SOLVER
    theta, l, Te = rate_args 
    no = (np.exp(o/Te) - 1)**(-1)
    gp = np.cos(theta)**2 * np.pi * l * o * no                                     # gamma_+
    gm = np.cos(theta)**2 * np.pi * l * o * (1 + no)                               # gamma_-
    T, dTdt = effTemp(efft_args, t)
    dodt = (o/T)*dTdt + T*(np.exp(o/T) + 1)*(gm * np.exp(-o/T) - gp)
    return dodt

def gr(o, theta, l, Te):                                                                 # o and theta are n-tuple, one value corresponding to one qubit at one time step, l and Te are single valued
    # COMPUTES THE DECAY RATES FOR EACH QUBIT AND PUTS THEM IN A LIST, COMPATIBLE WITH THE LINDBLAD
    gp, gz, gm = [], [], []
    for i in range(len(o)):
        no = (np.exp(o[i]/Te) - 1)**(-1)
        c = 0.5 * np.cos(theta[i])**2 * np.pi * l * o[i]
        gp.append(c * no)                                                                     # gamma_+ for each qubit 
        gm.append(c * (1 + no))                                                               # gamma_- for each qubit 
        gz.append(0.5 * np.sin(theta[i])**2 * np.pi * l * o[i] * (1/np.tanh(0.5*o[i]/Te)))    # gamma_z for each qubit 
    return gp, gm, gz                                                                         # gamma_+, gamma_-, gamma_z

def Lind_cons(n):                                                                          # n = no. of driven qubits, excluding the system qubit
    # CONSTRUCTS JUST THE MATRICES REQUIRED FOR THE HERMITIAN PART AND THE DISSIPATORS, NO SYSTEM QUBIT
    eye = np.identity(2)                                                                   # Code 1
    sz = np.array([[1,0],[0,-1]])                                                          # Code 2
    sp = np.array([[0,1],[0,0]])                                                           # Code 3
    sm = np.array([[0,0],[1,0]])                                                           # Code 4
    szz, spp, smm = [], [], []                                                             # The bits required for the Hamiltonian and the dissipators 
    lz, lp, lm = [], [], []
    if n == 1 :
        szz.append(sz)
        spp.append(sp)
        smm.append(sm)
    if n > 1 :
        lz, lp, lm = [2], [3], [4]                                                         # Using the codes here
        m = n - 1
        while m!= 0 :
            lz.append(1)
            lp.append(1)
            lm.append(1)
            m = m - 1

        for i in range(n):
            z_ops, p_ops, m_ops = [], [], []
            for j in range(n):
                if lz[j] == 1 : 
                    z_ops.append(eye)
                if lz[j] == 2 :
                    z_ops.append(sz)
            szz.append(ft.reduce(np.kron, z_ops))
            for j in range(n):
                if lp[j] == 1 : 
                    p_ops.append(eye)
                if lp[j] == 3 :
                    p_ops.append(sp)
            spp.append(ft.reduce(np.kron, p_ops))
            for j in range(n):
                if lm[j] == 1:
                    m_ops.append(eye)
                if lm[j] == 4:
                    m_ops.append(sm)
            smm.append(ft.reduce(np.kron, m_ops))
            lz = np.roll(lz, 1)
            lp = np.roll(lp, 1)
            lm = np.roll(lm, 1)
    return szz, spp, smm     

def dissVec(n):
    # CREATES THE DISSIPATORS IN THE VECTORIZED FORM
    dissp, dissm, dissz = [], [], []
    ham, spp, smm = Lind_cons(n)
    eyeT = np.identity(2**n)

    for i in spp:                                                                           # Heating dissipator in the vectorized form 
        d1 = np.kron(i, i) - 0.5*(np.kron(eyeT, i.T@i) + np.kron(i.T@i, eyeT))
        dissp.append(d1)
    
    for i in smm:                                                                           # Energy loss dissipator in the vectorised form
        d1 = np.kron(i, i) - 0.5*(np.kron(eyeT, i.T@i) + np.kron(i.T@i, eyeT))
        dissm.append(d1)
    
    for i in ham:                                                                           # Dephasing dissipator in the vectorized form
        d1 = np.kron(i, i) - 0.5*(np.kron(eyeT, i.T@i) + np.kron(i.T@i, eyeT))
        dissz.append(d1)
    
    return ham, dissp, dissm, dissz

def VecS(n): # n = 1 system qubit + (n - 1) driven ancilla qubits
    # CREATES THE DISSIPATORS IN THE VECTORIZED FORM PLUS , ADDS SYSTEM QUBIT, WORKS ON TOP OF Lind_cons(n)
    dissp, dissm, dissz = [], [], []
    szz, spp, smm = Lind_cons(n-1)
    eyeT = np.identity(2**n)

    sz = np.array([[1,0],[0,-1]])
    sp = np.array([[0,1],[0,0]])
    sm = np.array([[0,0],[1,0]])
    eye = np.identity(2)
    ham = []

    for i in range(n-1):
        szz[i] = np.kron(eye, szz[i])
        spp[i] = np.kron(eye, spp[i])
        smm[i] = np.kron(eye, smm[i])

    hz = [5] + [x for x in np.ones(n-1)]
    for i in range(n):
        h_ops = []
        for j in range(n):
            if hz[j] == 1:
                h_ops.append(eye)
            if hz[j] == 5:
                h_ops.append(sz)
        ham.append(ft.reduce(np.kron, h_ops))
        hz = np.roll(hz, 1)
    
    # THE INTERACTION TERM FOR THE TIME DEPENDENT 'SYSTEM' HAMILTONIAN
    hint = []                                                                                # Will contain the operators required for the system part of the interaction Hamiltonian
    sysp = ft.reduce(np.kron, [sp] + [np.identity(2) for i in range(n-1)])                   # Sigma_+ operator for the system
    sysm = ft.reduce(np.kron, [sm] + [np.identity(2) for i in range(n-1)])                   # Sigma_- operator for the system
    hint = [sysp] + [sysm]

    for i in spp:                                                                           # Heating dissipator in the vectorized form 
        d1 = np.kron(i, i) - 0.5*(np.kron(eyeT, i.T@i) + np.kron(i.T@i, eyeT))
        dissp.append(d1)
    
    for i in smm:                                                                           # Energy loss dissipator in the vectorised form
        d1 = np.kron(i, i) - 0.5*(np.kron(eyeT, i.T@i) + np.kron(i.T@i, eyeT))
        dissm.append(d1)
    
    for i in szz:                                                                           # Dephasing dissipator in the vectorized form
        d1 = np.kron(i, i) - 0.5*(np.kron(eyeT, i.T@i) + np.kron(i.T@i, eyeT))
        dissz.append(d1)
    return ham, hint, spp, smm, dissp, dissm, dissz

def LindVec(n, o, dr, lind_m):                                                               # n - no. of qubits, everything else is a tuple with n elements 
    # COMPUTES THE LIOUVILLIAN IN THE VECTORIZED FORM
    ham, dissp, dissm, dissz = lind_m
    eyeT = np.identity(2**n)

    gp, gm, gz = dr
    h = np.zeros((2**n, 2**n))
    dm = dp = dz = np.zeros((2**(2*n), 2**(2*n)))
    for i in range(n):
        h = 0.5 * o[i] * ham[i] + h                                                         # The Hamiltonian
        dm = gm[i] * dissm[i] + dm                                                          # The total energy loss dissipator for all the qubits
        dp = gp[i] * dissp[i] + dp                                                          # Total heating dissipator for all the qubits 
        dz = gz[i] * dissz[i] + dz                                                          # Total dephasing dissipator for all the qubits
    vham = np.kron(eyeT, h) - np.kron(h.T, eyeT)                                            # The hamiltonian commutator in the vectorized form
    lind = dm + dp + dz -1j * vham                                                          # The vectorized Liovillian L(t)
    return lind

def LindVecS(n, o, os, g, dr):
    # COMPUTES THE LIOUVILLIAN IN THE VECTORIZED FORM
    ham, hint, spp, smm, dissp, dissm, dissz = VecS(n)
    eyeT = np.identity(2**n)
    gp, gm, gz = dr
    h = np.zeros((2**n, 2**n))
    dm = dp = dz = np.zeros((2**(2*n), 2**(2*n)))
    h = 0.5 * os * ham[0]
    for i in range(n - 1):
        h = h + 0.5 * o[i] * ham[i + 1] + g[i] * (hint[0] @ smm[i] + hint[1] @ spp[i])      # The Hamiltonian
        dm = gm[i] * dissm[i] + dm                                                          # The total energy loss dissipator for all the qubits
        dp = gp[i] * dissp[i] + dp                                                          # Total heating dissipator for all the qubits 
        dz = gz[i] * dissz[i] + dz                                                          # Total dephasing dissipator for all the qubits
    vham = np.kron(eyeT, h) - np.kron(h.T, eyeT)                                            # The hamiltonian commutator in the vectorized form
    lind = dm + dp + dz -1j * vham                                                          # The vectorized Liovillian L(t)
    return lind

def LindwSys(n): # n = system qubit + driven qubit
    # ADDS THE SYSTEM QUBIT, WORKS ON TOP OF Lind_cons(n), NOT FOR THE VECTORIZED VERSION
    sz = np.array([[1,0],[0,-1]])
    sp = np.array([[0,1],[0,0]])
    sm = np.array([[0,0],[1,0]])

    szz, spp, smm = Lind_cons(n-1)
    eye = np.identity(2)
    ham = []
    for i in range(n-1):
        szz[i] = np.kron(eye, szz[i])
        spp[i] = np.kron(eye, spp[i])
        smm[i] = np.kron(eye, smm[i])
    
    hz = [5] + [x for x in np.ones(n-1)]
    for i in range(n):
        h_ops = []
        for j in range(n):
            if hz[j] == 1:
                h_ops.append(eye)
            if hz[j] == 5:
                h_ops.append(sz)
        ham.append(ft.reduce(np.kron, h_ops))
        hz = np.roll(hz, 1)
    # THE INTERACTION TERM FOR THE TIME DEPENDENT 'SYSTEM' HAMILTONIAN
    hint = []                                                                                # Will contain the operators required for the system part of the interaction Hamiltonian
    sysp = ft.reduce(np.kron, [sp] + [np.identity(2) for i in range(n-1)])                   # Sigma_+ operator for the system
    sysm = ft.reduce(np.kron, [sm] + [np.identity(2) for i in range(n-1)])                   # Sigma_- operator for the system
    hint = [sysp] + [sysm]
    return ham, hint, spp, smm, szz

def LindMrho(n, o, os, g, dr, cons, rho):
    # COMPUTES THE ACTION OF THE LIOUVILLIAN IN THE MATRIX FORM, CONTAINS THE SYSTEM QUBIT
    ham, hint, spp, smm, szz = cons
    dissp, dissm, dissz = [], [], []
    # g contains the strength of the interation between the system qubit and each ancilla qubit

    for i in range(n - 1):                                                                           
        d1 = spp[i] @ rho @ spp[i].T - 0.5 * (spp[i].T @ spp[i] @ rho + rho @ spp[i].T @ spp[i])
        dissp.append(d1)                                                                             # Heating dissipator in the matrix form 
        d2 = smm[i] @ rho @ smm[i].T - 0.5 * (smm[i].T @ smm[i] @ rho + rho @ smm[i].T @ smm[i])
        dissm.append(d2)                                                                             # Energy loss dissipator in the vectorised form
        d3 = szz[i] @ rho @ szz[i].T - 0.5 * (szz[i].T @ szz[i] @ rho + rho @ szz[i].T @ szz[i])
        dissz.append(d3)                                                                             # Dephasing dissipator in the vectorized form

    gp, gm, gz = dr
    h = np.zeros((2**n, 2**n))
    dm = dp = dz = np.zeros((2**n, 2**n))
    h = 0.5 * os * ham[0]
    for i in range(n - 1):
        h = h + 0.5 * o[i] * ham[i + 1] + g[i] * (hint[0] @ smm[i] + hint[1] @ spp[i])      # The first o[i] should be the system qubit frequency                                                          # The Hamiltonian
        dm = gm[i] * dissm[i] + dm                                                          # The total energy loss dissipator for all the qubits
        dp = gp[i] * dissp[i] + dp                                                          # Total heating dissipator for all the qubits
        dz = gz[i] * dissz[i] + dz    
                                                        
    vham = h @ rho - rho @ h                                                                # The hamiltonian commutator in the vectorized form
    lind = dm + dp + dz -1j * vham     
    return lind

def initalize(n, oi, T):                                                                    # n-number of qubits, oi-the initial frequencies as a list, T-initial effective temp
    # INITIALIZES THE QUBITS IN THE THERMAL STATE
    rl = [] 
    nTi = []                                                                                # Will contain all the individual initial density matrices
    for i in range(n):
        nTi.append(1/(np.exp(oi[i]/T) + 1))
        ri = np.array([[nTi[i], 0], [0, 1-nTi[i]]])
        rl.append(ri)
    rho_0 = ft.reduce(np.kron, rl)
    return rho_0, nTi

def rho_t(rho, n, o, dr, cons, dt):
    # TIME EVOLVES THE DENSITY MATRIX BY EXPONENTIATING THE VECTORIZED LINDBLAD 
    return sy.linalg.expm(LindVec(n, o, dr, cons)*dt)@rho

def apx_rho_t(rho, n, o, dr, cons, dt):
    # APPROXIMATING THE TIME EVOLUTION OF THE DENSITY MATRIX BY TAYLOR EXPANDING THE EXPONENT (VECTORIZED LINDBLAD)
    eye = np.identity(2**(2*n))
    return (eye + LindVec(n, o, dr, cons)*dt)@rho

'''
def qm(n, m):
    #n = number of qubits
    #m = a single qubit matrix
    
    #returns a multidimensional numpy array which contains the operators for each qubit
    #is dense
    eye = np.identity(2)
    part = np.zeros((n, 2, 2))
    part[0] = m
    part[1:] = eye
    m_list = [ft.reduce(np.kron, part)]
    for i in range(n - 1):
        part = np.roll(part, shift = 1, axis = 0)
        m_list.append(ft.reduce(np.kron, part))
    return m_list

def diss_matrices(m_list):
    #takes in a list of operators and return a list of the dissipators corresponding to said operators
    #in the dense form
    n = m_list[0].shape[0]
    eye_n = sy.sparse.identity(n)
    n = int(math.log2(n))
    print(n)
    dissipator = []
    
    for i in range(n):
        z = np.kron(m_list[i], m_list[i])  - 0.5 * (sy.sparse.kron(eye_n, m_list[i].T @ m_list[i]) + sy.sparse.kron(m_list[i].T @ m_list[i], eye_n))
        dissipator.append(z)
    
    return dissipator
'''