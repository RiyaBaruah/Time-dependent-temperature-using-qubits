import manyq_sparse as manqsp
import manyqfc as manqdense
import numpy as np
import scipy as sy
import matplotlib.pyplot as plt
import qutip as qp
import time

def effTemp(t, efft_args) :
    # EFFECTIVE TEMPERATURE HERE IS A PULSE
    a, b, c, baseline = efft_args
    T = a * np.exp(-(t - b)**2/(2 * c**2)) - a * np.exp(-(t - b - c)**2/(2 * c**2)) + baseline
    dTdt = - ((t - b)/c**2)*a * np.exp(-(t - b)**2/(2 * c**2)) + ((t - b - c)/c**2)*a * np.exp(-(t - b - c)**2/(2 * c**2)) 
    return T, dTdt

def omegasine(o, t, rate_args, efft_args): 
    # COMPUTES THE DERIVATIVE OF THE DRIVING FREQUENCIES TO BE FED INTO THE SOLVER
    theta, l, Te = rate_args
    no = (np.exp(o/Te) - 1)**(-1)                               # bath is Bosonic
    gp = np.pi * l * o * no                                     # gamma_+
    gm = np.pi * l * o * (1 + no)                               # gamma_-
    T, dTdt = effTemp(t, efft_args)
    dodt = (o/T)*dTdt + T*(np.exp(o/T) + 1)*(gm * np.exp(-o/T) - gp)
    return dodt

''' 
The Pauli matrices
'''
sp_pauli = np.array([[0,1],[0,0]])
sm_pauli = np.array([[0,0],[1,0]])
sz_pauli = np.array([[1,0],[0,-1]])

n = 8 # the total number of qubits, (system + ancilla)
''' 
1. Making the operators for the ancilla qubits 
'''
sz_list = manqsp.qm_sparse(n-1, sz_pauli)
sp_list = manqsp.qm_sparse(n-1, sp_pauli)
sm_list = manqsp.qm_sparse(n-1, sm_pauli)

''' 
2. Making Hilbert space for the system qubit 
First, implementing identity(2) kron (list of ancilla operators)
'''

sp_w_sys = manqsp.kron_sys_ancilla(sy.sparse.identity(2), sp_list)
sm_w_sys = manqsp.kron_sys_ancilla(sy.sparse.identity(2), sm_list)
sz_w_sys = manqsp.kron_sys_ancilla(sy.sparse.identity(2), sz_list)

'''
3. Generating the matrices required for the isolated hamiltonians, system kron. ancilla_1 kron. ancilla_2 ....
'''
ham = manqsp.qm_sparse(n, sz_pauli)

'''
4. Generating the interaction operators between the system and the ancillas i.e.
sigma(+)^(sys) kron (sigma(-)^(ancilla)) and sigma(-)^(sys) kron (sigma(+)^(ancilla)) 
'''     

hint_plus = manqsp.kron_sys_ancilla(sy.sparse.csr_array(sp_pauli), sm_list)
hint_minus = manqsp.kron_sys_ancilla(sy.sparse.csr_array(sm_pauli), sp_list)

'''
5. Putting the values of the constants and generating the frequencies of the ancilla qubits
'''

#l, Te = 0.0001, 0.5 # values for 0 environment, backflow
l, Te = 0.01, 1.5
theta = np.zeros(n - 1)
omi = np.linspace(0.75, 1.25, n - 1)
rate_args = [theta, l, Te]

omega = 0.3
dt = 0.05 # smaller time steps for larger frequencies since they have a very short time period
t = np.arange(0, 100, dt)
amp = Te
func_amp = 0.05
efft_args = [0.6, 20, 4, Te]
o = sy.integrate.odeint(omegasine, omi, t, args = (rate_args, efft_args))

o_system = 1

g = (0.4/(n-1)) * np.ones(n - 1)
dimq = [[2] * n, [2] * n]
'''
The initialization process
'''
T0 = effTemp(t[0], efft_args)[0]
T = manqdense.effTemp(T0, t[0])[0] # intital value of the effective temperature
initial = manqdense.initalize(n - 1, omi, T) # the initial density matrix for the 3 qubits 
rho_initial = initial[0]
nTe_sys = 1/(np.exp(o_system/(Te)) + 1)
rho_sys = np.array([[nTe_sys, 0], [0, 1-nTe_sys]])
rho_is = np.kron(rho_sys, rho_initial)
rho = sy.sparse.csr_array(rho_is) 

tr = np.zeros(len(t))
tr2 = np.zeros(len(t))
tr[0] = np.trace(rho_is)
tr2[0] = np.trace(rho_is@rho_is)

pop = np.zeros((len(t), n))
pop[0, 0] = rho_sys[0, 0]
pop[0, 1:] = initial[1]

all_matrices = [sp_w_sys, sm_w_sys, sz_w_sys, hint_plus, hint_minus, ham]
for i in range(len(t) - 1):
    strt_time = time.time()

    L_rho = manqsp.Lind_action(o[i+1, :], rho, all_matrices, o_system, g, rate_args)
    L2_rho = manqsp.Lind_action(o[i+1, :], L_rho, all_matrices, o_system, g, rate_args)
    L3_rho = manqsp.Lind_action(o[i+1, :], L2_rho, all_matrices, o_system, g, rate_args)
    L4_rho = manqsp.Lind_action(o[i+1, :], L3_rho, all_matrices, o_system, g, rate_args)
    L5_rho = manqsp.Lind_action(o[i+1, :], L4_rho, all_matrices, o_system, g, rate_args)
    rho = rho + dt * L_rho + 0.5 * (dt)**2 * L2_rho + 0.25 * (dt)**3 * L3_rho + 0.125 * (dt)**4 * L4_rho + (0.5 * dt)**5 * L5_rho
    
    rm = rho.toarray()
    
    tr[i + 1] = np.trace(rm).real
    tr2[i + 1] = np.trace(rm@rm).real
    q_rho = qp.Qobj(rm, dims = dimq)
    for j in range(n):
        #print(q_rho.ptrace(j)[0,1].real)
        pop[i + 1, j] = q_rho.ptrace(j)[0,0].real

temp_ancillas = np.zeros((len(t), n))
for i in range(len(t)):
    temp_ancillas[i, 0] = o_system/np.log((1/pop[i, 0]) - 1)
    for j in range(len(o[i])):
        temp_ancillas[i, j + 1] = o[i, j]/np.log((1/pop[i, j + 1]) - 1)

eff_temp = np.zeros(len(t))
for i in range(len(t)):
    eff_temp[i] = effTemp(t[i], efft_args)[0]

with open('frequencies_pulse_pt4.npy', 'wb') as f:
    np.save(f, t)
    np.save(f, o)

with open('temperatures_pulse_pt4.npy', 'wb') as g:
    np.save(g, eff_temp)
    np.save(g, temp_ancillas[:, 0])
    np.save(g, Te * np.ones(len(t)))

#plt.rcParams.update({'font.size': 20})
fig, axs = plt.subplots(2,1,figsize=(13.3, 10), sharex = True, gridspec_kw={'height_ratios': [1, 2]})
fig.subplots_adjust(wspace=0, hspace=0.1)
axs[0].plot(t, o, color = 'darkorange', linewidth = 2.5)
axs[0].plot(t, o_system * np.ones(len(t)), color = 'royalblue', linewidth = 2.5, label = 's_f')
axs[0].set_ylabel('$\omega_j(t)/\omega_s$')
axs[0].legend()

axs[1].plot(t, eff_temp, color ='k', label = 'effective_temp')
axs[1].plot(t, temp_ancillas[:, 0], color = 'royalblue', linewidth = 2.5, label = 'system_qubit_temp')
axs[1].plot(t, Te * np.ones(len(t)), color = 'red', label = 'Te')
plt.plot(t, temp_ancillas[:, 1], alpha = 0.5, label = 'ancilla1_temp')
plt.plot(t, temp_ancillas[:, 2], alpha = 0.5, label = 'ancilla2_temp')
plt.plot(t, temp_ancillas[:, 3], alpha = 0.5, label = 'ancilla3_temp')
plt.plot(t, temp_ancillas[:, 4], alpha = 0.5, label = 'ancilla4_temp')
#plt.plot(t, temp_ancillas[:, 5], alpha = 0.5, label = 'ancilla5_temp')
#plt.plot(t, temp_ancillas[:, 6], alpha = 0.5, label = 'ancilla6_temp')
#axs[1].set_ylabel('$T(t)/\omega_s$')
axs[1].set_xlabel('$\omega_s t$')
axs[1].legend()

plt.tight_layout()
plt.show()
#plt.savefig('images\with_sys\cosine_eff_temp_low_freq.png', dpi = 150)

    

