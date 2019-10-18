import numpy as np
from math import pi
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import Pauli, state_fidelity, basis_state, process_fidelity 

from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


# ----------------
Nstep = 10
Nshot = 2000

g1  = 2.
g2  = 1.
g12 = 1.
# ----------------


gp = np.sign(g2-g1)*np.sqrt((g1-g2)**2+4*g12**2)
ga = 0.5*(g1+g2-gp)
gb = 0.5*(g1+g2+gp)

u = np.sqrt((g1-g2+gp)/(2*gp))
theta = np.abs(2*np.arcsin(-1*u))
#theta = np.abs(2*np.arccos(np.sqrt((1-u**2))))
#print('theta=',theta)


# N steps
epsilon = 0.001 # cut-off
theta_min = epsilon
theta_max = 1.  # according to Fig.2 top-left
dlogtheta_step = (np.log(theta_max)-np.log(theta_min))/Nstep  # constant log(delta_theta) in each emission 
theta_m = [] # emission angle
for i in range(Nstep+1):
    theta_m.append(np.exp(np.log(theta_max)-dlogtheta_step*i))
print('theta_m =',theta_m)
#print('log(theta_m) =',np.log(theta_m))
delta_a_theta_m = [] # Sudakov factor for fa
delta_b_theta_m = [] # Sudakov factor for fb
for i in range(Nstep):
    delta_a_theta_m.append(np.exp(-1*(theta_m[i]-theta_m[i+1])*ga**2))
    delta_b_theta_m.append(np.exp(-1*(theta_m[i]-theta_m[i+1])*gb**2))
#print('delta_a_theta_m =',delta_a_theta_m)
#print('delta_b_theta_m =',delta_b_theta_m)


def single_step(q_phi, q_f, theta_a, theta_b):

    qc_meas.x(q_f)

    qc_meas.ry(0.25*theta_a,q_phi)
    qc_meas.cx(q_f, q_phi)
    qc_meas.ry(-0.5*theta_a,q_phi)
    qc_meas.cx(q_f, q_phi)
    qc_meas.ry(0.25*theta_a,q_phi)

    qc_meas.x(q_f)

    qc_meas.ry(0.25*theta_b,q_phi)
    qc_meas.cx(q_f, q_phi)
    qc_meas.ry(-0.5*theta_b,q_phi)
    qc_meas.cx(q_f, q_phi)
    qc_meas.ry(0.25*theta_b,q_phi)


backend = BasicAer.get_backend('qasm_simulator')
#backend = BasicAer.get_backend('statevector_simulator')

qc_meas = QuantumCircuit(Nstep+1,Nstep+1)
qc_meas.ry(theta,0)
for i in range(Nstep):
    theta_a = 2*np.arccos(np.sqrt(delta_a_theta_m[i]))
    theta_b = 2*np.arccos(np.sqrt(delta_b_theta_m[i]))
    single_step(i+1, 0, theta_a, theta_b)
qc_meas.ry(-1*theta,0)
bits = [i for i in range(Nstep+1)]
qc_meas.measure(bits,bits)
#print(qc_meas)


result = execute(qc_meas, backend, shots=Nshot).result()
x = result.get_counts()
#print('x =',x)
n_f = 0
n_phi = [0 for i in range(Nstep)]
nemit_shot = [0 for i in range(Nstep+1)]
ntheta_sum_shot = [0 for i in range(10)]
first_emit_shot = [0 for i in range(Nstep)]

for key in x.keys():
    a = str(key)
    if a[Nstep] == '1': n_f += int(x[key])
    nemit = 0
    theta_sum = 0
    first = False
    first_i = 0
    for i in range(Nstep):
        if a[Nstep-1-i] == '1': 
            n_phi[i] += int(x[key])
            nemit += 1
            theta_sum += theta_m[i]
            if first == False: 
                first_i = i
                first = True
                
    nemit_shot[nemit] += int(x[key])
    first_emit_shot[first_i] += int(x[key])

    '''
    if nemit > 0: 
        itsum = int(np.log(theta_sum))
        if itsum > -10: 
            ntheta_sum_shot[itsum+9] += int(x[key])
    '''

print('first_emit_shot =',first_emit_shot)
    
print('N(f)                =',n_f)
print('N( phi 1 - phi',Nstep,') =',n_phi)
print('N( 0 -',Nstep,'emission) =',nemit_shot)
#print('N(theta_sum) =',ntheta_sum_shot)

hcolor = 'royalblue'
if g12 == 1: hcolor = 'salmon'


plt.bar(np.log(theta_m[:-1]),n_phi,color=hcolor)
plt.xlabel(r'log($\theta$)')
plt.ylabel('Entries')
#plt.show()
filename='theta_noint_'+str(Nstep)+'step_'+str(Nshot)+'shot.png'
if g12 == 1: filename='theta_int_'+str(Nstep)+'step_'+str(Nshot)+'shot.png'
plt.savefig(filename)
plt.close()

plt.bar(np.log(theta_m[:-1]),first_emit_shot,color=hcolor)
plt.xlabel(r'log($\theta_{max}$)')
plt.ylabel('Entries')
#plt.show()
filename='thetamax_noint_'+str(Nstep)+'step_'+str(Nshot)+'shot.png'
if g12 == 1: filename='thetamax_int_'+str(Nstep)+'step_'+str(Nshot)+'shot.png'
plt.savefig(filename)
plt.close()

xbin = [i for i in range(Nstep+1)]
plt.bar(xbin,nemit_shot,color=hcolor)
plt.xlabel('Number of emissions')
plt.ylabel('Entries')
#plt.show()
filename='nemis_noint_'+str(Nstep)+'step_'+str(Nshot)+'shot.png'
if g12 == 1: filename='nemis_int_'+str(Nstep)+'step_'+str(Nshot)+'shot.png'
plt.savefig(filename)
plt.close()


