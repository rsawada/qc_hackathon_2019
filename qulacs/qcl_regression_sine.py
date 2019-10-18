import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

######## Parameters #############
nqubit = 3 ## number of qubits
c_depth = 3 ## circuit depth
time_step = 0.77  ## Time step for random Hamiltonian

## take num_x_train randomly from [x_min, x_max]
x_min = - 1.; x_max = 1.;
num_x_train = 50

## Function to learn
func_to_learn = lambda x: np.sin(x*np.pi)

## Random seed
random_seed = 0
## Initialize the random generator
np.random.seed(random_seed)

#### Preprare teacher data
x_train = x_min + (x_max - x_min) * np.random.rand(num_x_train)
y_train = func_to_learn(x_train)

# Add noise to data
mag_noise = 0.05
y_train = y_train + mag_noise * np.random.randn(num_x_train)

plt.plot(x_train, y_train, "o"); plt.show()

# Construct the input quantum state
from qulacs import QuantumState, QuantumCircuit

state = QuantumState(nqubit) # Initial state |000>
state.set_zero_state()
print(state.get_vector())

# Function to encode x
def U_in(x):
    U = QuantumCircuit(nqubit)

    angle_y = np.arcsin(x)
    angle_z = np.arccos(x**2)

    for i in range(nqubit):
        U.add_RY_gate(i, angle_y)
        U.add_RZ_gate(i, angle_z)

    return U

# Test an input state
# x = 0.1 # Arbitrary
# U_in(x).update_quantum_state(state) # U_in|000>
# print(state.get_vector())

## Basic gates
from qulacs.gate import X, Z
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()

## fullsize gate
def make_fullgate(list_SiteAndOperator, nqubit):
    '''
    Making a (2**nqubit, 2**nqubit) matrix
    from list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...],
    and inserting Identity, resulting
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    '''
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = [] ## 1-qubit gate array, reduced by np.kron
    cnt = 0
    for i in range(nqubit):
        if (i in list_Site):
            list_SingleGates.append( list_SiteAndOperator[cnt][1] )
            cnt += 1
        else: ## Put an identity if nothing at the site
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)

#### Operator for the time-evolution
ham = np.zeros((2**nqubit,2**nqubit), dtype = complex)
for i in range(nqubit): ## i runs 0 to nqubit-1
    Jx = -1. + 2.*np.random.rand() ## randm in -1~1
    ham += Jx * make_fullgate( [ [i, X_mat] ], nqubit)
    for j in range(i+1, nqubit):
        J_ij = -1. + 2.*np.random.rand()
        ham += J_ij * make_fullgate ([ [i, Z_mat], [j, Z_mat]], nqubit)

## Diagonize to make a time-evolution operator; H*P = P*D <-> H = P*D*P^dagger
diag, eigen_vecs = np.linalg.eigh(ham)
time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj()) # e^-iHT

# Convert it to a qulacs gate
from qulacs.gate import DenseMatrix
time_evol_gate = DenseMatrix([i for i in range(nqubit)], time_evol_op)

from qulacs import ParametricQuantumCircuit

# Construct an output gate U_out and initialization.
U_out = ParametricQuantumCircuit(nqubit)
for d in range(c_depth):
    U_out.add_gate(time_evol_gate)
    for i in range(nqubit):
        angle = 2.0 * np.pi * np.random.rand()
        U_out.add_parametric_RX_gate(i,angle)
        angle = 2.0 * np.pi * np.random.rand()
        U_out.add_parametric_RZ_gate(i,angle)
        angle = 2.0 * np.pi * np.random.rand()
        U_out.add_parametric_RX_gate(i,angle)

# Take the initial theta
parameter_count = U_out.get_parameter_count()
theta_init = [U_out.get_parameter(ind) for ind in range(parameter_count)]

# Function to update theta
def set_U_out(theta):
    global U_out

    parameter_count = U_out.get_parameter_count()

    for i in range(parameter_count):
        U_out.set_parameter(i, theta[i])

# Construct an observable gate
from qulacs import Observable
obs = Observable(nqubit)
obs.add_operator(2.,'Z 0') # Add an operator 2*Z. This number should be optimised.

# Function for prediction
def qcl_pred(x, U_out):
    state = QuantumState(nqubit)
    state.set_zero_state()

    # Input state
    U_in(x).update_quantum_state(state)

    # Output state
    U_out.update_quantum_state(state)

    # Output from the model
    res = obs.get_expectation_value(state)

    return res

# cost function L
def cost_func(theta):
    '''
    theta: c_depth * nqubit * 3 ndarray
    '''
    # Update theta in U_out
#     global U_out
    set_U_out(theta)

    # num_x_train
    y_pred = [qcl_pred(x, U_out) for x in x_train]

    # quadratic loss
    L = ((y_pred - y_train)**2).mean()

    return L

# Plot a graph with given theta
xlist = np.arange(x_min, x_max, 0.02)
y_init = [qcl_pred(x, U_out) for x in xlist]
plt.plot(xlist, y_init)

from scipy.optimize import minimize

# Training
result = minimize(cost_func, theta_init, method='Nelder-Mead')

# cost function after training
#print(result.fun)

# theta after training
theta_opt = result.x
#print(theta_opt)

# Put the optimized theta to U_out
set_U_out(theta_opt)

# Plot
plt.figure(figsize=(10, 6))

xlist = np.arange(x_min, x_max, 0.02)

# Training data
plt.plot(x_train, y_train, "o", label='Teacher')

# Plot with the initial theta
plt.plot(xlist, y_init, '--', label='Initial Model Prediction', c='gray')

# Plot prediction
y_pred = np.array([qcl_pred(x, U_out) for x in xlist])
plt.plot(xlist, y_pred, label='Final Model Prediction')

plt.legend()
plt.show()

plt.savefig('qcl_ml_regression.png')
