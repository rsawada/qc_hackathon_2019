import numpy as np
import matplotlib.pyplot as plt
import random
from functools import reduce
import pickle
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import X, Z
from qulacs import ParametricQuantumCircuit
from qulacs import Observable



######## Parameters #############
nqubit = 3 ## number of qubits
c_depth = 3 ## circuit depth

## Random seed
random_seed = 0
## Initialize the random generator
np.random.seed(random_seed)

#### Preprare teacher data
input_train = list()
value_train = list()
with open('Training.data', 'rb') as f:
    data = pickle.load(f)

state = QuantumState(nqubit)

for d in data:
    input_train.append(d[0])
    value_train.append(d[1])

## Basic gates
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()


# Construct an output gate U_out and initialization.
U_out = ParametricQuantumCircuit(nqubit)
for d in range(c_depth):
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


# Function to encode x
def U_in(x):
    U = QuantumCircuit(nqubit)

    angle_y = np.arcsin(x)
    angle_z = np.arccos(x**2)

    for i in range(nqubit):
        U.add_RY_gate(i, angle_y)
        U.add_RZ_gate(i, angle_z)

    return U

# Function to update theta
def set_U_out(theta):
    global U_out

    parameter_count = U_out.get_parameter_count()

    for i in range(parameter_count):
        U_out.set_parameter(i, theta[i])

# Construct an observable gate
obs = Observable(nqubit)
obs.add_operator(2.,'Z 0') # Add an operator 2*Z. This number should be optimised.

# Function for prediction
def qcl_pred(x, U_out):

    # Output state
    U_out.update_quantum_state(x)

    # Output from the model
    res = obs.get_expectation_value(state)

    return res

# cost function L
def cost_func(theta):
    '''
    theta: c_depth * nqubit * 3 ndarray
    '''
    # Update theta in U_out
    set_U_out(theta)

    # num_x_train
    value_pred = list()
    for x in input_train:
        state.set_zero_state() # U_in|000>
        U_in(x).update_quantum_state(state)
        value_pred.append(qcl_pred(state, U_out))

    # quadratic loss
    #L = ((value_pred - value_train)**2).mean()
    L = mean_squared_error(value_train, value_pred)

    return L


# Training
result = minimize(cost_func, theta_init, method='Nelder-Mead')

# cost function after training
#print(result.fun)

# theta after training
theta_opt = result.x
print(theta_opt)


# inference
n_data = 100
X = list()

# Put the optimized theta to U_out
set_U_out(theta_opt)

input_value = list()
for i in range(n_data):
    #state.set_Haar_random_state()
    x = random.random()
    input_value.append(x)
    state.set_zero_state() # U_in|000>
    U_in(x).update_quantum_state(state)
    X.append(qcl_pred(state, U_out))

plt.plot(input_value, X, 'o', color='black')
plt.show()
plt.savefig('qcl_regression.png')
