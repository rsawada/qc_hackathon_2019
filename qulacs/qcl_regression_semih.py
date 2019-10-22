import numpy as np
import matplotlib.pyplot as plt
import random
from functools import reduce
import pickle
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error,mean_squared_log_error
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import X, Z, Measurement
from qulacs import ParametricQuantumCircuit
from qulacs import Observable



######## Parameters #############
nqubit = 2 ## number of qubits
c_depth = 3 ## circuit depth

## Random seed
random_seed = 0
## Initialize the random generator
np.random.seed(random_seed)

#### Preprare teacher data
input_train = list()
value_train = list()

input_test = list()
value_test = list()

with open('Training.data', 'rb') as f:
    data = pickle.load(f)


training_data = data[0:80]
test_data = data[80:100]

f = open('loss2.txt',"w" )

print(training_data[0])
state = QuantumState(nqubit)

for d in data:
    input_train.append(d[0] )
    value_train.append(d[1])

# To have a basic training and cross validation set
for d in training_data:
    input_train.append(d[0])
    value_train.append(d[1])

for d in test_data:
    input_test.append(d[0])
    value_test.append(d[1])


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
meas0 = Measurement(0, 0)
U_out.add_gate(meas0)
meas1 = Measurement(1, 1)
U_out.add_gate(meas1)
#meas2 = Measurement(2, 2)
#U_out.add_gate(meas2)

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
obs.add_operator(1, "Z 0")
obs.add_operator(2, "Z 1")
#obs.add_operator(4, "Z 2")

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



# cost function L
def iterative_cost_func():

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

# Function to learn theta
def update_U_out(delta_theta, alpha=0.0001 ):
    parameter_count = U_out.get_parameter_count()
    # Follow the pattern in qulacs lib to be on safe side
    new_theta =  list()
    #theta = U_out.get_parameter()
    for i in range(parameter_count):
        U_out.set_parameter(i,(-1.0 * alpha  * delta_theta + U_out.get_parameter(i)) )

def iterative_method( theta, epochs=100):
    set_U_out(theta)
    for i in range(epochs):
        loss = iterative_cost_func()
        if i % 10 == 0:
            f.write(str(loss) + ',')
        if i % 40 == 0:
            f.write('\n')
        update_U_out(loss)

# Uncomment to use backpropagation
#iterative_method(theta_init, epochs=1000)


# Training
result = minimize(cost_func, theta_init, method='COBYLA')

# cost function after training
#print(result.fun)

# theta after training
theta_opt = result.x
print(theta_opt)


# inference over test data only
n_data = 20
X = list()

# Put the optimized theta to U_out
set_U_out(theta_opt)

input_value = list()
for i in range(n_data):
    state.set_Haar_random_state()
    #x = random.random()
    #input_value.append(x)
    state.set_zero_state() # U_in|000>
    # Input test data only 
    U_in(input_test[i]).update_quantum_state(state)
    .append(qcl_pred(state, U_out))
    
plt.plot(input_value, X, 'o', color='black')
plt.show()
plt.savefig('qcl_regression.png')


