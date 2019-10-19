import numpy as np
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge
import pickle
import random

nqubit = 3
state = QuantumState(nqubit)

n_data = 100

# Function to encode x
def U_in(x):
    U = QuantumCircuit(nqubit)

    angle_y = np.arcsin(x)
    angle_z = np.arccos(x**2)

    for i in range(nqubit):
        U.add_RY_gate(i, angle_y)
        U.add_RZ_gate(i, angle_z)

    return U

circuit = QuantumCircuit(nqubit)
circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1,0.5)

observable = Observable(nqubit)
observable.add_operator(2.0, "X 2 Y 1 Z 0")
observable.add_operator(-3.0, "Z 2")

data = list()

for i in range(n_data):
    #state.set_Haar_random_state()
    x = random.random()
    state.set_zero_state() # U_in|000>
    U_in(x).update_quantum_state(state)
    state_vec = state.get_vector()
    circuit.update_quantum_state(state)
    value = observable.get_expectation_value(state)
#    data.append((state_vec, value))
    data.append((x, value))

with open('Training.data', 'wb') as f:
    pickle.dump(data, f)

