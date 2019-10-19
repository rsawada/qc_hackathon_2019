import numpy as np
import matplotlib.pyplot as plt
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge,Measurement
import pickle
import random

nqubit = 2
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
#circuit.add_H_gate(0)
#circuit.add_H_gate(1)
#circuit.add_H_gate(2)
circuit.add_RX_gate(0, 0.1)
circuit.add_RX_gate(1, 0.1)
#circuit.add_RX_gate(2, 0.1)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
meas0 = Measurement(0, 0)
circuit.add_gate(meas0)
meas1 = Measurement(1, 1)
circuit.add_gate(meas1)
#meas2 = Measurement(2, 2)
#circuit.add_gate(meas2)
print(circuit)

observable = Observable(nqubit)
observable.add_operator(1, "Z 0")
observable.add_operator(2, "Z 1")
#observable.add_operator(4, "Z 2")

data = list()
x_value = list()
y_value = list()

for i in range(n_data):
    #state.set_Haar_random_state()
    x = random.random()
    state.set_zero_state() # U_in|000>
    U_in(x).update_quantum_state(state)
    state_vec = state.get_vector()
    circuit.update_quantum_state(state)
    value = observable.get_expectation_value(state)
    #print(value)
#    data.append((state_vec, value))
    data.append((x, value))
    x_value.append(x)
    y_value.append(value)

with open('Training.data', 'wb') as f:
    pickle.dump(data, f)

plt.plot(x_value, y_value, 'o', color='black')
plt.show()
plt.savefig('qcl_training.png')
