from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import Y,CNOT,merge
import pickle

state = QuantumState(3)

n_data = 100

circuit = QuantumCircuit(3)
circuit.add_X_gate(0)
merged_gate = merge(CNOT(0,1),Y(1))
circuit.add_gate(merged_gate)
circuit.add_RX_gate(1,0.5)

observable = Observable(3)
observable.add_operator(2.0, "X 2 Y 1 Z 0")
observable.add_operator(-3.0, "Z 2")

data = list()

for i in range(n_data):
    state.set_Haar_random_state()
    state_vec = state.get_vector()
    circuit.update_quantum_state(state)
    value = observable.get_expectation_value(state)
    data.append((state_vec, value))

with open('Training.data', 'wb') as f:
    pickle.dump(data, f)

