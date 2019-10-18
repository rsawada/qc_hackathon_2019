import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize, Bounds
from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit

from qcl_utils import create_time_evol_gate, min_max_scaling, softmax


class QclClassification:
    """ quantum circuit learning"""
    def __init__(self, nqubit, c_depth, num_class):
        """
        :param nqubit: #qubits
        :param c_depth: circuit depth
        :param num_class: # of classification (=# of measured qubits)
        """
        self.nqubit = nqubit
        self.c_depth = c_depth

        self.input_state_list = []  # list of |Psi_in>
        self.theta = []

        self.output_gate = None  # U_out

        self.num_class = num_class  # # of classification (=# of measured qubits)

        # Observable
        obs = [Observable(nqubit) for _ in range(num_class)]
        for i in range(len(obs)):
            obs[i].add_operator(1., 'Z '+str(i))  # Z0, Z1, Z3
        self.obs = obs

    def create_input_gate(self, x):
        # Encode x into quantum state
        # x = 2-dim. variables, [-1,1]
        u = QuantumCircuit(self.nqubit)
                
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)

        for i in range(self.nqubit):
            '''
            if i % 2 == 0:
                u.add_RY_gate(i, angle_y[0])
                u.add_RZ_gate(i, angle_z[0])
            else:
                u.add_RY_gate(i, angle_y[1])
                u.add_RZ_gate(i, angle_z[1])
            '''
            u.add_RY_gate(i, angle_y[i])
            u.add_RZ_gate(i, angle_z[i])
        return u

    def set_input_state(self, x_list):
        """List of input state"""
        x_list_normalized = min_max_scaling(x_list)  # x within [-1, 1]

        st_list = []
        
        for x in x_list_normalized:
            st = QuantumState(self.nqubit)
            input_gate = self.create_input_gate(x)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())
        self.input_state_list = st_list

    def create_initial_output_gate(self):
        """Output gate U_out and parameter setting"""
        u_out = ParametricQuantumCircuit(self.nqubit)
        time_evol_gate = create_time_evol_gate(self.nqubit)
        theta = 2.0 * np.pi * np.random.rand(self.c_depth, self.nqubit, 3)
        self.theta = theta.flatten()
        for d in range(self.c_depth):
            u_out.add_gate(time_evol_gate)
            for i in range(self.nqubit):
                u_out.add_parametric_RX_gate(i, theta[d, i, 0])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 1])
                u_out.add_parametric_RX_gate(i, theta[d, i, 2])
        self.output_gate = u_out
    
    def update_output_gate(self, theta):
        """Update U_out with parameter theta"""
        self.theta = theta
        parameter_count = len(self.theta)
        for i in range(parameter_count):
            self.output_gate.set_parameter(i, self.theta[i])

    def get_output_gate_parameter(self):
        """Get U_out parameter theta"""
        parameter_count = self.output_gate.get_parameter_count()
        theta = [self.output_gate.get_parameter(ind) for ind in range(parameter_count)]
        return np.array(theta)

    def pred(self, theta):
        """Get prediction for x_list"""

        # Prepare input state
        # st_list = self.input_state_list
        st_list = [st.copy() for st in self.input_state_list]  # Need to copy() for each element
        # Update U_out
        self.update_output_gate(theta)

        res = []
        # Output state and measurement
        for st in st_list:
            self.output_gate.update_quantum_state(st)
            r = [o.get_expectation_value(st) for o in self.obs]  
            r = softmax(r)
            res.append(r.tolist())
        return np.array(res)

    def cost_func(self, theta):
        """Cost function
        :param theta: List of rotation angle theta
        """

        y_pred = self.pred(theta)

        # cross-entropy loss
        loss = log_loss(self.y_list, y_pred)
        
        return loss

    # for BFGS
    def B_grad(self, theta):
        # Return the list of dB/dtheta
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]

        grad = [(self.pred(theta_plus[i]) - self.pred(theta_minus[i])) / 2. for i in range(len(theta))]

        return np.array(grad)

    # for BFGS
    def cost_func_grad(self, theta):
        y_minus_t = self.pred(theta) - self.y_list
        B_gr_list = self.B_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)

    def fit(self, x_list, y_list, maxiter=1000):
        """
        :param x_list: List of data x
        :param y_list: List of data y 
        :param maxiter: # of iterations in scipy.optimize.minimize
        :return: Value of loss function after training
        :return: Values of theta after training
        """

        # Input state
        self.set_input_state(x_list)

        # Make U_oiut
        self.create_initial_output_gate()
        theta_init = self.theta

        # True label
        self.y_list = y_list

        # for callbacks
        self.n_iter = 0
        self.maxiter = maxiter
        
        print("Initial parameter:")
        print(self.theta)
        print()
        print("Initial value of cost function:  ",self.cost_func(self.theta))
        print()
        print('============================================================')
        print("Iteration count...")
        '''
        bounds = Bounds(0.0, 2.0*np.pi)
        result = minimize(self.cost_func,
                          self.theta,
                          method='L-BFGS-B',
                          jac=self.cost_func_grad,
                          options={"disp":True,"maxiter":maxiter},
                          callback=self.callbackF,
                          bounds=bounds,
                          tol=1e-2)
        result = minimize(self.cost_func,
                          self.theta,
                          # method='Nelder-Mead',
                          method='BFGS',
                          jac=self.cost_func_grad,
                          options={"maxiter":maxiter},
                          callback=self.callbackF)
        '''
        result = minimize(self.cost_func,
                          self.theta,
                          method='COBYLA',
                          options={"disp":True,"maxiter":maxiter},
                          callback=self.callbackF)
        theta_opt = self.theta
        print('============================================================')
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print("Final value of cost function:  ",self.cost_func(self.theta))
        print()
        return result, theta_init, theta_opt

    def callbackF(self, theta):
            self.n_iter = self.n_iter + 1
            if 10 * self.n_iter % self.maxiter == 0:
                print("Iteration: ",self.n_iter / self.maxiter,",  Value of cost_func: ",self.cost_func(theta))


def main():
    # Random number
    random_seed = 0
    np.random.seed(random_seed)

    nqubit = 3 
    c_depth = 2 
    num_class = 3

    qcl = QclClassification(nqubit, c_depth, num_class)

    n_sample = 10
    x_list = np.random.rand(n_sample, 2)
    y_list = np.eye(num_class)[np.random.randint(num_class, size=(n_sample,))]

    qcl.fit(x_list, y_list)


if __name__ == "__main__":
    main()
