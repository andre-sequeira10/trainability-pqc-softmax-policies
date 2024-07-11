import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import torch
from pennylane.pauli import PauliWord, pauli_word_to_matrix, string_to_pauli_word
from helper import compute_policy_and_gradient
from torch.nn.functional import softmax
import itertools

#command line arguments
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--initialization', type=str, default="random")
parser.add_argument('--measurement', type=str, default="global")

args = parser.parse_args()

initialization = args.initialization
p = args.measurement

def create_circuit(n_qubits,n_layers=None,circ = "simplified_two_design",fim=False, shots=None, observables=None):

    dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)


    
    def S2D(init_params,params):
        #qml.SpecialUnitary(params, wires=range(n_qubits))
        qml.SimplifiedTwoDesign(initial_layer_weights=init_params, weights=params, wires=range(n_qubits))
        
        return [qml.expval(o) for o in observables]


    def simmpleRZRY(params):
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        

        qml.broadcast(qml.RZ, wires=range(n_qubits), pattern="single", parameters=params[0])
        qml.broadcast(qml.RY, wires=range(n_qubits), pattern="single", parameters=params[1])

        qml.broadcast(qml.CZ, wires=range(n_qubits), pattern="all_to_all")
 
        return [qml.expval(o) for o in observables]
    
    if circ == "simplified_two_design":
        qcircuit = S2D
    elif circ == "simpleRZRY":
        qcircuit = simmpleRZRY
    
    circuit = qml.QNode(qcircuit, dev,interface="torch", diff_method="best")

    return circuit


EPISODES = 100
GAMMA = 0.95
LEARNING_RATE = 0.1
N_ACTIONS = 16
TRIALS = 30
#policy_type = "local"

# policy gradient algorithm to solve contextual bandit using simplified two design ansatz as the parameterized policy 
# Main part of the code
n_qubits = 20
n_layers = n_qubits

best_arm = 0

for t in range(TRIALS,TRIALS*2):
    if initialization == "random":
        weights_tensor = torch.tensor(np.random.uniform(-np.pi,np.pi,size=(2,n_qubits)), requires_grad=True)
    elif initialization == "normal":
        weights_tensor = torch.tensor(np.random.normal(0,0.01,size=(2,n_qubits)), requires_grad=True)
    #weights_tensor = torch.tensor(np.zeros((2,n_qubits)), requires_grad=True)
    observables = []
    #ZZ0 = qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(1))
    #observables.append(qml.Hamiltonian([1.0], [ZZ0]))
    if p == "global":
    
        #for a in range(N_ACTIONS-1):
            #ps = "".join(np.random.choice(["X", "Y", "Z"], int(np.log(n_qubits))))
            #observab
        ps = ["".join(comb) for comb in itertools.product(["X","Z"], repeat=n_qubits)][-N_ACTIONS:]

        pw = [string_to_pauli_word(pss, wire_map={i:i for i in range(n_qubits)}) for pss in ps]

        observables = [qml.Hamiltonian([1.0], [pword]) for pword in pw]

    elif p == "local":
        ps = ["".join(comb) for comb in itertools.product(["X","Z"], repeat=4)][-N_ACTIONS:]
        pw = [string_to_pauli_word(pss, wire_map={i:i for i in range(4)}) for pss in ps]
        observables = [qml.Hamiltonian([1.0], [pword]) for pword in pw]
    elif p == "partial":
        ps = ["".join(comb) for comb in itertools.product(["X","Z"], repeat=n_qubits)][-(N_ACTIONS-1):]

        pw = [string_to_pauli_word(pss, wire_map={i:i for i in range(n_qubits)}) for pss in ps]

        observables = [qml.Hamiltonian([1.0], [pw[0]]), qml.Hamiltonian([1.0], [qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2)@qml.PauliZ(3)])]
        observables += [qml.Hamiltonian([1.0], [pword]) for pword in pw[1:]]
    elif p == "partialbest":
        ps = ["".join(comb) for comb in itertools.product(["X","Z"], repeat=n_qubits)][-(N_ACTIONS-1):]

        pw = [string_to_pauli_word(pss, wire_map={i:i for i in range(n_qubits)}) for pss in ps]

        observables = [qml.Hamiltonian([1.0], [qml.PauliZ(0)@qml.PauliZ(1)@qml.PauliZ(2)@qml.PauliZ(3)]),qml.Hamiltonian([1.0], [pw[0]])]
        observables += [qml.Hamiltonian([1.0], [pword]) for pword in pw[1:]]



    qc = create_circuit(n_qubits,n_layers=1,circ="simpleRZRY",fim=False, observables=observables)

    opt = torch.optim.Adam([weights_tensor], lr=LEARNING_RATE, amsgrad=True)

    total_rewards = []
    best_arms = []
    probs_best_a = []
    gradient_norm= [] 
    for ep in range(EPISODES):
        # Initialize the state
        #linear schedule for beta
        beta = 10*ep/EPISODES

        opt.zero_grad()
            
        out = qc(weights_tensor)
        
        pi = softmax(beta*torch.stack(out), dim=0)


        policy__ = torch.clone(pi).detach().numpy()

        best_a = np.argmax(policy__)
        prob_best_a = policy__[best_arm]

        dist = torch.distributions.Categorical(probs=pi)
        
        cost = 0
        for i in range(50):
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_ = action.item()

            #log_prob = torch.log(policy_[best_arm])
            cost -= (1/(action_+0.01))*log_prob
        cost /= 50  
        #cost = -dist.log_prob(torch.tensor(best_arm))      
        # Update the probs\

        cost.backward(retain_graph=True)
        opt.step()
        #weights_tensor.data.add_(-LEARNING_RATE * weights_tensor.grad)
        #weights_tensor.grad.zero_()

        #total_rewards.append(reward)
        best_arms.append(best_a)
        probs_best_a.append(prob_best_a)
        #save gradient norm
        gradient_norm.append(np.linalg.norm(weights_tensor.grad.detach().numpy(),2))

        print("Episode: {}, best arm {} || prob best arm: {} || best_arm_measured: {} || gradient norm: {} || policy {}".format(ep+1, best_arm, probs_best_a[-1] ,best_a, gradient_norm[-1],policy__))

        # Update the policy
    np.save("gradient_norm_{}_{}_{}_{}.npy".format(p,n_qubits,initialization,t),gradient_norm)
    np.save("weights_{}_{}_{}_{}.npy".format(p,n_qubits,initialization,t),weights_tensor.detach().numpy())
    np.save("best_arms_{}_{}_{}_{}.npy".format(p,n_qubits,initialization,t),best_arms)
    np.save("probs_best_a_{}_{}_{}_{}.npy".format(p,n_qubits,initialization,t),probs_best_a)
