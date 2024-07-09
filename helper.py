import numpy as np
import pennylane as qml 
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

def create_circuit(n_qubits,circ = "simplified_two_design", shots=None, observables=None, policy="softmax"):

    if policy == "parity-like" or policy == "mixed":
        dev = qml.device("lightning.qubit", wires=n_qubits+1, shots=shots)
    else:
        dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)

    def S2D(init_params,params):
        #qml.SpecialUnitary(params, wires=range(n_qubits))
        qml.SimplifiedTwoDesign(initial_layer_weights=init_params, weights=params, wires=range(n_qubits))
        
        if policy == "softmax":
            return [qml.expval(o) for o in observables]
        elif policy == "parity-like":
            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="chain")
            return qml.probs(wires=n_qubits-1)
        elif policy == "mixed":
            qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="chain")
            return [qml.expval(observables[0]), qml.probs(wires=n_qubits-1)]

    def SEL(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_qubits))
        if policy == "softmax":
            return [qml.expval(o) for o in observables]
        elif policy == "parity-like":
            qml.broadcast(qml.CNOT, wires=range(n_qubits+1), pattern="chain")
            return qml.probs(wires=n_qubits)
        elif policy == "mixed":
            qml.broadcast(qml.CNOT, wires=range(n_qubits+1), pattern="chain")
            return [qml.expval(observables[0]), qml.probs(wires=n_qubits)]
        
    def RandomLayers(params):
        #qml.RandomLayers(params, wires=range(n_qubits),ratio_imprim=0.5)
        qml.broadcast(qml.Hadamard, wires=range(n_qubits), pattern="single")
        r = np.random.choice(["Z","X","Y"], size=(n_qubits))
        for i in range(n_qubits):
            if r[i] == "Z":
                qml.RZ(params[i], wires=i)
            elif r[i] == "X":
                qml.RX(params[i], wires=i)
            elif r[i] == "Y":
                qml.RY(params[i], wires=i)
        qml.broadcast(qml.CZ, wires=range(n_qubits), pattern="chain")
        if policy == "softmax":
            return [qml.expval(o) for o in observables]
        elif policy == "parity-like":
            qml.broadcast(qml.CNOT, wires=range(n_qubits+1), pattern="chain")
            return qml.probs(wires=n_qubits)
        elif policy == "mixed":
            qml.broadcast(qml.CNOT, wires=range(n_qubits+1), pattern="chain")
            return [qml.expval(observables[0]), qml.probs(wires=n_qubits)]
    
    if circ == "simplified_two_design":
        qcircuit = S2D
    elif circ == "SEL":
        qcircuit = SEL
    elif circ == "random":
        qcircuit = RandomLayers
    
    circuit = qml.QNode(qcircuit, dev, interface="torch", diff_method="best")

    return circuit

def compute_gradient(log_prob, w,n_qubits):
    """Compute gradient of the log probability with respect to weights.
    
    Args:
    - log_prob (torch.Tensor): The log probability tensor.
    - w (torch.Tensor): The weights tensor, with requires_grad=True.

    Returns:
    - numpy.ndarray: The gradient of log_prob with respect to w, flattened.
    """
    if w.grad is not None:
        w.grad.zero_()
    log_prob.backward(retain_graph=True)
    
    if w.grad is None:
        raise ValueError("The gradient for the given log_prob with respect to w is None.")
    
    w_grad = w.grad.view(-1).clone().detach().numpy()
    return w_grad
  
    
def compute_policy_and_gradient(args):
    n_qubits, shapes, ansatz , n_actions, observables, initialization, n_layers, policy = args


    #qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy=policy)

    if initialization == "random":
        weights = [np.random.uniform(-np.pi,np.pi,size=shape) for shape in shapes] 
    elif initialization == "normal":
        weights = [np.random.normal(0,1/n_layers,size=shape) for shape in shapes]

    if ansatz == "simplified_two_design":
        #weights = [np.random.uniform(-np.pi,np.pi,size=shape) for shape in shapes]    
        weights_tensor_init = torch.tensor(weights[0], requires_grad=False)
        weights_tensor_params = torch.tensor(weights[1], requires_grad=True)
        
        if policy == "softmax":
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="softmax")
            out = qc(weights_tensor_init,weights_tensor_params)
        elif policy == "parity-like":
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="parity-like")
            out = qc(weights_tensor_init,weights_tensor_params)
        elif policy == "mixed":
            '''
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="softmax")
            out1 = qc(weights_tensor_init,weights_tensor_params)
            out1_1d = torch.tensor([out1[0].item()], dtype=torch.float64, requires_grad=True)
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="parity-like")
            out2 = qc(weights_tensor_init,weights_tensor_params)
            out = torch.cat((out1_1d,out2))
            '''
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="mixed")
            out_ = qc(weights_tensor_init,weights_tensor_params)
            out_[0] = out_[0].unsqueeze(0)

            # Now stack the tensors
            out = torch.cat(out_)

    elif ansatz == "SEL" or ansatz == "random":
        
        if initialization == "random":
            weights = np.random.uniform(-np.pi,np.pi,size=shapes)
        elif initialization == "normal":        #weights = [np.random.uniform(-np.pi,np.pi,size=shape) for shape in shapes]   
            weights = np.random.normal(0,1/n_layers,size=shapes)

        weights_tensor_params = torch.tensor(weights, requires_grad=True)
        if policy == "softmax":
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="softmax")
            out = qc(weights_tensor_params)
        elif policy == "parity-like":
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="parity-like")
            out = qc(weights_tensor_params)
        elif policy == "mixed":
            qc = create_circuit(n_qubits, circ=ansatz, shots=None, observables=observables, policy="mixed")
            out_ = qc(weights_tensor_params)
            out_[0] = out_[0].unsqueeze(0)

            # Now stack the tensors
            out = torch.cat(out_)
    if policy == "parity-like" or policy == "mixed":

        pi = F.softmax(out, dim=0)
    elif policy == "softmax":
        pi = F.softmax(torch.stack(out), dim=0)
     
    dist = torch.distributions.Categorical(probs=pi)
    
    action = dist.sample()
    log_prob = dist.log_prob(action)

    #gradient_no_clamp = np.linalg.norm(compute_gradient(log_prob, weights_tensor_params,n_qubits), 2)
    gradient_no_clamp = compute_gradient(log_prob, weights_tensor_params,n_qubits)

    return gradient_no_clamp


