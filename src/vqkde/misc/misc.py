import numpy as np

def _indices_qubits_classes(num_qubits_param, num_classes_param):
    num_qubits_classes_temp = int(np.ceil(np.log2(num_classes_param)))
    a = [np.binary_repr(i, num_qubits_param) for i in range(2**num_qubits_param)]
    b = [(np.binary_repr(i, num_qubits_classes_temp) + "0"*(num_qubits_param - num_qubits_classes_temp)) for i in range(num_classes_param)]
    indices_temp = []
    for i in range(len(a)):
        if a[i] in b:
            indices_temp.append(i)

    return indices_temp