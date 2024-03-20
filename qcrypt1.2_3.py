from qiskit.visualization import array_to_latex
from qiskit.quantum_info import Statevector, random_statevector
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit import QuantumCircuit
from qiskit.circuit.library import HGate, CXGate
import numpy as np

ket0 = [[1],[0]]
bra0 = [1, 0]
ket1 = [[0],[1]]
bra1 = [0, 1]
ketplus = [[1/np.sqrt(2)], [1/np.sqrt(2)]]
braplus = [1/np.sqrt(2), 1/np.sqrt(2)]

#Calculate the outer product of ket and bra 0
op_bra0 = Operator(bra0)
op_ket0 = Operator(ket0)
rho0 = op_bra0.tensor(op_ket0)

#Calculate the outer product of ket and bra 1
op_bra1 = Operator(bra1)
op_ket1 = Operator(ket1)
rho1 = op_bra1.tensor(op_ket1)

#Calculate the outer product of ket and bra plus
op_braplus = Operator(braplus)
op_ketplus = Operator(ketplus)
rhoplus = op_braplus.tensor(op_ketplus)

density_matrix = 1/3*(rho0 + rho1 +rhoplus)
print()
print("Density matrix:")
print(np.real(density_matrix))


X_gate = np.array([[0, 1], [1, 0]])
Y_gate = np.array([[0, -1j], [1j, 0]])
Z_gate = np.array([[1, 0], [0, -1]])


X_result = np.dot(X_gate, density_matrix)
Y_result = np.dot(Y_gate, density_matrix)
Z_result = np.real(np.dot(Z_gate, density_matrix))

print()
print("X*ψ:")
print(np.real(X_result))
print()
print("Y*ψ:")
print(Y_result)
print()
print("Z*ψ:")
print(np.real(Z_result))

print()
print("Calculate the coefficients: ")
print("X: ", np.real(X_result[0][0] + X_result[1][1]))
print("Y: ", np.real(Y_result[0][0] + Y_result[1][1]))
print("Z: ", np.real(Z_result[0][0] + Z_result[1][1]))


rank = np.linalg.matrix_rank(density_matrix)
print()
print("Rank of the density matrix:", rank)


rho_squared = np.dot(density_matrix, density_matrix)
    
trace_rho_squared = np.trace(rho_squared)

if np.isclose(trace_rho_squared, 1):
    print("The density matrix represents a pure state.")
else:
    print("The density matrix represents a non-pure state.")