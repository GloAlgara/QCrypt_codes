import numpy as np

# Define the state vector |ψ⟩
psi = np.array([1/np.sqrt(2), np.exp(1j*np.pi/4)/np.sqrt(2)])

# Calculate the outer product |ψ⟩⟨ψ|
density_matrix = np.outer(psi, np.conj(psi))

print("|ψ⟩⟨ψ|:")
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
print("Z: ", np.round(Z_result[0][0]) + np.round(Z_result[1][1]))


rank = np.linalg.matrix_rank(density_matrix)
print()
print("Rank of the density matrix:", rank)


rho_squared = np.dot(density_matrix, density_matrix)
    
trace_rho_squared = np.trace(rho_squared)

if np.isclose(trace_rho_squared, 1):
    print("The density matrix represents a pure state.")
else:
    print("The density matrix represents a non-pure state.")
