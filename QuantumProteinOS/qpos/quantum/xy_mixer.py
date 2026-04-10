import pennylane as qml

class XYMixer:
    """
    Local XY-mixer preserving Hamming weight.
    Local topology per residue block:
    H_M^XY = (1/2) Σ_{i=0}^{N-1} Σ_{j=0}^{n-1} (X_in+j X_in+(j+1)%n + Y_in+j Y_in+(j+1)%n)
    """
    def __init__(self, N, n):
        self.N = N
        self.n = n
        
    def apply(self, beta):
        """Apply the XY mixing unitary with parameter beta."""
        for i in range(self.N):
            base_idx = i * self.n
            for j in range(self.n):
                q1 = base_idx + j
                q2 = base_idx + ((j + 1) % self.n)
                # Apply XY mix (using IsingXY gate in pennylane)
                qml.IsingXY(beta, wires=[q1, q2])
