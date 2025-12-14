import cirq
import quimb.tensor as qtn
import numpy as np

class SupremacyBuilder:
    """
    Constructs a Tensor Network from a Cirq circuit specifically for 
    Sycamore L=105 architectures[cite: 77].
    """
    def __init__(self, qubits: list[cirq.GridQubit]):
        self.qubits = sorted(qubits)
        self.q_map = {q: i for i, q in enumerate(self.qubits)}
        self.tn = qtn.TensorNetwork()
        self.indices = {q: f"k{i}_0" for i, q in enumerate(self.qubits)}
        self.layer_count = 0

    def apply_circuit(self, circuit: cirq.Circuit):
        for moment in circuit:
            for op in moment:
                self._process_op(op)
            self.layer_count += 1

    def _process_op(self, op: cirq.Operation):
        gate = op.gate
        qubits = op.qubits
        
        # Identify Gate Type and Construct Unitary [cite: 79]
        if isinstance(gate, cirq.PhasedFSimGate):
            u_mat = cirq.unitary(gate)
            tags = {f'L{self.layer_count}', 'FSIM', '2Q'}
            self._add_tensor(u_mat, qubits, tags)
        elif len(qubits) == 1:
            u_mat = cirq.unitary(gate)
            tags = {f'L{self.layer_count}', '1Q'}
            self._add_tensor(u_mat, qubits, tags)

    def _add_tensor(self, data, qubits, tags):
        inds = []
        for q in qubits:
            inds.append(self.indices[q])
            
        new_inds = []
        for q in qubits:
            new_id = f"k{self.q_map[q]}_{self.layer_count + 1}"
            new_inds.append(new_id)
            self.indices[q] = new_id # Update current leg
            
        all_inds = inds + new_inds
        # Reshape data to match tensor indices (inputs, outputs)
        T = qtn.Tensor(data=data.reshape(2, 2, 2, 2), inds=all_inds, tags=tags)
        self.tn.add_tensor(T)

    def run_supremacy_xeb(self, circuit, target_bitstrings, chi_max=4096):
        # Initialize MPS in |0...0> state [cite: 104]
        mps = qtn.CircuitMPS(self.qubits, max_bond=chi_max)
        
        # Evolve layer by layer
        for i, moment in enumerate(circuit):
            mps.apply_circuit(cirq.Circuit(moment))
            if i % 5 == 0:
                entropy = mps.entropy(len(mps.sites)//2)
                print(f"Layer {i}: Half-chain Entropy = {entropy:.2f}")
        
        # Compute amplitudes for target bitstrings [cite: 106]
        amplitudes = []
        for bits in target_bitstrings:
            amp = mps.amplitude(bits)
            amplitudes.append(amp)
        return amplitudes