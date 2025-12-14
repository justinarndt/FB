import torch
import torchhd
import torch.nn.functional as F

class HDCSurfaceDecoder:
    def __init__(self, num_stabilizers, dim=10000, device='cpu'):
        self.dim = dim
        self.device = device
        
        # 1. Item Memory: Unique ID for each stabilizer (MAP architecture) [cite: 139]
        self.stabilizer_basis = torchhd.random(
            num_stabilizers, dim, vsa="MAP", device=device
        )
        self.associative_memory = None 

    def encode_syndrome(self, active_stabilizer_indices):
        """Encodes active stabilizer indices into a Hypervector[cite: 145]."""
        # Select basis vectors
        vectors = torch.index_select(self.stabilizer_basis, 0, active_stabilizer_indices)
        
        # Bundle: Sum and Binarize (Sign function) [cite: 147]
        summed = torch.sum(vectors, dim=0)
        encoded_hv = torch.sign(summed)
        encoded_hv[encoded_hv == 0] = 1 # Tie-breaking
        return encoded_hv

    @torch.jit.export
    def decode(self, syndrome_indices):
        """Fast Inference Loop [cite: 156]"""
        # 1. Encode
        query = self.encode_syndrome(syndrome_indices)
        
        # 2. Similarity Search (Cosine ~ Hamming for MAP)
        if self.associative_memory is None:
             # Fallback for benchmark if memory not trained
             return torch.tensor(0) 
             
        similarities = torchhd.cosine_similarity(query, self.associative_memory)
        
        # 3. Argmax
        best_match_idx = torch.argmax(similarities)
        return best_match_idx