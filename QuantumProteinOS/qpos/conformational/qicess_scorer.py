import numpy as np

def _kabsch_alignment(P, Q):
    """
    Computes the optimal rotation matrix using Kabsch algorithm.
    P and Q must be (N, 3) numpy arrays centered at origin.
    """
    C = np.dot(np.transpose(P), Q)
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    U = np.dot(V, W)
    return U

def imfdrMSD(pred_coords: np.ndarray, ref_coords: np.ndarray, inhibitory_residues: list) -> float:
    """
    imfdRMSD computed only over the inhibitory module residues 
    after global Kabsch alignment on all Cα atoms.
    """
    # 1. Global alignment on all CA atoms
    N = pred_coords.shape[0]
    p_center = np.mean(pred_coords, axis=0)
    r_center = np.mean(ref_coords, axis=0)
    
    P_centered = pred_coords - p_center
    R_centered = ref_coords - r_center
    
    U = _kabsch_alignment(P_centered, R_centered)
    
    # Rotate pred
    P_aligned = np.dot(P_centered, U)
    
    # 2. RMSD only over inhibitory_residues
    if len(inhibitory_residues) == 0:
        return 0.0
        
    p_inhib = P_aligned[inhibitory_residues]
    r_inhib = R_centered[inhibitory_residues]
    
    diff = p_inhib - r_inhib
    msd = np.sum(diff**2) / len(inhibitory_residues)
    return float(np.sqrt(msd))

class QICESSScorer:
    """
    QICESS Scorer for autoinhibited protein conformational ensembles.
    Composite weights:
      - quantum: 0.30
      - retrieval: 0.25
      - plddt: 0.25
      - pae: 0.15
      - rama: 0.05
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.weights = self.config.get('weights', {
            'quantum': 0.30, 'retrieval': 0.25, 'plddt': 0.25, 'pae': 0.15, 'rama': 0.05
        })
        self.contact_cutoff = self.config.get('contact_cutoff', 8.0)
        
    def ising_contact_energy(self, structure) -> float:
        """
        E = Σ_{i<j} MJ[aa_i, aa_j] * contact_ij
        Miyazawa-Jernigan 20×20 statistical potential matrix.
        """
        return -15.0 # MOCK MJ energy
        
    def score_ensemble(self, structures: list, reference_active: str=None, reference_inactive: str=None) -> list:
        # Mock scores
        return [{'structure': s, 'score': np.random.rand() * 100} for s in structures]
        
    def rank_ensemble(self, scores: list) -> list:
        return sorted(scores, key=lambda x: x['score'], reverse=True)
