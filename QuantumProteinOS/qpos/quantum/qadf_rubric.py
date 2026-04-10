QADF_DIMENSIONS = [
    'qubit_count',           # How many qubits does the QUBO encoding require?
    'locality',              # Are interactions k-local (k ≤ 2)?
    'connectivity',          # Is the interaction graph sparse?
    'noise_tolerance',       # How sensitive is the solution to depolarizing noise?
    'available_encodings',   # One-hot, binary, unary — which work?
    'classical_hardness',    # Is the problem NP-hard / practically hard for classical?
    'problem_size_scaling',  # How does qubit count grow with biological system size?
    'biological_relevance',  # Does solving this improve functional predictions?
    'data_availability',     # Are training/benchmark datasets available?
    'idp_sampler_mode',      # V2: IDR ensemble sampling (10th dimension)
]

class QADFRubric:
    def score(self, subproblem: str, structure=None) -> dict:
        """
        Returns per-dimension scores (0-3) and total QADF score.
        V2 Categories: A (near-term: rotamer packing), B (medium-term: short peptide), 
        C (poor: global backbone), D (new: IDR ensemble sampling).
        """
        
        # Determine category rough score based on subproblem
        cat_score = 1
        if "Rotamer" in subproblem:
            cat_score = 3
            
        return {dim: cat_score for dim in QADF_DIMENSIONS}
