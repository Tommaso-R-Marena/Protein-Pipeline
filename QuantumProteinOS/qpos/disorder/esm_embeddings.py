import torch

class ESMCPUFeatureExtractor:
    """Uses fair-esm 8M CPU model to extract baseline representations."""
    def __init__(self):
        try:
            import esm
            self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            self.batch_converter = self.alphabet.get_batch_converter()
            self.model.eval()
        except ImportError:
            self.model = None

    def get_embeddings(self, sequence: str) -> torch.Tensor:
        if self.model is None:
            raise ImportError("fair-esm is missing. Pip install fair-esm>=2.0.0")
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]
        # Return only sequence tokens (remove start/end)
        return token_representations[0, 1 : len(sequence) + 1]
