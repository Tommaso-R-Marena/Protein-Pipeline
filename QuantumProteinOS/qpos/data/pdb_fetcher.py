import os
import requests
import urllib.request
from Bio.PDB import PDBList

class PDBFetcher:
    """Download PDB structures via Biopython PDBList and RCSB/AlphaFold REST APIs."""
    
    def __init__(self, pdb_dir: str = "data/pdb"):
        self.pdb_dir = pdb_dir
        os.makedirs(self.pdb_dir, exist_ok=True)
        self.pdbl = PDBList(pdb=self.pdb_dir, obsolete_dir=os.path.join(self.pdb_dir, "obsolete"))
        
    def fetch_pdb(self, pdb_id: str) -> str:
        """Fetch a coordinate file from PDB and return the local path."""
        pdb_id = pdb_id.upper()
        # Retrieve PDB format; Biopython defaults to mmCIF, but we usually prefer PDB
        file_path = self.pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=self.pdb_dir)
        
        # Biopython will save the file with the prefix 'pdb' e.g. pdb1l2y.ent
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Failed to fetch {pdb_id}")
        return file_path
        
    def fetch_batch(self, pdb_ids: list) -> dict:
        """Batch download multiple PDB files."""
        results = {}
        for pid in pdb_ids:
            try:
                results[pid] = self.fetch_pdb(pid)
            except Exception as e:
                print(f"Error fetching {pid}: {e}")
        return results

    def fetch_alphafold(self, uniprot_id: str) -> str:
        """Fetch predicted structure from AlphaFold Database."""
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        dest_path = os.path.join(self.pdb_dir, f"AF-{uniprot_id}-F1-model_v4.pdb")
        if not os.path.exists(dest_path):
            try:
                urllib.request.urlretrieve(url, dest_path)
            except Exception as e:
                raise FileNotFoundError(f"Failed to fetch AlphaFold model for {uniprot_id}: {e}")
        return dest_path
