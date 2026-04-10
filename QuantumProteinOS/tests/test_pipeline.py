import os
import pytest
from qpos.pipeline import QuantumProteinOS

@pytest.fixture
def mock_pdb_fetcher(mocker):
    """Mocks network calls for PDBs using local fixtures."""
    def mock_fetch_pdb(self, pdb_id):
        # Redirect to fixtures
        path = os.path.join(os.path.dirname(__file__), 'fixtures', f"{pdb_id.upper()}.pdb")
        # Just create the file if it doesn't exist for test purposes
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("HEADER MOCK")
        return path
    
    mocker.patch('qpos.data.pdb_fetcher.PDBFetcher.fetch_pdb', mock_fetch_pdb)

def test_pipeline_instantiation():
    config = {
        'disorder': {},
        'conformational': {'n_nma_modes': 2, 'amplitude_scales': [2.0]},
        'qicess': {'weights': {'quantum': 1.0}},
        'rotamers': {}
    }
    qpos = QuantumProteinOS(config)
    assert qpos.disorder_model is not None
    assert qpos.ensemble_generator is not None
    assert qpos.rotamer_packer is not None
    
def test_pipeline_run(mock_pdb_fetcher):
    qpos = QuantumProteinOS({})
    result = qpos.run(sequence="NLYNLY", pdb_path="1L2Y")
    assert result.disorder_scores is not None
    assert len(result.ensemble) > 0
    assert result.qadf_score is not None
