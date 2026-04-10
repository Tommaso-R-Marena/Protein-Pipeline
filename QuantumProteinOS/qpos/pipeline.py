import os
import json
import numpy as np

from qpos.disorder import DisorderNetV6
from qpos.conformational import ConformationalSampler, QICESSScorer
from qpos.quantum import IWSQAOASolver, QADFRubric
from qpos.chirality import ChiralFoldAuditor, AF3ChiralityCorrector
from qpos.rotamers import RotamerPacker

class PipelineResult:
    def __init__(self, structure, ensemble, disorder_scores, chirality_report, qadf_score):
        self.structure = structure
        self.ensemble = ensemble
        self.disorder_scores = disorder_scores
        self.chirality_report = chirality_report
        self.qadf_score = qadf_score
        
    def save(self, output_dir: str):
        """Save the pipeline results into a structured directory."""
        os.makedirs(output_dir, exist_ok=True)
        ensemble_dir = os.path.join(output_dir, "ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Structure stub output
        struct_path = os.path.join(output_dir, "structure.pdb")
        with open(struct_path, "w") as f:
            f.write(f"REMARK 999 Stub packed structure: {self.structure}\n")
            
        # Ensemble output
        for i, conf_info in enumerate(self.ensemble):
            # conf_info is dict with 'structure' and 'score' in QICESSScorer stub
            conf = conf_info['structure'] if isinstance(conf_info, dict) else conf_info
            conf_path = os.path.join(ensemble_dir, f"conf_{i+1:03d}.pdb")
            with open(conf_path, "w") as f:
                f.write(f"REMARK 999 Ensemble conf: {conf}\n")
                
        # Disorder scores
        if self.disorder_scores is not None:
            np.savetxt(os.path.join(output_dir, "disorder_scores.csv"), self.disorder_scores, delimiter=",", header="Probability", comments="")
            
        # Chirality report
        if self.chirality_report is not None:
            with open(os.path.join(output_dir, "chirality_report.json"), "w") as f:
                json.dump(self.chirality_report, f, indent=2)
                
        # QADF
        if self.qadf_score is not None:
            with open(os.path.join(output_dir, "qadf_score.json"), "w") as f:
                json.dump(self.qadf_score, f, indent=2)
                
        # Summary
        with open(os.path.join(output_dir, "summary.txt"), "w") as f:
            f.write("QuantumProteinOS Execution Summary\n")
            f.write("==================================\n")
            f.write(f"Final structure: {struct_path}\n")
            f.write(f"Ensemble size: {len(self.ensemble)}\n")
            f.write("Results output successfully.\n")


class QuantumProteinOS:
    """
    Unified quantum-classical protein structure prediction pipeline.
    Addresses three principal AF3 failure modes:
      1. Conformational diversity (fold-switching, autoinhibition)
      2. Intrinsic disorder
      3. Stereochemical correctness
    """

    def __init__(self, config: dict):
        self.disorder_model = DisorderNetV6(config.get('disorder', {}))
        self.ensemble_generator = ConformationalSampler(config.get('conformational', {}))
        self.qicess = QICESSScorer(config.get('qicess', {}))
        
        # In a real run, IWSQAOASolver would be passed to RotamerPacker
        self.rotamer_packer = RotamerPacker(config.get('rotamers', {}))
        
        self.chirality_auditor = ChiralFoldAuditor(config.get('chirality', {}))
        self.af3_corrector = AF3ChiralityCorrector()
        self.qadf_rubric = QADFRubric()

    @classmethod
    def from_config(cls, config_path: str):
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(self, sequence: str, pdb_path: str = None) -> PipelineResult:
        """
        Full pipeline:
          1. Disorder prediction (DisorderNet v6)
          2. Conformational ensemble generation (NMA + rigid-body, QuantumFoldX)
          3. Quantum-scored ensemble ranking (QICESS v2, QuantumFoldBench)
          4. Rotamer packing on structured regions (IWS-QAOA, qprotein-iws)
          5. Chirality auditing + AF3 correction (ChiralFold)
          6. QADF quantum readiness assessment
        """
        # Step 1: Disorder
        disorder_scores = self.disorder_model.predict(sequence)
        ordered_mask = disorder_scores < 0.5

        # Step 2: Conformational ensemble
        ensemble = self.ensemble_generator.generate(
            pdb_path, n_modes=10, amplitude_scales=[2.0, 6.0],
            rigid_body_params={'translation': [5.0, 15.0], 'rotation': [20.0, 45.0]}
        )

        # Step 3: QICESS ensemble scoring
        scored = self.qicess.score_ensemble(ensemble)
        ranked = self.qicess.rank_ensemble(scored)

        # Step 4: Rotamer packing
        packed = self.rotamer_packer.pack_structure(
            ranked, ordered_mask=ordered_mask,
            window_size=4, n_rotamers=3, p=4, n_iter=3
        )

        # Step 5: Chirality audit + correction
        report = self.chirality_auditor.audit(packed)
        if report['chirality']['pct_correct'] < 100.0:
            packed = self.af3_corrector.correct(packed)

        # Step 6: QADF rubric
        qadf_score = self.qadf_rubric.score("Rotamer Packing", packed)

        return PipelineResult(
            structure=packed,
            ensemble=ranked,
            disorder_scores=disorder_scores,
            chirality_report=report,
            qadf_score=qadf_score
        )
