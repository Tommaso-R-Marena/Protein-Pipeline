"""Configuration management for QCFold."""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class EncoderConfig:
    type: str = "esm2"
    model_name: str = "esm2_t6_8M_UR50D"
    embed_dim: int = 320
    freeze: bool = True


@dataclass
class GeneratorConfig:
    type: str = "fragment_diffusion"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    diffusion_steps: int = 100
    num_candidates: int = 32


@dataclass
class QuantumConfig:
    enabled: bool = True
    backend: str = "default.qubit"
    num_qubits_per_residue: int = 2
    max_region_size: int = 16
    circuit_depth: int = 6
    num_layers: int = 4
    optimizer: str = "adam"
    lr: float = 0.01
    max_iterations: int = 200
    use_classical_fallback: bool = True
    fallback_method: str = "simulated_annealing"
    sa_temperature: float = 1.0
    sa_cooling_rate: float = 0.995


@dataclass
class PhysicsConfig:
    clash_weight: float = 10.0
    bond_weight: float = 5.0
    ramachandran_weight: float = 2.0
    hydrogen_bond_weight: float = 1.0
    contact_weight: float = 1.0
    clash_threshold: float = 2.0
    bond_tolerance: float = 0.05


@dataclass
class EnsembleConfig:
    num_conformations: int = 8
    temperature: float = 1.0
    diversity_weight: float = 0.3
    min_tm_diversity: float = 0.3


@dataclass
class RankingConfig:
    physics_weight: float = 0.4
    learned_weight: float = 0.4
    uncertainty_weight: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 4
    num_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    losses: Dict[str, float] = field(default_factory=lambda: {
        "coordinate_loss": 1.0,
        "distance_map_loss": 0.5,
        "torsion_loss": 0.3,
        "ensemble_coverage_loss": 0.5,
        "calibration_loss": 0.2,
        "clash_penalty": 0.1,
        "fold_switch_loss": 1.0,
    })


@dataclass
class EvalConfig:
    tm_score_threshold: float = 0.6
    rmsd_threshold: float = 5.0
    num_seeds: int = 5
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class QCFoldConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str) -> "QCFoldConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        
        config = cls()
        if "model" in raw:
            m = raw["model"]
            if "encoder" in m:
                config.encoder = EncoderConfig(**m["encoder"])
            if "generator" in m:
                config.generator = GeneratorConfig(**m["generator"])
            if "quantum" in m:
                config.quantum = QuantumConfig(**m["quantum"])
            if "physics" in m:
                config.physics = PhysicsConfig(**m["physics"])
            if "ensemble" in m:
                config.ensemble = EnsembleConfig(**m["ensemble"])
            if "ranking" in m:
                config.ranking = RankingConfig(**m["ranking"])
        if "training" in raw:
            t = raw["training"]
            losses = t.pop("losses", None)
            config.training = TrainingConfig(**t)
            if losses:
                config.training.losses = losses
        if "evaluation" in raw:
            config.evaluation = EvalConfig(**raw["evaluation"])
        config.device = raw.get("device", "cpu")
        config.seed = raw.get("seed", 42)
        config.output_dir = raw.get("output_dir", "outputs")
        return config
