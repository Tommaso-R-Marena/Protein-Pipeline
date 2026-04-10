SIMULATION_DISCLAIMER = (
    "WARNING: All quantum circuit results are CLASSICALLY SIMULATED via "
    "PennyLane default.qubit. No QPU hardware is used. Results labeled [CS] "
    "reflect exact statevector simulation."
)

print(SIMULATION_DISCLAIMER)

from .qubo_builder import build_qubo_from_window
from .xy_mixer import XYMixer
from .iws_qaoa import IWSQAOASolver
from .vqc import VQCOptimizer
from .zne import zero_noise_extrapolate
from .qadf_rubric import QADFRubric
from .ogp_router import should_use_quantum
