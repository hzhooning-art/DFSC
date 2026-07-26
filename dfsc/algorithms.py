"""Algorithm objects for the dfsc problem--solve interface."""

from __future__ import annotations

from dataclasses import dataclass

from .factory import MLSLConfig


@dataclass(frozen=True)
class AlgorithmSpec:
    """Machine-readable description of a dfsc algorithm family."""

    name: str
    status: str
    scope: str
    differentiable: bool
    gpu_capable: bool

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class MLSLDirect:
    """Direct Mittag-Leffler spectral propagation with the series evaluator."""

    config: MLSLConfig = MLSLConfig()

    @property
    def name(self) -> str:
        return "mlsl-direct"


@dataclass(frozen=True)
class MLSLStable:
    """Direct spectral propagation using the hybrid stable evaluator preset."""

    terms: int = 120
    wave_speed: float = 1.0
    beta: float = 2.0

    @property
    def name(self) -> str:
        return "mlsl-stable"

    @property
    def config(self) -> MLSLConfig:
        return MLSLConfig.stable(terms=self.terms, wave_speed=self.wave_speed, beta=self.beta)


@dataclass(frozen=True)
class MLSLForced:
    """Duhamel-style forced propagation built on an MLSL backbone."""

    config: MLSLConfig = MLSLConfig.stable()
    forcing_terms: int = 80

    @property
    def name(self) -> str:
        return "mlsl-forced"


@dataclass(frozen=True)
class MLSLOperator:
    """Construct MLSL from a user-supplied symmetric PSD operator."""

    config: MLSLConfig = MLSLConfig.stable()
    num_modes: int | None = None

    @property
    def name(self) -> str:
        return "mlsl-operator"


@dataclass(frozen=True)
class MLSLKrylov:
    """Lanczos approximation of the MLSL matrix-function action."""

    config: MLSLConfig = MLSLConfig.stable()
    krylov_dimension: int = 48
    breakdown_tol: float = 1e-12
    estimate_error: bool = True
    error_dimension_step: int = 8

    @property
    def name(self) -> str:
        return "mlsl-krylov"


@dataclass(frozen=True)
class MLSLAdaptive:
    """Tolerance-driven Lanczos propagation for symmetric PSD operators."""

    config: MLSLConfig = MLSLConfig.stable()
    dimension_schedule: tuple[int, ...] = (8, 16, 24, 32, 48, 64)
    rtol: float = 1e-7
    atol: float = 1e-9
    breakdown_tol: float = 1e-12
    strict: bool = False

    @property
    def name(self) -> str:
        return "mlsl-adaptive"


@dataclass(frozen=True)
class MLSLArnoldi:
    """Controlled Arnoldi Mittag-Leffler action for general operators."""

    arnoldi_dimension: int = 32
    terms: int = 120
    max_reduced_radius: float = 4.0
    breakdown_tol: float = 1e-12
    estimate_error: bool = True
    error_dimension_step: int = 8
    allow_unvalidated: bool = False

    @property
    def name(self) -> str:
        return "mlsl-arnoldi"


@dataclass(frozen=True)
class MLSLGeneralizedOperator:
    """Construct MLSL from symmetric stiffness and positive-definite mass matrices."""

    config: MLSLConfig = MLSLConfig.stable()
    num_modes: int | None = None

    @property
    def name(self) -> str:
        return "mlsl-generalized-operator"


@dataclass(frozen=True)
class MLSLGraph:
    """Construct MLSL from a dense undirected graph adjacency matrix."""

    config: MLSLConfig = MLSLConfig.stable()
    num_modes: int | None = None
    normalized: bool = False

    @property
    def name(self) -> str:
        return "mlsl-graph"


@dataclass(frozen=True)
class MLSLPicard:
    """Mild-form fixed-point iteration for semilinear MLSL dynamics."""

    config: MLSLConfig = MLSLConfig.stable()
    max_iterations: int = 30
    tolerance: float = 1e-7
    relaxation: float = 1.0
    quadrature_points: int = 48
    forcing_terms: int = 100

    @property
    def name(self) -> str:
        return "mlsl-picard"


@dataclass(frozen=True)
class AutoDFSC:
    """Select a supported dfsc algorithm from detached problem diagnostics."""

    direct_terms: int = 100
    stable_terms: int = 120
    direct_radius: float = 12.0
    dense_eigh_limit: int = 256
    krylov_dimension: int = 48
    history_fft_threshold: int = 128

    @property
    def name(self) -> str:
        return "auto-dfsc"


@dataclass(frozen=True)
class CaputoL1:
    """Implicit history-aware L1 fallback for constant-order Caputo systems."""

    @property
    def name(self) -> str:
        return "caputo-l1"


@dataclass(frozen=True)
class CaputoL1HistoryDirect:
    """Direct quadratic-cost Caputo-L1 trajectory convolution."""

    @property
    def name(self) -> str:
        return "caputo-l1-history-direct"


@dataclass(frozen=True)
class CaputoL1HistoryFFT:
    """FFT-accelerated Caputo-L1 full-trajectory convolution."""

    @property
    def name(self) -> str:
        return "caputo-l1-history-fft"


def algorithm_registry() -> list[dict[str, object]]:
    """Return the implemented and experimental dfsc algorithm families."""

    specs = (
        AlgorithmSpec(
            name="auto-dfsc",
            status="implemented",
            scope="diagnostic selection among currently implemented dfsc spectral algorithms",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="caputo-l1",
            status="implemented",
            scope="constant-order linear Caputo systems with 0 < alpha < 1 on uniform time grids",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="caputo-l1-history-direct",
            status="implemented",
            scope="reference full-trajectory Caputo-L1 convolution on uniform time grids",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="caputo-l1-history-fft",
            status="implemented",
            scope="FFT-accelerated full-trajectory Caputo-L1 residual evaluation",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-direct",
            status="implemented",
            scope="known retained spectral propagators with moderate real non-positive arguments",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-stable",
            status="implemented",
            scope="known retained spectral propagators in broader negative-real regimes",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-forced",
            status="implemented",
            scope="nonhomogeneous dynamics with sampled forcing and Duhamel-style quadrature",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-operator",
            status="implemented",
            scope="user-supplied symmetric positive semidefinite discrete operators",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-krylov",
            status="implemented",
            scope="matrix-function actions for dense, sparse, or matrix-free symmetric PSD operators",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-adaptive",
            status="implemented",
            scope="tolerance-driven differentiable matrix-function actions for symmetric PSD operators",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-arnoldi",
            status="implemented",
            scope="controlled moderate-radius actions for general real or complex linear operators",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-generalized-operator",
            status="implemented",
            scope="assembled symmetric stiffness and positive-definite mass matrix pairs",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-graph",
            status="implemented",
            scope="dense undirected graph Laplacians as retained spectral operators",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="mlsl-picard",
            status="implemented",
            scope="semilinear mild solutions on a known retained MLSL spectral backbone",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="variable-order-wrapper",
            status="experimental",
            scope="query-wise alpha sampling around an MLSL backbone",
            differentiable=True,
            gpu_capable=True,
        ),
        AlgorithmSpec(
            name="distributed-order-wrapper",
            status="experimental",
            scope="weighted quadrature mixtures over retained alpha nodes",
            differentiable=True,
            gpu_capable=True,
        ),
    )
    return [spec.to_dict() for spec in specs]
