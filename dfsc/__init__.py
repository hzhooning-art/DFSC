"""dfsc: differentiable fractional scientific computing.

The current dfsc artifact exposes a Mittag-Leffler spectral layer as its first
mature primitive while keeping a small ecosystem layer for component discovery,
inverse-order workflows, hybrid residual composition, diagnostics, and
reproducibility.
"""

from .basis import (
    dirichlet_laplacian_1d,
    dirichlet_laplacian_2d,
    mixed_laplacian_1d,
    mixed_laplacian_2d,
    neumann_laplacian_1d,
    neumann_laplacian_2d,
    periodic_laplacian_1d,
    periodic_laplacian_2d,
)
from .baselines import ConditionalMLPField, DeepONet1D, FNO1D, MLPField
from .diagnostics import capability_report, environment_report, smoke_test
from .factory import (
    MLSLConfig,
    build_dirichlet_mlsl_1d,
    build_dirichlet_mlsl_2d,
    build_mixed_mlsl_1d,
    build_mixed_mlsl_2d,
    build_mlsl,
    build_neumann_mlsl_1d,
    build_neumann_mlsl_2d,
    build_periodic_mlsl_1d,
    build_periodic_mlsl_2d,
)
from .forced_layer import ForcedMittagLefflerSpectralLayer
from .l1_baseline import l1_caputo_derivative_uniform, l1_caputo_relaxation
from .mittag_leffler import (
    hybrid_switch_region,
    mittag_leffler_e,
    mittag_leffler_e_ab,
    mittag_leffler_e_ab_hybrid,
)
from .spectral_layer import MittagLefflerSpectralLayer

_primitive_all = [
    "ConditionalMLPField",
    "DeepONet1D",
    "FNO1D",
    "ForcedMittagLefflerSpectralLayer",
    "MLPField",
    "MLSLConfig",
    "MittagLefflerSpectralLayer",
    "build_dirichlet_mlsl_1d",
    "build_dirichlet_mlsl_2d",
    "build_mixed_mlsl_1d",
    "build_mixed_mlsl_2d",
    "build_mlsl",
    "build_neumann_mlsl_1d",
    "build_neumann_mlsl_2d",
    "build_periodic_mlsl_1d",
    "build_periodic_mlsl_2d",
    "capability_report",
    "dirichlet_laplacian_1d",
    "dirichlet_laplacian_2d",
    "environment_report",
    "hybrid_switch_region",
    "l1_caputo_derivative_uniform",
    "l1_caputo_relaxation",
    "mittag_leffler_e",
    "mittag_leffler_e_ab",
    "mittag_leffler_e_ab_hybrid",
    "mixed_laplacian_1d",
    "mixed_laplacian_2d",
    "neumann_laplacian_1d",
    "neumann_laplacian_2d",
    "periodic_laplacian_1d",
    "periodic_laplacian_2d",
    "smoke_test",
]

from .capabilities import ApplicabilityReport, ecosystem_gap_report, mlsl_applicability_report
from .applications import (
    ApplicationCase,
    ApplicationProfile,
    advection_diffusion_case,
    anomalous_diffusion_case,
    application_catalog,
    assembled_relaxation_case,
    network_diffusion_case,
    periodic_advection_diffusion_operator_1d,
)
from .algorithms import (
    AlgorithmSpec,
    AutoDFSC,
    CaputoL1,
    CaputoL1HistoryDirect,
    CaputoL1HistoryFFT,
    MLSLArnoldi,
    MLSLAdaptive,
    MLSLDirect,
    MLSLForced,
    MLSLGeneralizedOperator,
    MLSLGraph,
    MLSLKrylov,
    MLSLOperator,
    MLSLPicard,
    MLSLStable,
    algorithm_registry,
)
from .arnoldi import ArnoldiDiagnostics, arnoldi_mittag_leffler_action
from .complex_ml import (
    ComplexMittagLefflerEvaluation,
    evaluate_complex_mittag_leffler,
    mittag_leffler_e_complex_series,
)
from .evaluators import AdaptiveMittagLefflerEvaluation, MittagLefflerEvaluation, evaluate_mittag_leffler, evaluate_mittag_leffler_adaptive
from .fast_history import (
    CaputoHistoryDiagnostics,
    caputo_l1_derivative_direct,
    caputo_l1_derivative_fft,
    caputo_l1_history,
    caputo_l1_weights,
)
from .history import caputo_l1_linear_solve
from .krylov import (
    AdaptiveKrylovDiagnostics,
    KrylovDiagnostics,
    PreparedLanczosBasis,
    adaptive_lanczos_mittag_leffler_action,
    apply_prepared_lanczos_basis,
    lanczos_mittag_leffler_action,
    prepare_lanczos_basis,
)
from .linear_operators import (
    GeneralLinearOperator,
    SelfAdjointLinearOperator,
    as_general_operator,
    as_self_adjoint_operator,
)
from .datasets import BenchmarkSpec, benchmark_targets, load_tensor_dataset, validate_dataset_manifest
from .experimental import DistributedOrderMLSL, VariableOrderMLSL
from .experimental_data import (
    SPTObservables,
    SPTTrajectoryDataset,
    empirical_spt_observables,
    estimate_wave_number,
    load_anomdiffdb_mat,
    split_trajectories,
)
from .operators import (
    build_generalized_operator_mlsl,
    build_graph_mlsl,
    build_operator_mlsl,
    graph_laplacian_from_adjacency,
    generalized_spectral_decomposition,
    spectral_decomposition_from_operator,
)
from .registry import (
    COMPONENTS,
    ComponentSpec,
    component_summary,
    implemented_components,
    list_components,
    primitive_entrypoints,
)
from .reliability import ReliabilityReport
from .error_budget import (
    ErrorBudget,
    ErrorBudgetReport,
    ErrorComponent,
    alternating_series_remainder_bound,
    compose_error_budget_report,
)
from .identifiability import IdentifiabilityReport, local_identifiability
from .problems import (
    ForcedSpectralProblem,
    CaputoL1Problem,
    CaputoHistoryProblem,
    FractionalSpectralProblem,
    GeneralizedOperatorSpectralProblem,
    GeneralOperatorProblem,
    GraphSpectralProblem,
    LinearOperatorSpectralProblem,
    OperatorSpectralProblem,
    SemilinearSpectralProblem,
    Solution,
)
from .solvers import solve
from .semilinear import PicardDiagnostics, semilinear_mild_picard
from .selection import AlgorithmDecision, choose_algorithm
from .workflows import HybridResidualModel, MittagLefflerResidualRegressor, TrainableOrders, make_trainable_orders, relative_l2_error

__version__ = "0.1.0rc1"
LIBRARY_NAME = "dfsc"

__all__ = _primitive_all + [
    "COMPONENTS",
    "AlgorithmSpec",
    "AlgorithmDecision",
    "ApplicationCase",
    "ApplicationProfile",
    "ApplicabilityReport",
    "ArnoldiDiagnostics",
    "AdaptiveKrylovDiagnostics",
    "AdaptiveMittagLefflerEvaluation",
    "BenchmarkSpec",
    "AutoDFSC",
    "CaputoL1",
    "CaputoL1HistoryDirect",
    "CaputoL1HistoryFFT",
    "CaputoL1Problem",
    "CaputoHistoryDiagnostics",
    "CaputoHistoryProblem",
    "ComponentSpec",
    "ComplexMittagLefflerEvaluation",
    "DistributedOrderMLSL",
    "ForcedSpectralProblem",
    "FractionalSpectralProblem",
    "GeneralizedOperatorSpectralProblem",
    "GeneralLinearOperator",
    "GeneralOperatorProblem",
    "GraphSpectralProblem",
    "HybridResidualModel",
    "KrylovDiagnostics",
    "PreparedLanczosBasis",
    "LinearOperatorSpectralProblem",
    "LIBRARY_NAME",
    "MLSLDirect",
    "MLSLAdaptive",
    "MLSLArnoldi",
    "MLSLForced",
    "MLSLGeneralizedOperator",
    "MLSLGraph",
    "MLSLKrylov",
    "MLSLOperator",
    "MLSLPicard",
    "MLSLStable",
    "MittagLefflerEvaluation",
    "MittagLefflerResidualRegressor",
    "OperatorSpectralProblem",
    "PicardDiagnostics",
    "ReliabilityReport",
    "ErrorBudget",
    "ErrorBudgetReport",
    "ErrorComponent",
    "IdentifiabilityReport",
    "SemilinearSpectralProblem",
    "Solution",
    "SelfAdjointLinearOperator",
    "SPTObservables",
    "SPTTrajectoryDataset",
    "TrainableOrders",
    "VariableOrderMLSL",
    "__version__",
    "algorithm_registry",
    "advection_diffusion_case",
    "anomalous_diffusion_case",
    "application_catalog",
    "arnoldi_mittag_leffler_action",
    "adaptive_lanczos_mittag_leffler_action",
    "apply_prepared_lanczos_basis",
    "as_general_operator",
    "as_self_adjoint_operator",
    "assembled_relaxation_case",
    "benchmark_targets",
    "build_graph_mlsl",
    "build_generalized_operator_mlsl",
    "build_operator_mlsl",
    "component_summary",
    "caputo_l1_linear_solve",
    "caputo_l1_derivative_direct",
    "caputo_l1_derivative_fft",
    "caputo_l1_history",
    "caputo_l1_weights",
    "choose_algorithm",
    "ecosystem_gap_report",
    "empirical_spt_observables",
    "estimate_wave_number",
    "evaluate_mittag_leffler",
    "evaluate_mittag_leffler_adaptive",
    "evaluate_complex_mittag_leffler",
    "alternating_series_remainder_bound",
    "compose_error_budget_report",
    "local_identifiability",
    "graph_laplacian_from_adjacency",
    "generalized_spectral_decomposition",
    "implemented_components",
    "list_components",
    "load_anomdiffdb_mat",
    "make_trainable_orders",
    "mittag_leffler_e_complex_series",
    "mlsl_applicability_report",
    "network_diffusion_case",
    "periodic_advection_diffusion_operator_1d",
    "primitive_entrypoints",
    "relative_l2_error",
    "solve",
    "split_trajectories",
    "spectral_decomposition_from_operator",
    "load_tensor_dataset",
    "lanczos_mittag_leffler_action",
    "prepare_lanczos_basis",
    "semilinear_mild_picard",
    "validate_dataset_manifest",
]
