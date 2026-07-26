"""Run the first-stage MLSL experiments in sequence."""

from __future__ import annotations

from experiments import (
    exp01_forward_demo,
    exp02_gradient_check,
    exp03_alpha_recovery,
    exp04_beta_gradient_check,
    exp05_alpha_beta_recovery,
    exp06_runtime_scaling,
    exp07_custom_backward_check,
    exp08_stable_evaluator_scan,
    exp09_reference_accuracy,
    exp10_long_time_prediction,
    exp11_noise_robustness,
    exp12_mode_sensitivity,
    exp13_hybrid_threshold_scan,
    exp14_fno_dataset_long_time,
    exp15_sparse_observation_inverse,
    exp16_deeponet_dataset_long_time,
    exp17_ood_alpha_generalization,
    exp18_2d_forward_inverse,
    exp19_fpinn_scalar_inverse,
    exp20_batch_scaling,
    exp30_primitive_contract_audit,
    exp32_reliability_and_inverse_robustness,
    exp34_evaluator_switch_ablation,
    exp35_software_artifact_audit,
)


def main() -> None:
    print("\n=== Experiment 01: forward demo ===")
    exp01_forward_demo.main()

    print("\n=== Experiment 02: gradient check ===")
    exp02_gradient_check.main()

    print("\n=== Experiment 03: alpha recovery ===")
    exp03_alpha_recovery.main()

    print("\n=== Experiment 04: beta gradient check ===")
    exp04_beta_gradient_check.main()

    print("\n=== Experiment 05: alpha/beta recovery ===")
    exp05_alpha_beta_recovery.main()

    print("\n=== Experiment 06: runtime scaling ===")
    exp06_runtime_scaling.main()

    print("\n=== Experiment 07: custom backward check ===")
    exp07_custom_backward_check.main()

    print("\n=== Experiment 08: stable evaluator scan ===")
    exp08_stable_evaluator_scan.main()

    print("\n=== Experiment 09: reference accuracy ===")
    exp09_reference_accuracy.main()

    print("\n=== Experiment 10: long-time prediction ===")
    exp10_long_time_prediction.main()

    print("\n=== Experiment 11: noise robustness ===")
    exp11_noise_robustness.main()

    print("\n=== Experiment 12: mode sensitivity ===")
    exp12_mode_sensitivity.main()

    print("\n=== Experiment 13: hybrid threshold scan ===")
    exp13_hybrid_threshold_scan.main()

    print("\n=== Experiment 14: FNO dataset long-time baseline ===")
    exp14_fno_dataset_long_time.main()

    print("\n=== Experiment 15: sparse observation inverse ===")
    exp15_sparse_observation_inverse.main()

    print("\n=== Experiment 16: DeepONet dataset long-time baseline ===")
    exp16_deeponet_dataset_long_time.main()

    print("\n=== Experiment 17: OOD alpha generalization ===")
    exp17_ood_alpha_generalization.main()

    print("\n=== Experiment 18: 2D forward inverse ===")
    exp18_2d_forward_inverse.main()

    print("\n=== Experiment 19: fPINN scalar inverse baseline ===")
    exp19_fpinn_scalar_inverse.main()

    print("\n=== Experiment 20: batch scaling ===")
    exp20_batch_scaling.main()

    print("\n=== Experiment 30: primitive contract audit ===")
    exp30_primitive_contract_audit.main()

    print("\n=== Experiment 32: evaluator reliability and inverse robustness ===")
    exp32_reliability_and_inverse_robustness.main()

    print("\n=== Experiment 34: evaluator switch ablation ===")
    exp34_evaluator_switch_ablation.main()

    print("\n=== Experiment 35: software artifact audit ===")
    exp35_software_artifact_audit.main()


if __name__ == "__main__":
    main()
