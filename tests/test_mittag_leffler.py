from __future__ import annotations

import unittest

import torch

import dfsc
from dfsc import mittag_leffler_e, mittag_leffler_e_ab


class MittagLefflerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)

    def test_value_at_zero_is_one(self) -> None:
        z = torch.zeros(4)
        values = mittag_leffler_e(torch.tensor(1.3), z, terms=40)
        self.assertTrue(torch.allclose(values, torch.ones_like(values), atol=1e-12))

    def test_custom_backward_matches_autograd(self) -> None:
        z = -torch.linspace(0.0, 1.0, 12)
        target = mittag_leffler_e(torch.tensor(1.2), z, terms=100).detach()

        alpha_auto = torch.tensor(1.55, requires_grad=True)
        loss_auto = torch.mean((mittag_leffler_e(alpha_auto, z, terms=80) - target) ** 2)
        loss_auto.backward()

        alpha_custom = torch.tensor(1.55, requires_grad=True)
        loss_custom = torch.mean(
            (mittag_leffler_e(alpha_custom, z, terms=80, custom_backward=True) - target) ** 2
        )
        loss_custom.backward()

        self.assertLess(
            abs(alpha_auto.grad.item() - alpha_custom.grad.item())
            / max(abs(alpha_auto.grad.item()), 1e-14),
            1e-10,
        )

    def test_hybrid_large_negative_is_finite(self) -> None:
        z = -torch.tensor([8.0, 20.0, 80.0])
        values = mittag_leffler_e(torch.tensor(0.65), z, terms=80, method="hybrid")
        self.assertTrue(torch.all(torch.isfinite(values)).item())

    def test_hybrid_low_alpha_switches_before_series_instability(self) -> None:
        z = -torch.tensor([6.0, 8.0])
        values = mittag_leffler_e(torch.tensor(0.45), z, terms=160, method="hybrid")
        self.assertTrue(torch.all(torch.isfinite(values)).item())
        self.assertTrue(torch.all(torch.abs(values) < 1.0).item())

    def test_hybrid_transition_keeps_finite_alpha_gradients(self) -> None:
        z = -torch.linspace(7.0, 9.0, 9)
        alpha = torch.tensor(0.65, requires_grad=True)
        values = mittag_leffler_e(alpha, z, terms=160, method="hybrid")
        loss = torch.mean(values**2)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(values)).item())
        self.assertTrue(torch.isfinite(alpha.grad).item())

    def test_hybrid_keeps_series_accuracy_longer_above_order_one(self) -> None:
        alpha = torch.tensor(1.35)
        z = torch.tensor([-8.0, -10.0])
        series = mittag_leffler_e(alpha, z, terms=160, method="series")
        hybrid = mittag_leffler_e(alpha, z, terms=160, method="hybrid")
        self.assertTrue(torch.allclose(hybrid, series, atol=1e-11, rtol=1e-11))

    def test_two_parameter_hybrid_large_negative_is_finite(self) -> None:
        z = -torch.tensor([8.0, 20.0, 80.0])
        values = mittag_leffler_e_ab(
            torch.tensor(0.75),
            torch.tensor(0.75),
            z,
            terms=80,
            method="hybrid",
        )
        self.assertTrue(torch.all(torch.isfinite(values)).item())

    def test_diagnostic_evaluator_preserves_autograd(self) -> None:
        alpha = torch.tensor(0.75, requires_grad=True)
        z = -torch.tensor([0.2, 2.0, 20.0])
        evaluation = dfsc.evaluate_mittag_leffler(alpha, z, method="auto", terms=100)
        evaluation.values.square().mean().backward()
        self.assertEqual(evaluation.method, "hybrid")
        self.assertEqual(sum(evaluation.branch_counts.values()), z.numel())
        self.assertTrue(evaluation.finite)
        self.assertTrue(torch.isfinite(alpha.grad).item())
        self.assertEqual(
            evaluation.error_estimate_kind,
            "embedded-truncation-disagreement-not-a-rigorous-bound",
        )

    def test_diagnostic_evaluator_two_parameter_series(self) -> None:
        z = -torch.linspace(0.0, 0.5, 6)
        evaluation = dfsc.evaluate_mittag_leffler(
            torch.tensor(0.9),
            z,
            beta=torch.tensor(1.2),
            method="series",
            terms=80,
        )
        self.assertEqual(evaluation.method, "series")
        self.assertTrue(evaluation.finite)
        self.assertTrue(evaluation.converged)
        self.assertEqual(evaluation.branch_counts["series"], z.numel())

    def test_diagnostic_evaluator_rejects_unvalidated_order(self) -> None:
        with self.assertRaises(ValueError):
            dfsc.evaluate_mittag_leffler(torch.tensor(2.2), -torch.ones(2))

    def test_reliability_contract_marks_series_regime_high(self) -> None:
        evaluation = dfsc.evaluate_mittag_leffler(
            torch.tensor(0.8),
            -torch.linspace(0.0, 0.5, 8),
            method="series",
            terms=100,
            strict=True,
        )
        self.assertEqual(evaluation.reliability.level, "high")
        self.assertTrue(evaluation.reliability.trusted)
        self.assertTrue(evaluation.reliability.rigorous_error_bound)
        self.assertEqual(
            evaluation.error_estimate_kind,
            "rigorous-alternating-series-remainder-bound",
        )

    def test_reliability_contract_marks_hybrid_regime_moderate(self) -> None:
        evaluation = dfsc.evaluate_mittag_leffler(
            torch.tensor(0.7),
            -torch.tensor([0.1, 20.0]),
            method="hybrid",
            terms=120,
        )
        self.assertEqual(evaluation.reliability.level, "moderate")
        self.assertEqual(evaluation.reliability.gradient_reliability, "moderate")

    def test_strict_mode_rejects_outside_validated_domain(self) -> None:
        with self.assertRaises(RuntimeError):
            dfsc.evaluate_mittag_leffler(
                torch.tensor(0.8),
                torch.tensor([0.1]),
                method="series",
                strict=True,
            )

    def test_two_parameter_rejects_nonpositive_beta(self) -> None:
        with self.assertRaises(ValueError):
            dfsc.evaluate_mittag_leffler(
                torch.tensor(0.8),
                -torch.ones(2),
                beta=torch.tensor(0.0),
            )


if __name__ == "__main__":
    unittest.main()
