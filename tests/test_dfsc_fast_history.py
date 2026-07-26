from __future__ import annotations

import unittest

import torch

import dfsc


class DfscFastHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(41)

    def test_fft_matches_direct_values_and_gradients(self) -> None:
        values_direct = torch.randn(2, 65, 3, requires_grad=True)
        values_fft = values_direct.detach().clone().requires_grad_(True)
        alpha_direct = torch.tensor(0.72, requires_grad=True)
        alpha_fft = alpha_direct.detach().clone().requires_grad_(True)
        direct, _ = dfsc.caputo_l1_derivative_direct(
            values_direct, alpha=alpha_direct, final_time=1.0
        )
        fft, diagnostics = dfsc.caputo_l1_derivative_fft(
            values_fft, alpha=alpha_fft, final_time=1.0
        )
        self.assertTrue(torch.allclose(fft, direct, rtol=1e-11, atol=1e-11))
        direct.square().mean().backward()
        fft.square().mean().backward()
        self.assertTrue(torch.allclose(alpha_fft.grad, alpha_direct.grad, rtol=1e-10, atol=1e-10))
        self.assertTrue(torch.allclose(values_fft.grad, values_direct.grad, rtol=1e-10, atol=1e-10))
        self.assertEqual(diagnostics.fft_length, 128)

    def test_power_function_matches_caputo_derivative(self) -> None:
        alpha = torch.tensor(0.65)
        power = torch.tensor(2.0)
        steps = 1024
        times = torch.linspace(0.0, 1.0, steps + 1)
        numerical, _ = dfsc.caputo_l1_derivative_fft(
            times.pow(power), alpha=alpha, final_time=1.0
        )
        exact = (
            torch.exp(torch.lgamma(power + 1.0) - torch.lgamma(power + 1.0 - alpha))
            * times[1:].pow(power - alpha)
        )
        relative = torch.linalg.vector_norm(numerical - exact) / torch.linalg.vector_norm(exact)
        self.assertLess(float(relative), 2e-4)

    def test_auto_selects_history_method_by_length(self) -> None:
        short = dfsc.CaputoHistoryProblem(torch.linspace(0.0, 1.0, 33), 0.7, 1.0)
        long = dfsc.CaputoHistoryProblem(torch.linspace(0.0, 1.0, 258), 0.7, 1.0)
        self.assertEqual(dfsc.solve(short).algorithm, "caputo-l1-history-direct")
        long_solution = dfsc.solve(long)
        self.assertEqual(long_solution.algorithm, "caputo-l1-history-fft")
        self.assertFalse(long_solution.stats["online_time_stepper"])
        self.assertEqual(long_solution.final.ndim, 0)

    def test_invalid_order_and_method_are_rejected(self) -> None:
        values = torch.linspace(0.0, 1.0, 8)
        with self.assertRaises(ValueError):
            dfsc.caputo_l1_history(values, alpha=1.2, final_time=1.0)
        with self.assertRaises(ValueError):
            dfsc.caputo_l1_history(values, alpha=0.7, final_time=1.0, method="unknown")


if __name__ == "__main__":
    unittest.main()
