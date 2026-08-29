import unittest
import numpy as np

from p5_memory_protocol import (
    CurveRecord,
    conformal_upper_pvalue,
    conformal_upper_quantile,
    continuous_spectrum_curves,
    fit_oscillatory_shared,
    fit_partially_shared,
    generalized_design,
    grouped_conformal_audit,
)


class ExtensionTests(unittest.TestCase):
    def test_conjugate_pair_design_is_real_and_stable(self):
        time = np.linspace(0.0, 5.0, 41)
        design = generalized_design(
            time, decay_rates=[0.4], oscillatory_pairs=[(0.2, 3.0)]
        )
        self.assertEqual(design.shape, (41, 4))
        self.assertTrue(np.isrealobj(design))
        self.assertTrue(np.isfinite(design).all())

    def test_oscillatory_fit_recovers_shared_pair(self):
        time = np.linspace(0.0, 8.0, 100)
        rows = []
        for group in range(4):
            for channel in range(2):
                envelope = np.exp(-0.18 * time)
                value = (
                    0.03 * group
                    + (1.0 + 0.1 * channel) * envelope * np.cos(2.4 * time)
                    + (0.2 + 0.03 * group) * envelope * np.sin(2.4 * time)
                )
                rows.append(CurveRecord(
                    f"u{group}_{channel}", f"g{group}", f"c{channel}", time, value
                ))
        result = fit_oscillatory_shared(rows, starts=5)
        damping, frequency = result["oscillatory_pairs"][0]
        self.assertAlmostEqual(damping, 0.18, places=6)
        self.assertAlmostEqual(frequency, 2.4, places=6)
        self.assertTrue(result["success"])

    def test_partial_sharing_preserves_geometric_centre(self):
        time = np.linspace(0.0, 6.0, 80)
        true_rates = {"g0": 0.22, "g1": 0.30, "g2": 0.40}
        rows = []
        for group, rate in true_rates.items():
            for channel in range(2):
                value = 0.02 * channel + (1.0 + 0.1 * channel) * np.exp(-rate * time)
                rows.append(CurveRecord(f"{group}_{channel}", group, str(channel), time, value))
        result = fit_partially_shared(rows, 1, shrinkage=1.0e-5, starts=3)
        fitted = np.asarray([value[0] for value in result["group_rates"].values()])
        centre = result["shared_geometric_centre"][0]
        self.assertAlmostEqual(float(np.exp(np.mean(np.log(fitted)))), centre, places=9)
        for group, rate in true_rates.items():
            self.assertLess(abs(np.log(result["group_rates"][group][0] / rate)), 0.03)

    def test_continuous_spectrum_generator_is_deterministic(self):
        time = np.linspace(0.0, 20.0, 60)
        first = continuous_spectrum_curves(time, groups=4, seed=17)
        second = continuous_spectrum_curves(time, groups=4, seed=17)
        self.assertEqual(len(first), 8)
        np.testing.assert_array_equal(first[3].value, second[3].value)
        self.assertTrue(np.isfinite(first[0].value).all())

    def test_group_conformal_uses_finite_sample_rank(self):
        calibration = {f"g{k}": float(k) for k in range(1, 20)}
        self.assertEqual(conformal_upper_quantile(list(calibration.values()), 0.10), 18.0)
        self.assertEqual(conformal_upper_pvalue(list(calibration.values()), 20.0), 0.05)
        audit = grouped_conformal_audit(calibration, {"held": 20.0}, alpha=0.10)
        self.assertTrue(audit["test_records"]["held"]["exceeds_threshold"])
        self.assertIn("exchangeable", audit["guarantee_scope"])


if __name__ == "__main__":
    unittest.main()
