from __future__ import annotations

import unittest

import numpy as np

import dfsc


class DfscExperimentalDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracks = (
            np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
            np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]),
            np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        )

    def test_split_is_deterministic_and_disjoint(self) -> None:
        first = dfsc.split_trajectories(self.tracks, train_fraction=2 / 3, seed=4)
        second = dfsc.split_trajectories(self.tracks, train_fraction=2 / 3, seed=4)
        self.assertEqual([id(track) for track in first[0]], [id(track) for track in second[0]])
        self.assertFalse({id(track) for track in first[0]} & {id(track) for track in first[1]})

    def test_observables_match_known_linear_tracks(self) -> None:
        observables = dfsc.empirical_spt_observables(self.tracks[:2], [1, 2], wave_number=0.2)
        np.testing.assert_allclose(observables.msd, [1.0, 4.0])
        np.testing.assert_allclose(observables.scattering, [(1.0 + np.cos(0.2)) / 2, (1.0 + np.cos(0.4)) / 2])
        np.testing.assert_array_equal(observables.sample_counts, [6, 4])

    def test_wave_number_uses_median_step(self) -> None:
        self.assertAlmostEqual(dfsc.estimate_wave_number(self.tracks[:2]), 0.5)


if __name__ == "__main__":
    unittest.main()
