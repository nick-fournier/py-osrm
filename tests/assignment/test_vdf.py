"""Tests for BiParabolicVDF.

Validates the bi-parabolic flow-density model against known theoretical
properties from Fournier et al.
"""

import numpy as np
import pytest

from osrm.assignment.vdf import BiParabolicVDF


@pytest.fixture
def vdf():
    return BiParabolicVDF(kc_ratio=1.0 / 3.0, min_speed_kmh=5.0)


# --- Scalar boundary conditions ---


class TestBoundaryConditions:
    def test_freeflow_at_zero_density(self, vdf):
        """v(0) = v_f"""
        v = vdf.density_to_speed(
            k=np.array([0.0]),
            v_f=np.array([60.0]),
            k_j=np.array([150.0]),
        )
        np.testing.assert_allclose(v, [60.0], atol=1e-10)

    def test_critical_speed_at_kc(self, vdf):
        """v(k_c) = v_f / 2"""
        k_j = np.array([150.0])
        k_c = vdf.critical_density(k_j)
        v = vdf.density_to_speed(
            k=k_c,
            v_f=np.array([60.0]),
            k_j=k_j,
        )
        np.testing.assert_allclose(v, [30.0], atol=1e-10)

    def test_zero_speed_at_jam(self, vdf):
        """v(k_j) should be at or near min_speed (clamped)"""
        v = vdf.density_to_speed(
            k=np.array([150.0]),
            v_f=np.array([60.0]),
            k_j=np.array([150.0]),
        )
        np.testing.assert_allclose(v, [5.0], atol=1e-10)

    def test_capacity_flow_at_kc(self, vdf):
        """q(k_c) = q_c = v_f * k_c / 2"""
        k_j = np.array([150.0])
        v_f = np.array([60.0])
        k_c = vdf.critical_density(k_j)
        q_c = vdf.capacity_flow(v_f, k_j)
        q = vdf.density_to_flow(k_c, v_f, k_j)
        np.testing.assert_allclose(q, q_c, atol=1e-10)


class TestContinuity:
    def test_c1_continuity_value(self, vdf):
        """v(k_c⁻) ≈ v(k_c⁺): value matches at junction."""
        k_j = np.array([150.0])
        v_f = np.array([60.0])
        k_c = vdf.critical_density(k_j)
        eps = 1e-6
        v_minus = vdf.density_to_speed(k_c - eps, v_f, k_j)
        v_plus = vdf.density_to_speed(k_c + eps, v_f, k_j)
        np.testing.assert_allclose(v_minus, v_plus, atol=1e-3)

    def test_c1_continuity_derivative(self, vdf):
        """Slope dv/dk should match at k_c (both sides → same slope)."""
        k_j = np.array([150.0])
        v_f = np.array([60.0])
        k_c = vdf.critical_density(k_j)
        eps = 1e-4
        # Numerical derivative from left
        v_l1 = vdf.density_to_speed(k_c - 2 * eps, v_f, k_j)
        v_l2 = vdf.density_to_speed(k_c - eps, v_f, k_j)
        slope_left = (v_l2 - v_l1) / eps
        # Numerical derivative from right
        v_r1 = vdf.density_to_speed(k_c + eps, v_f, k_j)
        v_r2 = vdf.density_to_speed(k_c + 2 * eps, v_f, k_j)
        slope_right = (v_r2 - v_r1) / eps
        np.testing.assert_allclose(slope_left, slope_right, atol=0.1)


class TestMonotonicity:
    def test_speed_nonincreasing(self, vdf):
        """Speed should be non-increasing over [0, k_j]."""
        k_j = 150.0
        v_f = 60.0
        k = np.linspace(0, k_j, 1000)
        v = vdf.density_to_speed(k, np.full_like(k, v_f), np.full_like(k, k_j))
        diffs = np.diff(v)
        assert np.all(diffs <= 1e-10), "Speed must be non-increasing with density"


class TestFlowToDensityInverse:
    def test_zero_flow_gives_zero_density(self, vdf):
        """k(0) = 0"""
        k = vdf.flow_to_density(
            q=np.array([0.0]),
            v_f=np.array([60.0]),
            k_j=np.array([150.0]),
        )
        np.testing.assert_allclose(k, [0.0], atol=1e-10)

    def test_capacity_flow_gives_kc(self, vdf):
        """k(q_c) = k_c"""
        v_f = np.array([60.0])
        k_j = np.array([150.0])
        q_c = vdf.capacity_flow(v_f, k_j)
        k_c = vdf.critical_density(k_j)
        k = vdf.flow_to_density(q_c, v_f, k_j)
        np.testing.assert_allclose(k, k_c, atol=1e-10)

    def test_overcapacity_clamped_to_kc(self, vdf):
        """k(q > q_c) should clamp to k_c."""
        v_f = np.array([60.0])
        k_j = np.array([150.0])
        q_c = vdf.capacity_flow(v_f, k_j)
        k = vdf.flow_to_density(q_c * 1.5, v_f, k_j)
        k_c = vdf.critical_density(k_j)
        np.testing.assert_allclose(k, k_c, atol=1e-10)

    def test_roundtrip_q_to_k_to_q(self, vdf):
        """q → k(q) → q(k) = q for flows in [0, q_c]."""
        v_f = np.full(50, 60.0)
        k_j = np.full(50, 150.0)
        q_c = vdf.capacity_flow(v_f, k_j)
        q_in = np.linspace(0, q_c[0] * 0.99, 50)
        k = vdf.flow_to_density(q_in, v_f, k_j)
        q_out = vdf.density_to_flow(k, v_f, k_j)
        np.testing.assert_allclose(q_out, q_in, atol=1e-6)

    def test_numerical_accuracy(self, vdf):
        """Verify |q - k * v(k)| < ε for random flows."""
        rng = np.random.default_rng(42)
        v_f = rng.uniform(30, 120, 100)
        k_j = rng.uniform(80, 200, 100)
        q_c = vdf.capacity_flow(v_f, k_j)
        q = rng.uniform(0, 0.95, 100) * q_c
        k = vdf.flow_to_density(q, v_f, k_j)
        v = vdf.density_to_speed(k, v_f, k_j)
        q_check = k * v
        np.testing.assert_allclose(q_check, q, atol=1e-4)


class TestVectorized:
    def test_vectorized_matches_scalar(self, vdf):
        """Vectorized output should match element-wise scalar evaluation."""
        rng = np.random.default_rng(99)
        n = 20
        v_f = rng.uniform(30, 120, n)
        k_j = rng.uniform(80, 200, n)
        k = rng.uniform(0, 0.9, n) * k_j  # stay below jam

        v_vec = vdf.density_to_speed(k, v_f, k_j)
        v_scalar = np.array([
            vdf.density_to_speed(
                np.array([k[i]]),
                np.array([v_f[i]]),
                np.array([k_j[i]]),
            )[0]
            for i in range(n)
        ])
        np.testing.assert_allclose(v_vec, v_scalar, atol=1e-10)


class TestFlowToSpeed:
    def test_zero_flow_freeflow(self, vdf):
        """Zero flow → free-flow speed."""
        v = vdf.flow_to_speed(
            q=np.array([0.0]),
            v_f=np.array([60.0]),
            k_j=np.array([150.0]),
        )
        np.testing.assert_allclose(v, [60.0], atol=1e-10)

    def test_capacity_flow_half_speed(self, vdf):
        """Flow at capacity → v_f / 2."""
        v_f = np.array([60.0])
        k_j = np.array([150.0])
        q_c = vdf.capacity_flow(v_f, k_j)
        v = vdf.flow_to_speed(q_c, v_f, k_j)
        np.testing.assert_allclose(v, [30.0], atol=1e-10)
