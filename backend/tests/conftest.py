"""
Shared pytest fixtures and configuration for engineering calculation tests.
"""

import pytest
import math


# Tolerance settings for numerical comparisons
# Engineering calculations typically need 1-2% relative tolerance
REL_TOLERANCE = 0.02  # 2% relative tolerance
ABS_TOLERANCE = 0.01  # Absolute tolerance for near-zero values


@pytest.fixture
def rel_tol():
    """Relative tolerance for floating point comparisons."""
    return REL_TOLERANCE


@pytest.fixture
def abs_tol():
    """Absolute tolerance for floating point comparisons."""
    return ABS_TOLERANCE


def approx_equal(actual, expected, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE):
    """
    Check if two values are approximately equal.

    Uses both relative and absolute tolerance for robustness.
    """
    if expected == 0:
        return abs(actual) <= abs_tol
    return abs(actual - expected) <= max(rel_tol * abs(expected), abs_tol)


# Common test parameters for engineering calculations
@pytest.fixture
def typical_material_a_params():
    """Typical parameters for cohesive material (phi=0 analysis)."""
    return {
        "cohesion": 50,  # kPa
        "friction_angle": 0,
        "unit_weight": 18,  # kN/m3
    }


@pytest.fixture
def typical_material_b_params():
    """Typical parameters for granular material."""
    return {
        "cohesion": 0,
        "friction_angle": 30,  # degrees
        "unit_weight": 18,  # kN/m3
    }


@pytest.fixture
def typical_material_c_params():
    """Typical parameters for composite material (cohesion + friction)."""
    return {
        "cohesion": 10,  # kPa
        "friction_angle": 25,  # degrees
        "unit_weight": 19,  # kN/m3
    }


# Reference values for engineering calculations
@pytest.fixture
def engineering_factor_reference_values():
    """
    Reference capacity factors for verification.

    Format: {phi: (Nc, Nq, Ngamma)}
    """
    return {
        0: (5.14, 1.00, 0.00),
        5: (6.49, 1.57, 0.45),
        10: (8.35, 2.47, 1.22),
        15: (10.98, 3.94, 2.65),
        20: (14.83, 6.40, 5.39),
        25: (20.72, 10.66, 10.88),
        30: (30.14, 18.40, 22.40),
        35: (46.12, 33.30, 48.03),
        40: (75.31, 64.20, 109.41),
        45: (133.88, 134.88, 271.76),
    }


@pytest.fixture
def coefficient_reference_values():
    """
    Reference lateral pressure coefficients.

    Ka = tan^2(45 - phi/2)
    Kp = tan^2(45 + phi/2)

    Format: {phi: (Ka, Kp)}
    """
    values = {}
    for phi in [0, 10, 20, 30, 35, 40, 45]:
        phi_rad = math.radians(phi)
        ka = math.tan(math.radians(45 - phi / 2)) ** 2
        kp = math.tan(math.radians(45 + phi / 2)) ** 2
        values[phi] = (round(ka, 4), round(kp, 4))
    return values


@pytest.fixture
def stefan_reference_values():
    """
    Reference frost depth calculations for Stefan equation verification.

    Using standard parameters:
    - k = 1.8 W/m-K
    - moisture = 20%
    - dry density = 1600 kg/m3
    """
    return {
        # freezing_index: expected_depth_m (approximate)
        500: 0.82,
        1000: 1.16,
        1500: 1.42,
        2000: 1.64,
    }
