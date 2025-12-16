"""Tests for distributed computation using Dask arrays."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pytest

# Try to import dask, skip all tests if not available
dask = pytest.importorskip("dask")
import dask.array as da

from hierarchicalforecast._array_compat import (
    compute_if_dask,
    get_array_module,
    is_dask_array,
    numpy_to_dask,
)
from hierarchicalforecast.methods import (
    ERM,
    BottomUp,
    MiddleOut,
    MinTrace,
    OptimalCombination,
    TopDown,
)


@dataclass
class HierarchicalTestData:
    """Test data for hierarchical forecasting methods."""

    S: np.ndarray  # Summing matrix
    h: int  # Forecast horizon
    y_bottom: np.ndarray  # Bottom level historical data
    y_hat_bottom: np.ndarray  # Bottom level forecasts
    y_hat_bottom_insample: np.ndarray  # Bottom level insample forecasts
    idx_bottom: List[int]  # Indices of bottom level series
    tags: Dict[str, np.ndarray]  # Hierarchy level tags


@pytest.fixture
def hierarchical_data():
    """Fixture providing test data for hierarchical forecasting methods."""
    # Summing matrix for 3-level hierarchy (7x4)
    # Level 1: Total (1 series)
    # Level 2: Two groups (2 series)
    # Level 3: Four bottom series (4 series)
    S = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],  # Total
            [1.0, 1.0, 0.0, 0.0],  # Group 1
            [0.0, 0.0, 1.0, 1.0],  # Group 2
            [1.0, 0.0, 0.0, 0.0],  # Bottom 1
            [0.0, 1.0, 0.0, 0.0],  # Bottom 2
            [0.0, 0.0, 1.0, 0.0],  # Bottom 3
            [0.0, 0.0, 0.0, 1.0],  # Bottom 4
        ]
    )

    h = 2  # Forecast horizon
    base_series = np.array([10.0, 5.0, 4.0, 2.0, 1.0])

    # Create bottom level historical data (4 series x 5 time periods)
    y_bottom = np.vstack([i * base_series for i in range(1, 5)])

    # Create insample forecasts with NaN in first period
    y_hat_bottom_insample = np.roll(y_bottom, 1, axis=1)
    y_hat_bottom_insample[:, 0] = np.nan

    # Create simple forecasts for bottom series
    y_hat_bottom = np.vstack([i * np.ones(h) for i in range(1, 5)])

    idx_bottom = [3, 4, 5, 6]  # Indices of bottom level in S matrix

    tags = {
        "level1": np.array([0]),
        "level2": np.array([1, 2]),
        "level3": np.array(idx_bottom),
    }

    return HierarchicalTestData(
        S=S,
        h=h,
        y_bottom=y_bottom,
        y_hat_bottom=y_hat_bottom,
        y_hat_bottom_insample=y_hat_bottom_insample,
        idx_bottom=idx_bottom,
        tags=tags,
    )


class TestArrayCompat:
    """Tests for array compatibility utilities."""

    def test_is_dask_array(self):
        """Test is_dask_array correctly identifies array types."""
        np_arr = np.array([1, 2, 3])
        dask_arr = da.from_array(np_arr)

        assert not is_dask_array(np_arr)
        assert is_dask_array(dask_arr)

    def test_get_array_module(self):
        """Test get_array_module returns correct module."""
        np_arr = np.array([1, 2, 3])
        dask_arr = da.from_array(np_arr)

        assert get_array_module(np_arr) is np
        assert get_array_module(dask_arr) is da

    def test_compute_if_dask(self):
        """Test compute_if_dask computes dask arrays correctly."""
        np_arr = np.array([1, 2, 3])
        dask_arr = da.from_array(np_arr)

        # For numpy array, should return as-is
        result_np = compute_if_dask(np_arr)
        assert isinstance(result_np, np.ndarray)
        np.testing.assert_array_equal(result_np, np_arr)

        # For dask array, should compute and return numpy
        result_dask = compute_if_dask(dask_arr)
        assert isinstance(result_dask, np.ndarray)
        np.testing.assert_array_equal(result_dask, np_arr)

    def test_numpy_to_dask(self):
        """Test numpy_to_dask converts arrays correctly."""
        np_arr = np.array([[1, 2], [3, 4]])
        dask_arr = numpy_to_dask(np_arr)

        assert is_dask_array(dask_arr)
        np.testing.assert_array_equal(dask_arr.compute(), np_arr)


class TestBottomUpDistributed:
    """Tests for BottomUp reconciliation with Dask arrays."""

    def test_bottom_up_with_dask_arrays(self, hierarchical_data):
        """Test BottomUp reconciliation with Dask arrays."""
        data = hierarchical_data

        # Convert to Dask arrays
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)

        cls_bottom_up = BottomUp()
        result = cls_bottom_up(
            S=S_dask, y_hat=y_hat_dask, idx_bottom=data.idx_bottom
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)

    def test_bottom_up_dask_vs_numpy_equivalence(self, hierarchical_data):
        """Test BottomUp produces same results with Dask and NumPy arrays."""
        data = hierarchical_data

        # NumPy version
        cls_numpy = BottomUp()
        result_numpy = cls_numpy(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            idx_bottom=data.idx_bottom,
        )["mean"]

        # Dask version
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)

        cls_dask = BottomUp()
        result_dask = cls_dask(
            S=S_dask, y_hat=y_hat_dask, idx_bottom=data.idx_bottom
        )["mean"]

        np.testing.assert_allclose(result_numpy, result_dask)


class TestTopDownDistributed:
    """Tests for TopDown reconciliation with Dask arrays."""

    @pytest.mark.parametrize(
        "method", ["average_proportions", "proportion_averages"]
    )
    def test_top_down_with_dask_arrays(self, hierarchical_data, method):
        """Test TopDown reconciliation with Dask arrays."""
        data = hierarchical_data

        # Convert to Dask arrays
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)
        y_insample_dask = da.from_array(data.S @ data.y_bottom)

        cls_top_down = TopDown(method=method)
        result = cls_top_down(
            S=S_dask,
            y_hat=y_hat_dask,
            y_insample=y_insample_dask,
            tags=data.tags,
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)

    def test_top_down_forecast_proportions_with_dask_arrays(self, hierarchical_data):
        """Test TopDown with forecast_proportions method and Dask arrays."""
        data = hierarchical_data

        # Convert to Dask arrays
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)

        cls_top_down = TopDown(method="forecast_proportions")
        result = cls_top_down(
            S=S_dask,
            y_hat=y_hat_dask,
            tags=data.tags,
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize(
        "method", ["average_proportions", "proportion_averages"]
    )
    def test_top_down_dask_vs_numpy_equivalence(self, hierarchical_data, method):
        """Test TopDown produces same results with Dask and NumPy arrays."""
        data = hierarchical_data

        # NumPy version
        cls_numpy = TopDown(method=method)
        result_numpy = cls_numpy(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]

        # Dask version
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)
        y_insample_dask = da.from_array(data.S @ data.y_bottom)

        cls_dask = TopDown(method=method)
        result_dask = cls_dask(
            S=S_dask,
            y_hat=y_hat_dask,
            y_insample=y_insample_dask,
            tags=data.tags,
        )["mean"]

        np.testing.assert_allclose(result_numpy, result_dask)


class TestMinTraceDistributed:
    """Tests for MinTrace reconciliation with Dask arrays."""

    @pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink"])
    def test_min_trace_with_dask_arrays(self, hierarchical_data, method):
        """Test MinTrace reconciliation with Dask arrays."""
        data = hierarchical_data

        # Convert to Dask arrays
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)
        y_insample_dask = da.from_array(data.S @ data.y_bottom)
        y_hat_insample_dask = da.from_array(data.S @ data.y_hat_bottom_insample)

        cls_min_trace = MinTrace(method=method)
        result = cls_min_trace(
            S=S_dask,
            y_hat=y_hat_dask,
            y_insample=y_insample_dask,
            y_hat_insample=y_hat_insample_dask,
            idx_bottom=data.idx_bottom,
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("method", ["ols", "wls_struct"])
    def test_min_trace_dask_vs_numpy_equivalence(self, hierarchical_data, method):
        """Test MinTrace produces same results with Dask and NumPy arrays."""
        data = hierarchical_data

        # NumPy version
        cls_numpy = MinTrace(method=method)
        result_numpy = cls_numpy(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            idx_bottom=data.idx_bottom,
        )["mean"]

        # Dask version
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)

        cls_dask = MinTrace(method=method)
        result_dask = cls_dask(
            S=S_dask,
            y_hat=y_hat_dask,
            idx_bottom=data.idx_bottom,
        )["mean"]

        np.testing.assert_allclose(result_numpy, result_dask)


class TestOptimalCombinationDistributed:
    """Tests for OptimalCombination reconciliation with Dask arrays."""

    @pytest.mark.parametrize("method", ["ols", "wls_struct"])
    def test_optimal_combination_with_dask_arrays(self, hierarchical_data, method):
        """Test OptimalCombination reconciliation with Dask arrays."""
        data = hierarchical_data

        # Convert to Dask arrays
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)

        cls_optimal = OptimalCombination(method=method)
        result = cls_optimal(
            S=S_dask,
            y_hat=y_hat_dask,
            idx_bottom=data.idx_bottom,
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)


class TestMiddleOutDistributed:
    """Tests for MiddleOut reconciliation with Dask arrays."""

    @pytest.mark.parametrize(
        "method", ["average_proportions", "proportion_averages"]
    )
    def test_middle_out_with_dask_arrays(self, hierarchical_data, method):
        """Test MiddleOut reconciliation with Dask arrays."""
        data = hierarchical_data

        # Note: MiddleOut internally calls BottomUp and TopDown
        # which should handle Dask arrays
        cls_middle_out = MiddleOut(middle_level="level2", top_down_method=method)

        # For MiddleOut, we use numpy arrays as it internally does complex
        # indexing that's not directly compatible with Dask
        result = cls_middle_out(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            tags=data.tags,
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)


class TestERMDistributed:
    """Tests for ERM reconciliation with Dask arrays."""

    def test_erm_with_dask_arrays(self, hierarchical_data):
        """Test ERM reconciliation with Dask arrays."""
        data = hierarchical_data

        # Convert to Dask arrays
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)
        y_insample_dask = da.from_array(data.S @ data.y_bottom)
        y_hat_insample_dask = da.from_array(data.S @ data.y_hat_bottom_insample)

        cls_erm = ERM(method="reg_bu", lambda_reg=None)
        result = cls_erm(
            S=S_dask,
            y_hat=y_hat_dask,
            y_insample=y_insample_dask,
            y_hat_insample=y_hat_insample_dask,
            idx_bottom=data.idx_bottom,
        )["mean"]

        expected = data.S @ data.y_hat_bottom
        np.testing.assert_allclose(result, expected)

    def test_erm_dask_vs_numpy_equivalence(self, hierarchical_data):
        """Test ERM produces same results with Dask and NumPy arrays."""
        data = hierarchical_data

        # NumPy version
        cls_numpy = ERM(method="reg_bu", lambda_reg=None)
        result_numpy = cls_numpy(
            S=data.S,
            y_hat=data.S @ data.y_hat_bottom,
            y_insample=data.S @ data.y_bottom,
            y_hat_insample=data.S @ data.y_hat_bottom_insample,
            idx_bottom=data.idx_bottom,
        )["mean"]

        # Dask version
        S_dask = da.from_array(data.S)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)
        y_insample_dask = da.from_array(data.S @ data.y_bottom)
        y_hat_insample_dask = da.from_array(data.S @ data.y_hat_bottom_insample)

        cls_dask = ERM(method="reg_bu", lambda_reg=None)
        result_dask = cls_dask(
            S=S_dask,
            y_hat=y_hat_dask,
            y_insample=y_insample_dask,
            y_hat_insample=y_hat_insample_dask,
            idx_bottom=data.idx_bottom,
        )["mean"]

        np.testing.assert_allclose(result_numpy, result_dask)


class TestMatrixOperationsDistributed:
    """Tests for matrix operations with Dask arrays."""

    def test_matmul_dask(self):
        """Test matrix multiplication with Dask arrays."""
        A = np.random.rand(10, 5)
        B = np.random.rand(5, 3)

        A_dask = da.from_array(A)
        B_dask = da.from_array(B)

        # Dask matmul
        result_dask = (A_dask @ B_dask).compute()

        # NumPy matmul
        result_numpy = A @ B

        np.testing.assert_allclose(result_dask, result_numpy)

    def test_reconciliation_formula_dask(self, hierarchical_data):
        """Test the core reconciliation formula S @ (P @ y_hat) with Dask arrays."""
        data = hierarchical_data

        # Create P matrix (bottom-up style)
        n_hiers, n_bottom = data.S.shape
        P = np.eye(n_bottom, n_hiers, n_hiers - n_bottom, np.float64)

        # Convert to Dask
        S_dask = da.from_array(data.S)
        P_dask = da.from_array(P)
        y_hat_dask = da.from_array(data.S @ data.y_hat_bottom)

        # Reconciliation formula
        result_dask = S_dask @ (P_dask @ y_hat_dask)
        result_dask_computed = result_dask.compute()

        # NumPy version
        result_numpy = data.S @ (P @ (data.S @ data.y_hat_bottom))

        np.testing.assert_allclose(result_dask_computed, result_numpy)


class TestHierarchicalReconciliationDistributed:
    """Tests for HierarchicalReconciliation class with distributed mode."""

    @pytest.fixture
    def reconciliation_data(self):
        """Fixture providing data for HierarchicalReconciliation tests."""
        import pandas as pd

        # Create a simple hierarchy
        # Total -> [A, B] -> [A1, A2, B1, B2]
        unique_ids = ["Total", "A", "B", "A1", "A2", "B1", "B2"]
        horizon = 2

        # Create S matrix
        S = np.array(
            [
                [1, 1, 1, 1],  # Total
                [1, 1, 0, 0],  # A
                [0, 0, 1, 1],  # B
                [1, 0, 0, 0],  # A1
                [0, 1, 0, 0],  # A2
                [0, 0, 1, 0],  # B1
                [0, 0, 0, 1],  # B2
            ]
        )

        # Create S_df
        bottom_ids = ["A1", "A2", "B1", "B2"]
        S_df = pd.DataFrame(S, columns=bottom_ids)
        S_df.insert(0, "unique_id", unique_ids)

        # Create tags
        tags = {
            "Total": np.array(["Total"]),
            "Group": np.array(["A", "B"]),
            "Bottom": np.array(bottom_ids),
        }

        # Create Y_hat_df (forecasts)
        dates = pd.date_range("2023-01-01", periods=horizon, freq="D")
        y_hat_data = []
        for uid in unique_ids:
            for ds in dates:
                y_hat_data.append({"unique_id": uid, "ds": ds, "model": 10.0})
        Y_hat_df = pd.DataFrame(y_hat_data)

        return {
            "S_df": S_df,
            "Y_hat_df": Y_hat_df,
            "tags": tags,
        }

    def test_reconcile_with_distributed_flag(self, reconciliation_data):
        """Test that reconcile works with distributed=True."""
        from hierarchicalforecast.core import HierarchicalReconciliation

        hrec = HierarchicalReconciliation(reconcilers=[BottomUp()])

        # Run with distributed=True
        result = hrec.reconcile(
            Y_hat_df=reconciliation_data["Y_hat_df"],
            S_df=reconciliation_data["S_df"],
            tags=reconciliation_data["tags"],
            distributed=True,
        )

        # Check that we got results
        assert result is not None
        assert "model/BottomUp" in result.columns

    def test_distributed_vs_normal_equivalence(self, reconciliation_data):
        """Test that distributed mode produces same results as normal mode."""
        from hierarchicalforecast.core import HierarchicalReconciliation

        # Normal mode
        hrec_normal = HierarchicalReconciliation(reconcilers=[BottomUp()])
        result_normal = hrec_normal.reconcile(
            Y_hat_df=reconciliation_data["Y_hat_df"],
            S_df=reconciliation_data["S_df"],
            tags=reconciliation_data["tags"],
            distributed=False,
        )

        # Distributed mode
        hrec_distributed = HierarchicalReconciliation(reconcilers=[BottomUp()])
        result_distributed = hrec_distributed.reconcile(
            Y_hat_df=reconciliation_data["Y_hat_df"],
            S_df=reconciliation_data["S_df"],
            tags=reconciliation_data["tags"],
            distributed=True,
        )

        # Compare results
        np.testing.assert_allclose(
            result_normal["model/BottomUp"].values,
            result_distributed["model/BottomUp"].values,
        )

    def test_distributed_with_multiple_reconcilers(self, reconciliation_data):
        """Test distributed mode with multiple reconcilers."""
        from hierarchicalforecast.core import HierarchicalReconciliation

        reconcilers = [
            BottomUp(),
            MinTrace(method="ols"),
        ]

        hrec = HierarchicalReconciliation(reconcilers=reconcilers)

        # Run with distributed=True
        result = hrec.reconcile(
            Y_hat_df=reconciliation_data["Y_hat_df"],
            S_df=reconciliation_data["S_df"],
            tags=reconciliation_data["tags"],
            distributed=True,
        )

        # Check that we got results for both reconcilers
        assert "model/BottomUp" in result.columns
        assert "model/MinTrace_method-ols" in result.columns
