# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HierarchicalForecast is a Python library for probabilistic hierarchical forecasting with statistical and econometric methods. It provides reconciliation methods that ensure coherent forecasts across different aggregation levels in hierarchical or grouped time series data.

## Development Commands

### Environment Setup
```bash
# Create virtual environment with Python 3.10+
uv venv --python 3.10
source .venv/bin/activate  # MacOS/Linux
# .\.venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
pre-commit run --files hierarchicalforecast/*
```

### Testing
```bash
# Run all tests (requires 70% coverage)
uv run pytest

# Run specific test file
uv run pytest tests/test_methods.py

# Run specific test function
uv run pytest tests/test_methods.py::test_mint_ensemble_forecast_reconciliation

# Run tests matching pattern
uv run pytest -k "emint"

# Run with verbose output and show print statements
uv run pytest -v -s

# Generate coverage report
uv run pytest --cov=hierarchicalforecast --cov-report=html
```

### Code Quality
```bash
# Format and lint (automatically fixes issues)
ruff check --fix
ruff format

# Type checking (if needed)
mypy hierarchicalforecast/
```

### Documentation
```bash
# Build all documentation
make all_docs

# Preview documentation locally
make preview_docs
```

## Architecture

### Core Components

**`hierarchicalforecast/core.py`** - `HierarchicalReconciliation` class
- Main entry point for users
- Orchestrates reconciliation across multiple methods
- Handles DataFrame operations using Narwhals for polars/pandas compatibility
- Validates hierarchical structure and summing matrix S

**`hierarchicalforecast/methods.py`** - Reconciliation methods
- All reconciliation algorithms inherit from `HReconciler` base class
- Each method implements `fit()` and `predict()` for two-stage workflow
- Methods compute projection matrix `P` and weight matrix `W`
- Core reconciliation formula: `y_reconciled = S @ P @ y_hat`

**`hierarchicalforecast/probabilistic_methods.py`** - Probabilistic forecasting
- Extends base methods with uncertainty quantification
- `Normality`, `Bootstrap`, `PERMBU` samplers
- Generates coherent prediction intervals

**`hierarchicalforecast/utils.py`** - Utility functions
- Hierarchy validation and summing matrix construction
- Aggregation and disaggregation operations
- Sparse matrix utilities for large hierarchies

**`hierarchicalforecast/evaluation.py`** - Evaluation metrics
- Compute accuracy metrics across hierarchy levels
- Per-level and aggregate evaluation

### Key Reconciliation Methods

**BottomUp / TopDown / MiddleOut** (`methods.py:46-1251`)
- Simple aggregation-based reconciliation
- No optimization required
- Fast, interpretable baselines

**MinTrace** (`methods.py:1281-2150`)
- Optimal reconciliation minimizing trace of forecast error covariance
- Multiple variants: `ols`, `wls_struct`, `wls_var`, `mint_shrink`, `mint_cov`, `emint`
- Supports non-negative constraints via quadratic programming

**ERM** (`methods.py:2151+`)
- Empirical Risk Minimization with L1 regularization
- Learns optimal reconciliation matrix from data

### Data Flow

1. **Input**: Base forecasts `y_hat` (unreconciled, potentially incoherent)
2. **Summing Matrix S**: Defines aggregation structure (n_hiers × n_bottom)
3. **Projection Matrix P**: Method-specific (n_bottom × n_hiers)
4. **Reconciliation**: `y_tilde = S @ P @ y_hat`
5. **Output**: Coherent forecasts satisfying `y_top = sum(y_bottom)`

## Important Implementation Details

### Non-negative Reconciliation
- When `nonnegative=True`, uses quadratic programming solver (clarabel)
- Constraints: `P @ y_hat >= 0` and `S @ P = S @ P_base`
- Only compatible with `intervals_method="normality"` for probabilistic forecasts
- Bootstrap/PERMBU samplers incompatible with nonnegative constraints

### Type Annotations
- Narwhals frames use `FrameT` generic for polars/pandas compatibility

## Testing Patterns

### Test Structure
- `tests/conftest.py` - Fixtures for hierarchical test data
- `hierarchical_data` fixture provides S matrix, bottom forecasts, tags
- `interval_reconciler_args` fixture for probabilistic methods

### Common Test Patterns
```python
# Test coherence (aggregation constraints)
bottom_forecasts = result[idx_bottom, :]
aggregated = S @ bottom_forecasts
np.testing.assert_allclose(result, aggregated, rtol=1e-10)

# Test non-negative constraint
assert np.all(result >= -1e-6)

# Test error handling
with pytest.raises(ValueError, match="expected error message"):
    reconciler.method_call(...)
```

### Coverage Requirements
- Minimum 70% coverage (configured in pyproject.toml)
- Focus on testing: fit/predict patterns, error paths, edge cases, coherence validation

## Common Gotchas

1. **EMinT without nonnegative**: Only guarantees coherence, not exact forecast recovery (unlike other MinTrace methods)

2. **Empty arrays after NaN filtering**: Always validate that data remains after removing NaN observations (especially in emint/ensemble methods)

3. **Summing matrix S**: Shape is (n_hiers, n_bottom) where n_hiers includes aggregated + bottom levels

4. **idx_bottom**: When `None`, inferred as last n_bottom rows; must be validated against S dimensions

5. **Method state mutation**: Avoid temporarily changing `self.method` during execution (not thread-safe); prefer helper methods with explicit parameters

## Git Workflow

### Branch Naming
- Features: `feature/descriptive-name` or `feat/descriptive-name`
- Fixes: `fix/descriptive-name`
- Issues: `issue/issue-number`

### Commit Guidelines
- Run pre-commit hooks before committing
- Keep commits focused and atomic
- Test coverage must not decrease

### PR Guidelines
- Each PR should be focused on one feature/fix
- Do not mix style changes with functional changes
- Respond to review comments with commits in the same PR
- CI runs pytest with coverage checks - must maintain 70% minimum
