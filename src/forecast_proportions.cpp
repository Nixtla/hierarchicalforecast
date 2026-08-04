#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------- Floating-point strictness ----------
// The traversal below must stay bit-identical to the sequential NumPy
// reference implementation (`_reconcile_fcst_proportions` in
// hierarchicalforecast/methods.py). The rest of the extension is compiled
// with -ffast-math (/fp:fast on MSVC), which licenses the compiler to
// rewrite `x * fp / s` into `x * fp * (1.0 / s)` (reciprocal hoisting of
// the loop-invariant divisor), to re-associate/contract expressions, to
// reorder the pairwise-summation accumulators, and -- via
// finite-math-only -- to assume no NaN/inf, letting a NaN child sum slip
// past the `|sum| < 1e-8` guard into the equal-split branch instead of
// propagating NaN. This file is therefore excluded from fast-math at the
// build level (see STRICT_FP_SOURCES in setup.py: -fno-fast-math on
// gcc/clang, /fp:precise on MSVC); that per-source flag is the primary
// guarantee. The pragmas below are secondary, defense-in-depth only (they
// cannot fully undo a driver-level -ffast-math on every toolchain -- in
// particular they do not restore NaN semantics under clang's
// finite-math-only). The kernel is memory-bound scalar code, so fast-math
// buys nothing here anyway.
#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("no-fast-math")
#endif

// Clang scopes its FP pragmas to compound statements; used inside functions.
#if defined(__clang__) && __clang_major__ >= 14
#define HF_STRICT_FP _Pragma("clang fp reassociate(off) reciprocal(off) contract(off)")
#elif defined(__clang__)
#define HF_STRICT_FP _Pragma("clang fp reassociate(off) contract(off)")
#else
#define HF_STRICT_FP
#endif

namespace reconciliation {

namespace py = pybind11;

namespace {

// Faithful port of NumPy's `DOUBLE_pairwise_sum` (numpy/_core/src/umath/
// loops_utils.h.src): sequential below 8 elements, an 8-accumulator unrolled
// block up to PW_BLOCKSIZE = 128, and a recursive split (rounded down to a
// multiple of 8) above that. NumPy ships a single generic C source for this
// reduction (no per-architecture SIMD dispatch variant; verified for
// numpy >= 1.21 -- the unroll is designed so compilers can vectorize it
// *without changing the summation order*), so reproducing the association
// tree here yields sums bit-identical to `np.sum` over the same contiguous
// child-ordered values on every platform. Note: numpy < 2.0 seeded the n < 8
// base case with `0.` instead of `-0.0`; the difference is unobservable here
// (it only shows for an all `-0.0` child set, whose sum takes the
// `|sum| < 1e-8` equal-split branch either way). `val(i)` abstracts the
// gather `y_hat[child_idx[i], col]`; association depends only on `n`, not on
// memory layout.
template <typename F>
double numpy_pairwise_sum(const F &val, int64_t off, int64_t n) {
  HF_STRICT_FP
  if (n < 8) {
    // Start with -0.0 to preserve -0.0 (matches NumPy: `-0 + -0 == -0`
    // while `0 + -0 == 0`).
    double res = -0.0;
    for (int64_t i = 0; i < n; ++i) {
      res += val(off + i);
    }
    return res;
  } else if (n <= 128) {
    double r[8];
    for (int64_t j = 0; j < 8; ++j) {
      r[j] = val(off + j);
    }
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (int64_t j = 0; j < 8; ++j) {
        r[j] += val(off + i + j);
      }
    }
    double res =
        ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) {
      res += val(off + i);
    }
    return res;
  } else {
    int64_t n2 = n / 2;
    n2 -= n2 % 8;
    return numpy_pairwise_sum(val, off, n2) +
           numpy_pairwise_sum(val, off + n2, n - n2);
  }
}

} // namespace

// ---------- _forecast_proportions_traversal ----------
// Batched top-down "forecast proportions" tree traversal.
//
// Replaces the 4-level nested Python loop in `_reconcile_fcst_proportions`
// (top nodes x levels x parents x children), which was re-executed once per
// horizon column and, for bootstrap prediction intervals, once per bootstrap
// sample (a 100-10,000x multiplier). This kernel performs the full traversal
// for every (sample, horizon) pair in a single call.
//
// The child-node structure arrives flattened CSR-style (see
// `_flatten_child_nodes` in methods.py): parents are listed in traversal
// order (levels top-to-bottom, insertion order within each level), and
// `child_idx[child_indptr[p]:child_indptr[p+1]]` are the children of
// `parent_idx[p]`. Because every parent's reconciled value is written (as a
// top node or as a child of the previous level) before it is read, a single
// pass over this flat list is exactly equivalent to the reference's
// level-by-level traversal.
//
// Bit-exactness: per-parent child sums use `numpy_pairwise_sum` above (the
// same association tree as the reference's `y_hat[idx_childs].sum()`), and
// the per-child update is the reference's scalar `y * fp / s` chain with
// fast-math transforms disabled for this translation unit. OpenMP
// parallelism only spans the fully independent (sample, horizon) pairs --
// never inside a node's accumulation -- so results are deterministic and
// bit-identical to a serial run for any thread count.
//
// samples:      (n_samples, n_nodes, h) float64, base forecasts per sample.
// parent_idx:   (n_parents,) int64, node index of each parent.
// child_indptr: (n_parents + 1,) int64, CSR offsets into child_idx.
// child_idx:    (n_children_total,) int64, node index of each child.
// top_idx:      (n_top,) int64, node indices seeded with their own forecast.
// Returns:      (n_samples, n_nodes, h) float64 reconciled forecasts; nodes
//               not reached by the traversal stay 0, as in the reference.
py::array_t<double> forecast_proportions_traversal(
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &samples_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        &parent_idx_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        &child_indptr_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        &child_idx_arr,
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast>
        &top_idx_arr) {
  const auto sm = samples_arr.unchecked<3>();
  const auto par = parent_idx_arr.unchecked<1>();
  const auto iptr = child_indptr_arr.unchecked<1>();
  const auto cidx = child_idx_arr.unchecked<1>();
  const auto top = top_idx_arr.unchecked<1>();

  const py::ssize_t n_samples = sm.shape(0);
  const py::ssize_t n_nodes = sm.shape(1);
  const py::ssize_t h = sm.shape(2);
  const py::ssize_t n_parents = par.shape(0);
  const py::ssize_t n_children_total = cidx.shape(0);
  const py::ssize_t n_top = top.shape(0);

  if (iptr.shape(0) != n_parents + 1) {
    throw std::invalid_argument(
        "child_indptr must have shape (n_parents + 1,)");
  }
  if (iptr(0) != 0 || iptr(n_parents) != n_children_total) {
    throw std::invalid_argument(
        "child_indptr must start at 0 and end at len(child_idx)");
  }
  for (py::ssize_t p = 0; p < n_parents; ++p) {
    if (iptr(p) > iptr(p + 1)) {
      throw std::invalid_argument("child_indptr must be non-decreasing");
    }
    if (par(p) < 0 || par(p) >= n_nodes) {
      throw std::invalid_argument("parent_idx out of bounds");
    }
  }
  for (py::ssize_t i = 0; i < n_children_total; ++i) {
    if (cidx(i) < 0 || cidx(i) >= n_nodes) {
      throw std::invalid_argument("child_idx out of bounds");
    }
  }
  for (py::ssize_t k = 0; k < n_top; ++k) {
    if (top(k) < 0 || top(k) >= n_nodes) {
      throw std::invalid_argument("top_idx out of bounds");
    }
  }

  py::array_t<double> out_arr({n_samples, n_nodes, h});
  auto out = out_arr.mutable_unchecked<3>();
  double *out_ptr = out_arr.mutable_data();

  {
    // Manual release rather than this module's usual
    // `py::call_guard<py::gil_scoped_release>` (see reconciliation.cpp):
    // the validation and output allocation above deliberately stay under
    // the GIL; only this pure-compute region runs without it.
    py::gil_scoped_release release;

    const py::ssize_t total = n_samples * n_nodes * h;
    std::fill(out_ptr, out_ptr + total, 0.0);

    // Each (sample, column) pair is a fully independent serial traversal
    // writing to disjoint output entries: parallelizing across them is
    // deterministic and bit-identical to a serial run. The pair loop is
    // flattened by hand because MSVC's OpenMP 2.0 lacks `collapse(2)`.
    const py::ssize_t n_pairs = n_samples * h;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (py::ssize_t sc = 0; sc < n_pairs; ++sc) {
      HF_STRICT_FP
      const py::ssize_t s = sc / h;
      const py::ssize_t c = sc % h;
      for (py::ssize_t k = 0; k < n_top; ++k) {
        const int64_t t = top(k);
        out(s, t, c) = sm(s, t, c);
      }
      for (py::ssize_t p = 0; p < n_parents; ++p) {
        const double fcst_parent = out(s, par(p), c);
        const int64_t i0 = iptr(p);
        const int64_t i1 = iptr(p + 1);
        const double child_sum = numpy_pairwise_sum(
            [&](int64_t i) { return sm(s, cidx(i), c); }, i0, i1 - i0);
        if (std::abs(child_sum) < 1e-8) {
          const double n_children = static_cast<double>(i1 - i0);
          for (int64_t i = i0; i < i1; ++i) {
            out(s, cidx(i), c) = fcst_parent / n_children;
          }
        } else {
          for (int64_t i = i0; i < i1; ++i) {
            const int64_t ch = cidx(i);
            out(s, ch, c) = sm(s, ch, c) * fcst_parent / child_sum;
          }
        }
      }
    }
  }
  return out_arr;
}

// ---------- Module init ----------
void init_forecast_proportions(py::module_ &recon) {
  recon.def("_forecast_proportions_traversal", &forecast_proportions_traversal,
            py::arg("samples"), py::arg("parent_idx"), py::arg("child_indptr"),
            py::arg("child_idx"), py::arg("top_idx"),
            "Batched top-down forecast-proportions traversal over all "
            "(sample, horizon) pairs.");
}

} // namespace reconciliation

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif
