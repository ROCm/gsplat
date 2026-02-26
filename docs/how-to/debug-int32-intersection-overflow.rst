.. meta::
  :description: Root-cause analysis for int32 intersection overflow VM faults in GSplat rasterization.
  :keywords: GSplat, ROCm, VM fault, int32 overflow, rasterize_to_pixels_3dgs_bwd, tile_offsets

.. _gsplat-int32-intersection-overflow:

********************************************************************
Debugging int32 intersection overflow VM faults
********************************************************************

Summary
=======

GSplat rasterization kernels currently consume ``tile_offsets`` and
``flatten_ids`` through 32-bit index paths. When the intersection count
(``n_isects``) exceeds ``INT32_MAX`` (2,147,483,647), offset arithmetic can
wrap and produce negative or otherwise invalid ranges. On ROCm this can
manifest as a GPU VM fault (for example, ``Memory access fault by GPU``),
followed by process abort during device synchronization.


Observed failure signature
==========================

Typical runtime symptom:

* command runs correctly for smaller scenes
* process aborts for larger scenes at or beyond a threshold
* ROCm reports a memory access fault and aborts the process

Native backtrace usually shows:

* ``rocr::core::Runtime::VMFaultHandler(...)`` (abort path)
* HIP synchronization path in ``libamdhip64.so``
* GSplat launch path leading to
  ``rasterize_to_pixels_3dgs_bwd_kernel`` in ``gsplat/csrc.so``


Root cause in detail
====================

The backward rasterization kernel reads per-tile intersection ranges from
``tile_offsets`` (``int32_t``) and then indexes into ``flatten_ids``:

* ``range_start = tile_offsets[tile_id]``
* ``range_end = tile_offsets[tile_id + 1]`` (or ``n_isects`` for the last tile)
* derived ``idx`` values are used to load ``flatten_ids[idx]``

If host-side offsets exceed int32 range and are cast or interpreted as int32,
values wrap by 2^32. This can create invalid ranges and out-of-bounds indices.
Once a kernel thread dereferences an invalid global memory address, ROCm raises
a VM fault.

In a representative synthetic repro:

* ``n_isects`` is computed as ``NNZ_PER_CAM * C * ISECTS_PER_GAUSS``
* threshold crossing occurs near ``C=81`` for constants that place
  ``n_isects`` slightly above ``INT32_MAX``
* offset tensors contain wrapped negative values after int32 conversion
* kernel subsequently faults in backward rasterization


Why the current patch is a mitigation
=====================================

The current patch adds host-side validation before kernel launch and raises a
``TORCH_CHECK`` when ``flatten_ids.numel() > INT32_MAX``. This prevents hard
GPU faults and turns the failure into a deterministic, actionable error.

This is a safety mitigation, not a full functional fix for very large scenes.


Functional fix proposal (recommended)
=====================================

To support ``n_isects > INT32_MAX`` correctly, migrate intersection indexing to
64-bit end-to-end:

1. **Public/operator boundary**

   * Accept 64-bit intersection offsets/indices in C++ operator boundaries.
   * Keep explicit validation for dtype/range contracts.

2. **Kernel interfaces**

   * Update kernel signatures and launch parameters to use 64-bit types for:

     * ``n_isects``
     * ``tile_offsets``
     * loop/range/index temporaries derived from offsets

   * Avoid mixed signed/unsigned arithmetic that can silently wrap.

3. **All rasterization variants**

   * Apply consistently across 3DGS, 2DGS, and world-space rasterization paths
     (forward, backward, and index-only kernels) to prevent partial fixes.

4. **Additional invariants**

   * Validate monotonic non-decreasing offsets.
   * Validate final offset against ``flatten_ids.numel()``.


Suggested validation plan
=========================

* Boundary tests around ``INT32_MAX`` (exactly below, equal, above).
* Synthetic stress tests with controlled offset construction.
* ROCm and CUDA parity checks for behavior and numerical correctness.
* Negative tests that verify clean ``TORCH_CHECK`` failures for invalid inputs.


Practical guidance until full migration lands
=============================================

* Keep the fail-fast guard enabled to avoid hard VM faults.
* If large scenes are required immediately, split work into chunks such that
  each launch keeps ``n_isects <= INT32_MAX``.
