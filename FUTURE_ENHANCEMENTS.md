# FUTURE ENHANCEMENTS - PHASE 2/3

**Created**: 2025-12-07  
**Status**: DEFERRED to future phases

---

## 1. ECEF Coordinate Transformation

**Current**: WGS84 planar approximation (`sfa/utils.py`)
```python
km_per_degree_lat = 111.1
km_per_degree_lon = 111.1 * np.cos(np.radians(mean_latitude))
```

**Upgrade to**: ECEF (Earth-Centered, Earth-Fixed) full 3D transformation

**Implementation**: Already exists in `sfa/core.py::GeodeticTransformer.geodetic_to_ecef()`

**Benefits**:
- Cascadia (40-50°N): Reduce error from ~1-2% to <0.1%
- High latitude regions: Better precision
- Mathematically rigorous for large regions

**Impact on Phase 1**:
- Current error <1% for most regions (acceptable)
- Only Cascadia affected (~0.01-0.02 units in D₂)
- NOT worth invalidating current validation

**Decision**: POSTPONED to Phase 2/3 (user confirmed 2025-12-07 09:45)

**When implementing**:
1. Modify `sfa/data.py::retrieve_catalog()` to use `GeodeticTransformer`
2. Re-run ALL regions to maintain consistency
3. Re-validate against documented (expect slight differences)
4. Update paper with methodology change

---

## 2. Performance Accelerations

**Pending**:
- Add `optimize_d2_computation()` calls at script start
- Enable Cython/Numba backends
- Implement adaptive subsampling for N>20k

**Status**: LOW PRIORITY for Phase 1

---

## 3. Synthetic Test Warnings

**Pending**:
- Update `SyntheticValidator` warnings
- Clarify 3D cube NaN = EXPECTED for N<5000
- Not a failure, just known limitation

**Status**: NICE TO HAVE

---

## 4. Multi-fractal Analysis

**Potential**:
- Extend from D₂ to full Rényi spectrum (D_q)
- Use `sfa/multifractal.py` (already implemented)
- Add to region analysis pipeline

**Status**: FUTURE FEATURE

---

**END OF FUTURE ENHANCEMENTS**
