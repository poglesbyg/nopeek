# Changelog

## 0.1.1

Both of these were found by running nopeek against a real pipeline rather than
by testing it in isolation, and neither was visible from inside the suite.

### Fixed

- **Poisoning silently did nothing on float32, float16 and the narrow integer
  types.** pandas refuses an assignment that would lose precision rather than
  downcasting, so generating poison as float64 raised for every one of those
  columns — and the handler recorded it as an "unsupported dtype" note instead
  of a failure. On a float32 feature matrix, which is what most of them are,
  the poison strategy was a no-op while reporting success. Replacement values
  are now generated in the column's own dtype, with a magnitude capped by what
  that dtype can represent so float16 is not poisoned with `inf`. Every skipped
  column now names the actual exception rather than blaming the dtype.

- **Float round-off was reported as leakage.** The default `atol` of 0 meant no
  relative tolerance could ever call `0.0` and `1.6e-7` close, so a rolling
  standard deviation over a constant window — 0 in exact arithmetic, about one
  ulp in float32 — came back as a finding. The absolute floor is now derived
  from the data's own dtype. Differences below it are reported in
  `report.notes` rather than dropped, and `atol=0.0` restores exact comparison.

### Changed

- `report.notes` distinguishes "identical", "different within numerical
  resolution", and "different" — the middle case used to be indistinguishable
  from the last.
