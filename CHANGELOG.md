# Changelog

## [0.6.0] - 2026-01-27
### Added
- `lambda_corr_nb(...)`: Numba-compatible entrypoint (prevalidated inputs).
- CHANGELOG.md

### Changed
- Docs: expanded module docstring and added to README example.

## [0.8.0] - 2026-03-03

### Changed
- lambda_corr: Changed p-value calculation from Edgeworth expansion to Beta-mixed distribution. 
               This is much more accurate. Small n uses exact lookup table.
- _core, _pvals: Moved functions to these files for clarity.
-/example: Added real-world example using the Lambda correlation to do the Telescope Array Supergalactic 
           Structure analysis on Pierre Auger Observatory cosmic ray public data.
-/HPC: Added High-Performance Computing fully distributed code that was used to fit Beta distributions and
       SLURM submission scripts.
- README: Reflects the p-value changes and some other editing.
