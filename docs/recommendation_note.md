# Initial recommendation: cut-based baseline vs SPANET

The first SPANET iteration should be evaluated as an additive calibration tool, not as a replacement for the existing selection.

Recommended decision criteria:

- Pairing gain: compare SPANET pairing accuracy and dilution against the current two-jet baseline using the same event weights.
- Quality classifier: scan `P_reco` thresholds and measure efficiency, purity, and weighted yield stability.
- Regression utility: use neutrino/top regression outputs first as auxiliary quality observables unless they clearly improve calibrated observables.
- Independence: keep b-tag and jet-charge discriminators out of v1 inputs, then add them only in controlled ablations.

Adopt SPANET cuts only if they improve pairing dilution or reconstructibility purity without pathological sculpting in jet, lepton, MET, and top-mass observables.
