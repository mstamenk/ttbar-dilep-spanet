# HDF5 schema

The converter writes a native SPANET v2 HDF5 file for OS e-mu, exactly-two-jet dileptonic ttbar events. The native groups mirror [configs/ttbar_dilep_event.yaml](../configs/ttbar_dilep_event.yaml).

## Groups

### `INPUTS`

SPANET reads one dataset per feature.

| Dataset | Shape | Description |
| --- | --- | --- |
| `INPUTS/Jets/MASK` | `(N, 2)` | valid jet mask, true for both selected jets |
| `INPUTS/Jets/pt` | `(N, 2)` | selected jet pT |
| `INPUTS/Jets/eta` | `(N, 2)` | selected jet eta |
| `INPUTS/Jets/sin_phi` | `(N, 2)` | selected jet sin(phi) |
| `INPUTS/Jets/cos_phi` | `(N, 2)` | selected jet cos(phi) |
| `INPUTS/Jets/mass` | `(N, 2)` | selected jet mass |
| `INPUTS/Leptons/*` | `(N,)` | global electron and muon features |
| `INPUTS/Met/*` | `(N,)` | global MET features |
| `INPUTS/Event/*` | `(N,)` | global engineered event features |

### `TARGETS`

SPANET assignment targets are event particles with one daughter each:

| Dataset | Shape | Description |
| --- | --- | --- |
| `TARGETS/TopE/b` | `(N,)` | jet index assigned to the electron-side top, or `-1` |
| `TARGETS/TopE/MASK` | `(N,)` | valid electron-side assignment |
| `TARGETS/TopMu/b` | `(N,)` | jet index assigned to the muon-side top, or `-1` |
| `TARGETS/TopMu/MASK` | `(N,)` | valid muon-side assignment |

Partial events are retained by setting one or both masks false instead of dropping the event.

### `REGRESSIONS`

| Dataset | Shape | Description |
| --- | --- | --- |
| `REGRESSIONS/EVENT/nu_px` etc. | `(N,)` | event-level neutrino and top regression targets |

Missing regression values are stored as `NaN`, which SPANET masks in the regression loss.

### `CLASSIFICATIONS`

| Dataset | Shape | Description |
| --- | --- | --- |
| `CLASSIFICATIONS/EVENT/reco_quality` | `(N,)` | `1`: fully matched/reconstructible, `0`: partial, `-1`: unavailable |

SPANET treats `-1` as ignored for classification loss.

## Analysis Compatibility Groups

The file also stores lower-case compatibility groups for sanity scripts and external checks.

### `inputs`

| Dataset | Shape | Description |
| --- | --- | --- |
| `inputs/jets` | `(N, 2, 4)` | selected jets ordered as in the source ntuple: `pt, eta, phi, mass` |
| `inputs/leptons` | `(N, 2, 5)` | electron then muon: `pt, eta, phi, mass, charge` |
| `inputs/met` | `(N, 2)` | MET `pt, phi` |
| `inputs/event` | `(N, 8)` | engineered event features: `m_ee_unused, dphi_emu, dr_emu, dphi_jj, dr_jj, m_jj, m_ej_min, m_muj_min` |

`m_ee_unused` is reserved and currently filled with zero so the schema can stay stable if an event-level scalar is added later.

### `targets`

| Dataset | Shape | Description |
| --- | --- | --- |
| `targets/pair_label` | `(N,)` | `0`: jet0 is e-side and jet1 is mu-side, `1`: jet1 is e-side and jet0 is mu-side, `-1`: unavailable |
| `targets/reco_quality` | `(N,)` | binary reconstructibility label: `1`: fully matched/reconstructible, `0`: partial or not fully matched |
| `targets/nu` | `(N, 2, 3)` | neutrino then antineutrino `px, py, pz` |
| `targets/top` | `(N, 2, 4)` | top then antitop `px, py, pz, E` |

For prod_v2 files that store gen particles as `pt, eta, phi, mass`, the converter derives these Cartesian targets during HDF5 production.

### `masks`

| Dataset | Shape | Description |
| --- | --- | --- |
| `masks/pair` | `(N,)` | valid pairing label |
| `masks/reco` | `(N,)` | valid reconstructibility classification label |
| `masks/nu` | `(N, 2)` | valid neutrino and antineutrino regression targets |
| `masks/top` | `(N, 2)` | valid top and antitop regression targets |

### `weights`

| Dataset | Shape | Description |
| --- | --- | --- |
| `weights/event` | `(N,)` | event weight from the ntuple, or `1.0` if no weight branch is found |

### `metadata`

| Dataset | Shape | Description |
| --- | --- | --- |
| `metadata/run` | `(N,)` | run number, or zero if unavailable |
| `metadata/luminosityBlock` | `(N,)` | lumi block, or zero if unavailable |
| `metadata/event` | `(N,)` | event id, or source-order index if unavailable |
| `metadata/split` | `(N,)` | deterministic split code: `0=train`, `1=val`, `2=test` |
| `metadata/truth_available` | `(N,)` | source gen-truth availability flag, or true if the source has no such branch |

File attributes include feature names, target names, split fractions, source files, tree name, and branch aliases used by the converter.
