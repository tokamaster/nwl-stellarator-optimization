# Bosch-Hale DESC/MC scratch comparison

All powers and surface areas are for the 90-degree comparison domain.
MC values come from the saved batch aggregates.

## Global quantities

| Case | Method | Min | Max | Pn | S_A | Pn/S_A | Peak | DESC-MC Pn % | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Circular axisymmetric | MC | 2.33 | 3.14 | 84.27 | 30.26 | 2.78 | 1.13 |  | ok |
| Circular axisymmetric | DESC | 2.46 | 3.30 | 85.18 | 30.25 | 2.82 | 1.17 | 1.08 | ok |
| ITER-like axisymmetric | MC | 2.12 | 6.86 | 366.73 | 78.35 | 4.68 | 1.47 |  | ok |
| ITER-like axisymmetric | DESC | 2.57 | 7.14 | 371.27 | 78.35 | 4.74 | 1.51 | 1.24 | ok |
| Non-axisymmetric | MC | 0.77 | 5.41 | 889.09 | 280.16 | 3.17 | 1.71 |  | ok |
| Non-axisymmetric | DESC | 0.80 | 6.23 | 902.29 | 280.19 | 3.22 | 1.93 | 1.49 | ok |

## MC uncertainty

| Case | min rel. err. % | area-mean rel. err. % | area-median rel. err. % | area-p95 rel. err. % | max rel. err. % | high-NWL area-mean rel. err. % |
|---|---:|---:|---:|---:|---:|---:|
| Circular axisymmetric | 0.65 | 1.01 | 0.96 | 1.34 | 2.09 | 0.86 |
| ITER-like axisymmetric | 0.54 | 1.06 | 0.97 | 1.82 | 3.15 | 0.72 |
| Non-axisymmetric | 0.47 | 1.08 | 0.94 | 1.86 | 3.06 | 0.65 |

## Interpretation

- All three MC aggregates are now consistent with the regenerated DESC Bosch-Hale power scale.
- DESC remains high relative to MC in integrated neutron power by about 1.1-1.5% across the three cases.
