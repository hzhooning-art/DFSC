# UCI gas-sensor flow-modulation data

- Source: UCI Machine Learning Repository dataset 308, *Gas sensor array under flow modulation*.
- DOI: `10.24432/C5BG7G`.
- License: CC BY 4.0.
- Official archive: `https://archive.ics.uci.edu/static/public/308/gas%2Bsensor%2Barray%2Bunder%2Bflow%2Bmodulation.zip`.
- Archive MD5: `b475320820f15ebfcbfe89c20138acb5`.
- `rawdata.csv.gz` SHA-256: `ed2aae124aa733fd475af0ede16431252103e15644fb43d519bb322800c3925d`.
- `features.csv` SHA-256: `2bb992aded17f698e84dc34c86dfdb17e7bdbae64ae7ef60d8c3eb47e494f7a7`.

The frozen P5 task uses the registered room-air recovery interval from 180 to
300 seconds. Samples are aggregated into the ten ventilator cycles (12 seconds
per cycle); this is protocol-aware aggregation, not a fitted smoother.
