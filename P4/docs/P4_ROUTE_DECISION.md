# P4 Route Decision

## Decision

The independent P4 research line is frozen after the targeted external
validation failed its preregistered exit rule.

## Evidence

- Elastomer relaxation: stretched exponential was best on all four curves.
- Anomalous-diffusion trajectory diagnostic: nonnegative Prony was best on
  both H-actin and Brownian datasets.
- The P4 candidate family was 164x to 200x worse than the traditional best on
  the targeted trajectory observable.

## Retained component

The following parts remain useful and should be integrated as optional DFSC/P3
diagnostics:

- candidate mechanism comparison;
- bootstrap selection confidence;
- temporal validation agreement;
- risk--coverage curves;
- explicit abstention when mechanism evidence is weak.

These components must be presented as an audit and routing layer, not as a
universal fractional solver or as a superior replacement for stretched
exponential or Prony models.

## Reopening criterion

P4 should only be reopened if a new externally grounded observable and a new
candidate family satisfy both conditions:

1. outperform the strongest conventional baseline on at least one independent
   dataset;
2. remain within twice the strongest baseline error on the remaining dataset.
