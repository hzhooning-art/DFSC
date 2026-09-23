# Frozen cross-design score stress protocol

Date: 2026-09-11. Frozen before outcomes.

Test whether phase/cone calibration and efficiency survive changes in channel
count, time count, rate, time geometry, signal energy, and variance-calibration
degrees of freedom. Use m=16384, alpha=.05, rate interval [.2,.8], p=.5,
B/S=.04 on the diversity branch, and target leading noncentrality 8.

Twelve designs are fixed: the C=6,n=24 baseline; C=2 and 12; n=12 and 48;
lambda=.25 and .65; one deterministically irregular 24-time design; S=100;
fixed scale nu=144; and two combined designs (C=4,n=16,lambda=.3,S=1000,
irregular,nu=144) and (C=8,n=36,lambda=.6,S=1000,irregular,native nu).
All remaining designs use S=10000, equally spaced times, lambda=.4 and native
nu=C*n*(m-1), except for the factor explicitly changed.

For each design simulate 1024 null datasets and 512 datasets on each of the
diversity, crossover, and proportional paths. Fit the one-rate null over the
same rate interval. Apply the omnibus union F test, phase score F test, and
cone score chi-bar-F test without empirical recalibration. Predict alternative
power using the exact population score projection and noncentral F law.

Predeclared gates:

1. phase and cone null rejection each lie in [.02,.08] for every design;
2. phase empirical power differs from its exact-projection prediction by at
   most .07 in every alternative cell;
3. phase score retains at least 85% of full profiled-distance noncentrality;
4. phase power exceeds omnibus power by at least .08 in every alternative cell;
5. cone power is no more than .05 below phase power in any cell;
6. all component amplitudes are positive and all statistics are finite.

Passing supports robustness within the declared compact regular regime. It
does not cover rates on the search boundary, correlated errors, estimated
prewhitening, or increasing C/n asymptotics.
