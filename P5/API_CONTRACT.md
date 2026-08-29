# P5 public API contract

Version: `0.1.0`

The supported public surface is intentionally small:

- `fit(curves, rank, ...)`: fit positive shared decay rates with curve-specific
  signed amplitudes and intercepts;
- `evaluate(curves, ranks, ...)`: perform leave-one-group-out value and
  late-window prediction audits;
- `decide(evaluation, gates)`: apply frozen nested-rank gates and return either
  `SUPPORTED_RANK_k` or `INDETERMINATE`;
- `report(payload, output)`: write a deterministic JSON audit record.

`CurveRecord` declares the independent unit, grouping variable, channel,
registered time grid, and values. `GateConfig` contains the four public
decision thresholds. Result records use schema version `1.0.0`, defined in
`schemas/memory_protocol_result.schema.json`.

## Reproduction command

From `P5`:

```powershell
..\P1\.venv\Scripts\python.exe -m p5_memory_protocol reproduce --task uci-gas
```

The available task names are `public`, `pva`, and `uci-gas`. The command exits
nonzero if any delegated experiment fails.

## Supported scope

The current backend assumes a common registered time grid and a finite sum of
positive real exponential rates. Signed curve-specific amplitudes are allowed.
Irregular grids, complex poles, distributed spectra, correlated-noise
likelihoods, and universal hypothesis-test calibration are not part of version
`0.1.0`. A successful numerical fit outside this contract is not a supported
mechanism claim.

Backward-incompatible changes to these four functions, the tri-state decision
semantics, or schema `1.0.0` require a new major API or schema version.
