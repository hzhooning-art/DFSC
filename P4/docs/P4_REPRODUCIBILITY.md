# P4 reproducibility entry point

The local evidence chain can be rerun from the repository root with:

```bash
python P4/experiments/run_protocol_audit.py
python -m unittest discover -s P4/tests -p "test_*.py" -v
```

The command executes the actual P1 `dfsc` MLSL implementation through its
public factory API, rebuilds the backend registry, runs the dependency-free
protocol smoke test, and writes
`P4/results/p4_reproducibility_manifest.json`.

The default environment must provide PyTorch and mpmath for the MLSL
validation. The protocol package itself remains dependency-free. The generated
manifest records the interpreter, platform, executed steps, backend count, and
the six MLSL gate states. It is a local reproducibility artifact, not an
external replication study or a claim of cross-machine performance equality.
