# Application Domains

`dfsc` is strongest when a fractional model has a known linear propagator or a
discrete operator whose Mittag-Leffler action can be evaluated directly. The
application layer translates four recurring model classes into the same
problem--algorithm--solution interface.

| Domain | dfsc path | Specific advantage | Current boundary |
|---|---|---|---|
| Anomalous diffusion | Regular spectral MLSL | Batched arbitrary-time queries and trainable temporal/spatial orders without storing history | 1D/2D tensor-product domains and constant orders |
| Linear viscoelastic or diffusion relaxation | Generalized stiffness/mass MLSL | Mass-aware projection from an existing finite-element discretization | Symmetric linear systems; matrices are supplied by the user |
| Network memory diffusion | Graph MLSL | A known graph propagator can be inserted directly into a neural computation graph | Undirected dense adjacency in the direct adapter |
| Fractional advection--diffusion | General-operator Arnoldi | Non-self-adjoint matrix-function actions without eigenvector decomposition | Periodic centered discretization and reduced argument radius at most four |

## Unified Application Object

Each constructor returns an `ApplicationCase`. It contains the configured
problem, recommended algorithm, differentiable parameters, assumptions, and
limitations. Calling `case.solve()` uses the recommended method; passing an
algorithm explicitly remains possible.

```python
case = dfsc.anomalous_diffusion_case(
    initial=lambda x: torch.sin(torch.pi * x),
    times=torch.linspace(0.0, 0.1, 8),
    alpha=torch.tensor(0.8, requires_grad=True),
    beta=torch.tensor(1.7, requires_grad=True),
    diffusivity=0.1,
)
solution = case.solve()
print(case.summary())
```

The catalogue is machine readable:

```python
for profile in dfsc.application_catalog():
    print(profile["name"], profile["fit"], profile["limitations"])
```

## Interpretation of Coverage

The four templates expand application usability around the MLSL core; they do
not claim that dfsc is a complete domain simulator. Mesh generation,
constitutive model selection, experimental-data calibration, nonlinear fluxes,
and arbitrary memory kernels remain outside these templates. They can be
coupled through supplied operators, forcing, semilinear, or neural-residual
interfaces when the corresponding mathematical assumptions are justified.
