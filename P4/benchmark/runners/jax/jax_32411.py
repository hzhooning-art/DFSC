"""Role-independent reproducer for JAX issue #32411."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
from jax._src.api import vjp3


NUM_LAYERS = 2
EMBEDDING_SIZE = 32
NUM_MICROBATCHES = 3
MICROBATCH_SIZE = 4


def microbatch_loss(weights, inputs, targets):
    activations, _ = jax.lax.scan(
        lambda value, weight: (jnp.tanh(jnp.dot(value, weight)), None),
        inputs,
        weights,
    )
    predictions = jnp.mean(activations, axis=-1)
    return jnp.mean((targets - predictions) ** 2)


def ref_accumulated_gradient(weights, inputs, targets):
    accumulator = jax.new_ref(jnp.zeros_like(weights))

    def body(_, inputs_targets):
        local_inputs, local_targets = inputs_targets
        value, pullback = vjp3(
            lambda candidate: microbatch_loss(candidate, local_inputs, local_targets),
            weights,
        )
        pullback.with_refs(accumulator)(jnp.ones_like(value))
        return (), value

    shaped_inputs = inputs.reshape(NUM_MICROBATCHES, MICROBATCH_SIZE, EMBEDDING_SIZE)
    shaped_targets = targets.reshape(NUM_MICROBATCHES, MICROBATCH_SIZE)
    _, values = jax.lax.scan(body, (), (shaped_inputs, shaped_targets))
    return jax.ref.freeze(accumulator) / NUM_MICROBATCHES, values


def value_accumulated_gradient(weights, inputs, targets):
    def body(accumulator, inputs_targets):
        local_inputs, local_targets = inputs_targets
        value, gradient = jax.value_and_grad(microbatch_loss)(
            weights, local_inputs, local_targets
        )
        return accumulator + gradient, value

    shaped_inputs = inputs.reshape(NUM_MICROBATCHES, MICROBATCH_SIZE, EMBEDDING_SIZE)
    shaped_targets = targets.reshape(NUM_MICROBATCHES, MICROBATCH_SIZE)
    gradient, values = jax.lax.scan(
        body, jnp.zeros_like(weights), (shaped_inputs, shaped_targets)
    )
    return gradient / NUM_MICROBATCHES, values


def main():
    key = jax.random.PRNGKey(0)
    weights = jax.random.normal(
        key, (NUM_LAYERS, EMBEDDING_SIZE, EMBEDDING_SIZE)
    ) / jnp.sqrt(EMBEDDING_SIZE)
    inputs = jax.random.normal(
        key, (NUM_MICROBATCHES * MICROBATCH_SIZE, EMBEDDING_SIZE)
    )
    targets = jnp.full((NUM_MICROBATCHES * MICROBATCH_SIZE,), 0.5)

    ref_gradient, ref_values = jax.jit(ref_accumulated_gradient)(
        weights, inputs, targets
    )
    value_gradient, value_values = jax.jit(value_accumulated_gradient)(
        weights, inputs, targets
    )
    gradient_error = float(jnp.max(jnp.abs(ref_gradient - value_gradient)))
    value_error = float(jnp.max(jnp.abs(ref_values - value_values)))
    passed = gradient_error < 1e-6 and value_error < 1e-6
    print(json.dumps({
        "schema": "P4-Role-Independent-Runner-v1",
        "case_id": "jax_32411",
        "package": "jax",
        "version": jax.__version__,
        "outcome": "pass" if passed else "fail",
        "invocations": 2,
        "details": {
            "max_absolute_gradient_error": gradient_error,
            "max_absolute_value_error": value_error,
            "reference_method": "value-and-grad-scan-accumulation",
            "shape": [NUM_LAYERS, EMBEDDING_SIZE, EMBEDDING_SIZE],
            "microbatches": NUM_MICROBATCHES,
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
