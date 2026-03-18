"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes polynomial evaluation at odd powers of a primitive
root. Given coefficients x[0], x[1], ..., x[N-1], the output is:

    y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

where ψ is a primitive 2N-th root of unity (ψ^N ≡ -1 mod q).

This is equivalent to a cyclic NTT on "twisted" input, where each coefficient
x[n] is first multiplied by ψ^n.
"""

import jax
import jax.numpy as jnp
import provided
import numpy as np

# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

@jax.jit
def mod_add(a, b, q):
    """(a + b) mod q, elementwise."""
    # res = a + b
    # needs_sub = (res < a) | (res >= q)
    # return res - jnp.where(needs_sub, q, jnp.uint32(0))
    res = a + b
    return res - (res >= q) * q

@jax.jit
def mod_sub(a, b, q):
    """(a - b) mod q, elementwise."""
    # needs_wrap = a < b
    # res = a - b
    # return res + jnp.where(needs_wrap, q, 0)
    res = a - b
    return res + (res < 0) * q

@jax.jit(static_argnames=['q'])
def mod_mul(a, b, q):
    """(a * b) mod q, elementwise."""
    return (a * b) % q

# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------

@jax.jit
def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT.

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi_powers: Precomputed ψ^n table
        twiddles: Precomputed twiddle table

    Returns:
        jnp.ndarray: NTT output, same shape as input
    """
    A = x.astype(jnp.int64)
    A = A[:, psi_powers]
    batch, N = A.shape
    length = 1
    layer = 0
    while length < N:
        twiddle = twiddles[layer]
        num_blocks = N // (2 * length)
        A = A.reshape((batch, num_blocks, 2, length))
        U = A[:, :, 0, :]
        V = A[:, :, 1, :]
        T = mod_mul(V, twiddle, q)
        U1 = mod_add(U, T, q)
        V1 = mod_sub(U, T, q)
        A = jnp.stack([U1, V1], axis=2)
        A = A.reshape((batch, N))
        length *= 2
        layer += 1
    return A.astype(jnp.uint32)


def prepare_tables(*, q, psi_powers, twiddles):
    """
    Optional one-time table preparation.

    Override this if you want faster modular multiplication than JAX's "%".
    For example, you can convert the provided tables into Montgomery form
    (or any other domain) once here, then run `ntt` using your mod_mul.
    This function runs before timing, so its cost is not counted as latency.
    Must return (psi_powers, twiddles) in the form expected by `ntt`.
    """
    N = psi_powers.shape[0]
    log2N = int(np.log2(N))
    psi = provided.negacyclic_psi(N, q)
    rev_idx = np.zeros(N, dtype=int)
    for i in range(N):
        rev_idx[i] = int(format(i, f'0{log2N}b')[::-1], 2)
    psi_powers = np.ones(2 * N, dtype=np.int64)
    for i in range(1, 2 * N):
        psi_powers[i] = (psi_powers[i - 1] * psi) % q
    twiddles = []
    length = 1
    while length < N:
        j = np.arange(length)
        power = (2 * j + 1) * (N // (2 * length))
        twiddles.append(jnp.array(psi_powers[power], dtype=jnp.int64))
        length *= 2
    return jnp.array(rev_idx, dtype=jnp.uint32), twiddles
