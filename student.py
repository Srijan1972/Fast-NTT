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

q_inv = 0

@jax.jit
def mod_add(a, b, q):
    """(a + b) mod q, elementwise."""
    # res = a + b
    # needs_sub = (res < a) | (res >= q)
    # return res - jnp.where(needs_sub, q, jnp.uint32(0))
    # res = a + b
    # return res - (res >= q) * q
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    res = a64 + b64
    return jnp.where(res >= q, res - q, res).astype(jnp.uint32)

@jax.jit
def mod_sub(a, b, q):
    """(a - b) mod q, elementwise."""
    # needs_wrap = a < b
    # res = a - b
    # return res + jnp.where(needs_wrap, q, 0)
    # res = a - b
    # return res + (res < 0) * q
    a64 = a.astype(jnp.int64)
    b64 = b.astype(jnp.int64)
    res = a64 - b64
    return jnp.where(res < 0, res + q, res).astype(jnp.uint32)

@jax.jit
def mod_mul(a, b, q):
    """(a * b) mod q, elementwise."""
    # return (a * b) % q
    # a64 = a.astype(jnp.uint64)
    # b64 = b.astype(jnp.uint64)
    # return ((a64 * b64) % q).astype(jnp.uint32)
    a64 = a.astype(jnp.uint64)
    b64 = b.astype(jnp.uint64)
    ab = a64 * b64
    ab_low = ab.astype(jnp.uint32)
    m = (ab_low * q_inv).astype(jnp.uint32)
    mq = m.astype(jnp.uint64) * jnp.uint64(q)
    t = ((ab + mq) >> 32).astype(jnp.uint32)
    return jnp.where(t >= q, t - q, t)



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
    x = x[:, psi_powers]
    batch, N = x.shape
    length = 1
    offset = 0
    while length < N:
        twiddle = twiddles[offset : offset + length]
        num_blocks = N // (2 * length)
        x = x.reshape((batch, num_blocks, 2 * length))
        U = x[:, :, :length]
        V = x[:, :, length:]
        T = mod_mul(V, twiddle, q)
        U1 = mod_add(U, T, q)
        V1 = mod_sub(U, T, q)
        x = jnp.concatenate([U1, V1], axis=2)
        x = x.reshape((batch, N))
        offset += length
        length *= 2
    return x


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
        psi_powers[i] = psi_powers[i - 1] * psi % q
    twiddles = np.zeros(N - 1, dtype=np.uint32)
    length = 1
    offset = 0
    while length < N:
        j = np.arange(length)
        power = (2 * j + 1) * (N // (2 * length))
        twiddles[offset : offset + length] = psi_powers[power].astype(np.uint32)
        offset += length
        length *= 2
    q_inv = (-pow(int(q), -1, 1 << 32)) % (1 << 32)
    R = 1 << 32
    Rq = R % q
    twiddles = (twiddles.astype(np.uint64) * Rq) % q
    return jnp.array(rev_idx, dtype=jnp.uint32), jnp.array(twiddles, dtype=jnp.uint32)
