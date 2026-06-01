# driven_qubits

Code accompanying *Fast, Accurate, and Local Temperature Control Using Qubits*
(Baruah, Portugal, Wabnig, Flindt — [arXiv:2410.04796](https://arxiv.org/abs/2410.04796)).

A target system qubit is coupled to a chain of ancilla ("control") qubits
that exchange heat with a bosonic bath at fixed temperature `Te`. By
modulating the ancilla energy splittings `ωⱼ(t)` in time, the joint system
relaxes the target qubit toward a chosen effective temperature `T(t)`. The
scripts here numerically solve the inverse problem — given a desired
`T(t)`, integrate for the required `ωⱼ(t)` — and then evolve the joint
density matrix under the resulting drive to verify the protocol.

## Layout

### Libraries

- [manyq_sparse.py](manyq_sparse.py) — sparse Liouvillian implementation
  for the joint system-plus-ancillas density matrix. Exposes:
  - `Lind_action` — the main model, where only the ancillas couple to the
    bath and the system qubit equilibrates indirectly via energy transfer with
    the driven ancillas.
  - `Lind_action_with_sys_bath` — referee-response variant in which the
    system qubit also couples directly to the bath with rate `g_sys`. See
    the comment block at the top of the function for the precise
    differences from `Lind_action`.
- [manyqfc.py](manyqfc.py) — dense helpers: target temperature profiles
  `effTemp`, the inverse-problem ODE `omegasine` (right-hand side fed to
  `scipy.integrate.odeint` to recover `ωⱼ(t)`), initial-state construction
  `initalize`, and the dense Liouvillian primitives.

### Driver scripts

Each driver builds the operators, integrates the inverse ODE for
`ωⱼ(t)`, evolves the joint sparse density matrix with a 5th-order
expansion of `exp(L·dt)·ρ`, and saves the time axis, ancilla frequencies,
target `T(t)`, recovered system-qubit temperature, and bath temperature
into a paired `.npy` files under `data/`.

- [constant_effective_temperature.py](constant_effective_temperature.py)
  — constant target `T(t) = T₀`. Writes `data/frequencies_fig_1.npy`,
  `data/temperatures_fig_1.npy`.
- [constant_temperature_review.py](constant_temperature_review.py) — same
  setup but uses `Lind_action_with_sys_bath` (referee-response figure
  where the target qubit is not perfectly isolated from the bath).
- [cosine_temperature.py](cosine_temperature.py) — cosine target
  `T(t) = a · cos(Ωt) + b` for `n ∈ {3, 5, 8}` ancillas (set at the top of
  the script). Writes `data/frequencies_cos{n}.npy`,
  `data/temperatures_cos{n}.npy`.
- [temperature_pulse.py](temperature_pulse.py) — gaussian-pulse target
  `T(t)`. Writes `data/frequencies_pulse_pt{x}.npy`,
  `data/temperatures_pulse_pt{x}.npy`.

### Notebooks

- [plots.ipynb](plots.ipynb) — loads the `.npy` files from `data/` and
  produces the published figures: `cosine_n_alt_f.svg` (cosine drive vs
  ancilla count), `constant_n_alt.svg` (constant drive vs ancilla count),
  `fig6_2.svg` (pulse drive), `constant_fig_1.svg` (single-ancilla
  reference for the constant case).
- [1qubit.ipynb](1qubit.ipynb) — single-qubit reference calculations
  (sine, sawtooth, and square temperature profiles in both the forward and
  inverse directions) producing `1qubit_grndstate.svg` and
  `1qubit_inverse_grndstate.svg`.
