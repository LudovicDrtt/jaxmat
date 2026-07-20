# Contributing to `jaxmat`

Thanks for your interest in `jaxmat`! Contributions of all kinds are welcome:
bug reports, questions, documentation fixes, new demos, and new material models.
This guide explains how to set up a development environment and the conventions
we follow.

If anything here is unclear, please [open an
issue](https://github.com/erc-automatix/jaxmat/issues) — improving these
instructions is itself a valuable contribution.

## Ways to contribute

- **Report a bug** or unexpected numerical behavior via the bug report template.
- **Request a feature** (a new yield surface, hardening law, flow rule, potential,
  neural component, solver, …) via the feature request template.
- **Ask a question** about usage or modeling. Questions often reveal gaps in the
  documentation and are genuinely useful to us.
- **Improve the documentation or demos**. Even small clarifications help.
- **Contribute code**: bug fixes, tests, or new modeling components.

## Development setup

`jaxmat` targets Python 3.11 and 3.12. We recommend a fresh virtual environment.

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/jaxmat.git
cd jaxmat

# Editable install with the test dependencies
pip install -e ".[test]"

# Install the pre-commit hooks (linting + formatting run on every commit)
pre-commit install
```

To build the documentation locally, install the docs extra instead (or in
addition):

```bash
pip install -e ".[docs]"
```

## Code style

We use [`ruff`](https://docs.astral.sh/ruff/) for both linting and formatting,
enforced through `pre-commit` (line length 100). The same hooks run in CI, so a
formatting slip will fail the pipeline. To check and fix everything before
committing:

```bash
pre-commit run --all-files
```

## Tests

We use `pytest`. Please run the suite before opening a pull request:

```bash
pytest tests/ -q
```

CI runs the tests on Python 3.11 and 3.12 against two dependency legs: the
lower *pinned* bounds and the *latest* release of the JAX solver stack
(`optimistix`, `lineax`, `equinox`, `diffrax`, `paramax`). This guards against
dependency-drift regressions, so if your change touches solver behavior, it is
worth testing against an up-to-date stack locally too:

```bash
pip install -U optimistix lineax equinox diffrax paramax
pytest tests/ -q
```

**New behaviors and tensor utilities should come with tests.**
Ideally, good things to check for a new material model include: correctness against
an analytical or reference solution, that `jax.jit` and  `jax.vmap` work on the
`constitutive_update`, and that constitutive tangents and gradients w.r.t. parameters are
finite and correct.

## Adding a material model or component

`jaxmat` is built on a few consistent patterns; following them keeps new
components composable:

- Models are [`equinox`](https://docs.kidger.site/equinox/) modules
  (pytrees), so they are automatically `jit`/`vmap`-compatible and
  differentiable. Store parameters as fields.
- Small- and finite-strain behaviors implement a `constitutive_update` returning
  the updated stress/state. Variational models can be easily expressed through the
  generalized standard material (GSM) framework (a free energy + a dissipation
  potential) so that stresses and tangents come from automatic differentiation
  rather than hand-derived expressions.
- Keep components **independently swappable**: a yield surface, hardening law,
  or flow rule should not hard-code assumptions about the others. This
  modularity, including the ability to replace a component with a neural network,
  is a core design goal of the library.
- Use constrained/latent parameters (`Positive`, `Bounded`, `Scaled`, …) for
  quantities with feasibility constraints. This insures more robust calibration,
  avoiding evaluating the model with non-admissible parameters.

If you are planning a larger contribution, please open an issue first so we can
discuss the design — this avoids duplicated effort.

## Documentation and demos

Documentation is built with [Jupyter Book](https://jupyterbook.org/) and demos
are written as [`jupytext`](https://jupytext.readthedocs.io/) `.py` files in the
*percent* format (paired with MyST Markdown). Demos are expected to run and
to track the current API. A demo that no longer runs is treated as a bug.

When you add or change public API, please update the relevant demo or docs page
so the examples stay executable.

## Submitting a pull request

1. Create a branch from `main` for your change.
2. Make sure `pre-commit run --all-files` and `pytest tests/ -q` pass.
3. Add or update tests and documentation as appropriate.
4. Add a short entry to `CHANGELOG.md` under the *Unreleased* section.
5. Open the pull request against `main` with a clear description of the change
   and the motivation. Reference any related issue.

CI must pass before a review can be completed. Maintainers may request changes —
this is a normal part of the process and not a reflection on the contribution.

## Reporting bugs and asking questions

Please use the issue templates. For bugs, a **minimal reproducible example** (a
short self-contained script and the full traceback, plus your `jax`,
`equinox`, and `optimistix` versions) makes it dramatically easier to help.

## Use of GenAI tools

Use of GenAI or agents for generating code or documentation is permitted. However,
code must follow the library coding style and conventions. Documentation or demos
should follow the written style of the documentation. Contributors should **systematically**
review the produced code and are responsible for the generated content.

## License

`jaxmat` is distributed under the **LGPL-3.0** license. By contributing, you
agree that your contributions will be licensed under the same terms.

## Acknowledgements

`jaxmat` is developed as part of the ERC project AUTOMATIX (CoG project no.
101229452). We are grateful to everyone who reports issues, asks questions, and
contributes code.
