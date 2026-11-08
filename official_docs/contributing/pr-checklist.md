# Pull Request (PR) Checklist & Workflow

Before submitting a Pull Request, verify that your branch satisfies all requirements in this checklist.

---

## 1. Development Workflow

```bash
# 1. Branch from main
git checkout -b feature/my-new-feature

# 2. Implement your code and unit tests
# ...

# 3. Format and lint
ruff check --fix .
ruff format .

# 4. Run the unit test suite
pytest tests/ -v

# 5. Commit with descriptive semantic message
git commit -m "feat(losses): add AlphaDivergence criterion"

# 6. Push to your fork and create Pull Request
git push origin feature/my-new-feature
```

---

## 2. The Contributor PR Checklist

Ensure each item is checked in your PR description:

- [ ] **Tests Added:** Unit tests added in `tests/` covering both forward loss computation and backward gradient flow without `NaN`s.
- [ ] **Schema & Validation:** Running `lmkd check -c <recipe.yaml>` passes with zero schema errors or warnings.
- [ ] **No Regression:** All existing test suites pass (`pytest tests/`).
- [ ] **Code Quality:** Code adheres to 120-char line limit and passes `ruff check .` with zero errors.
- [ ] **Documentation:** If adding a baseline or plugin:
  - Added documentation page in `methods/` or `guides/`.
  - Added entry in the feature matrix in `official_docs/index.md`.
  - Linked the originating research paper in the docstring and `@REGISTRY.register`.

