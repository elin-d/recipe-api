Thank you for your interest in contributing to **Recipe-API**! By participating in this project, you agree to abide by these guidelines.

---

## Issues

A well-crafted issue helps maintainers understand and reproduce your problem quickly. A good issue should include:

- A clear title and description of the bug or feature request.
- A short, self-contained, correct example demonstrating the problem (an SSCCE):
  ```python
  from rest_framework.test import APIClient

  client = APIClient()
  response = client.get('/api/recipes/999/')
  assert response.status_code == 404
  ```
- Relevant logs or tracebacks.
    - Enable Django debug logging by setting `LOGGING` level to `DEBUG` in your `settings.py`, then copy the relevant lines.
    - **Warning:** remove any sensitive information (API keys, passwords) before posting.

If you cannot provide all of the above, please open an issue anyway and a maintainer will follow up.

---

## Pull Requests

When submitting a pull request, please ensure it describes:

1. What problem you’re solving. Why is this change needed?
2. How you solved it. Include a summary of your approach and any trade-offs.

**Tips for PRs:**

- Break your changes into logical, atomic commits with descriptive messages.
- Update or add unit tests (see Automated Tests below) to cover new behavior or bug fixes.
- Where applicable, update Swagger/OpenAPI schema or API documentation.

---

## Coding Style

We follow the [Black](https://github.com/psf/black) coding style and [isort](https://pycqa.github.io/isort/) import sorting.

> Dev dependencies (Black, isort, flake8, pytest, etc.) are defined in `pyproject.toml` and installed via Poetry.

- Format code:
  ```bash
  poetry run black .
  poetry run isort .
  ```
- Lint with flake8:
  ```bash
  poetry run flake8 .
  ```

---

## Automated Tests

We use pytest and Django’s test framework. Dev dependencies are managed by Poetry.

1. **Install dependencies:**
   ```bash
   poetry install
   ```
2. **Run all tests:**
   ```bash
   poetry run pytest
   ```

If you add new functionality, please add corresponding tests and ensure coverage does not decrease.

---

## Django Migrations

If your change includes model updates:

1. Create a new migration:
   ```bash
   poetry run python manage.py makemigrations recipes
   ```
2. Apply migrations locally to verify:
   ```bash
   poetry run python manage.py migrate
   ```
3. Commit the generated migration file.

4. **Documentation updates**
    - Mark deprecated endpoints or fields in your API docs (e.g., OpenAPI/Swagger) using the `deprecated: true` flag.
    - Add a **migration guide** section describing alternative APIs or usage patterns.

---

Thanks for helping make Recipe-API better! We look forward to your contributions.

