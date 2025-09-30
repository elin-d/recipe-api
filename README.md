# Recipe-API

**Recipe-API** is a Django REST Framework application that provides a simple, robust API for managing recipes, ingredients, categories, and user interactions.

## Features

- CRUD operations for recipes, ingredients, and categories
- User authentication and permissions (JWT or Session Auth)
- Filtering, pagination, and search support

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/recipe-api.git
cd recipe-api

# Install dependencies and activate environment
poetry install
poetry shell
```

## Quickstart

```bash
# Apply database migrations
poetry run python manage.py migrate

# Create a superuser for the admin interface
poetry run python manage.py createsuperuser

# Run the development server
poetry run python manage.py runserver
```

Open your web browser and navigate to:

- `http://localhost:8000/api/recipes` to list all recipes
- `http://localhost:8000/api/recipes` to create a new recipe
- `http://localhost:8000/api/recipes/1` to retrieve a single recipe

## Development

We use Poetry for dependency management and virtual environments.

1. **Install dependencies** (including dev dependencies):
   ```bash
   poetry install
   poetry shell
   ```
2. **Format & Lint**:
   ```bash
   poetry run black . && poetry run isort . && poetry run flake8 .
   ```
3. **Run tests**:
   ```bash
   poetry run pytest
   ```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/the-nulldev/recipes-api/blob/main/CONTRIBUTING.md) for detailed guidelines on issues, pull requests, coding style, and testing.

## License

This project is licensed under the MIT License. See [LICENSE](https://github.com/the-nulldev/recipes-api/blob/main/LICENSE) for details.

