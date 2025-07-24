# Photo Metadata AI - Agent Guidelines

## Build/Test Commands
- **Install dependencies**: `pip install -e .` or `pip install -r requirements.txt`
- **Run scripts**: Use project scripts defined in pyproject.toml (e.g., `photo-categorizer`, `ai-analyzer`)
- **Python version**: Requires Python >= 3.11
- **Run tests**: `uv run pytest` or `uv run pytest tests/test_photo_data_processor.py::TestUtilityFunctions::test_is_image_file`
- **Run with coverage**: `uv run pytest --cov=src --cov-report=term-missing`

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local imports. Use absolute imports for src modules
- **Naming**: Classes use PascalCase (e.g., `RekognitionCategorizer`), functions use snake_case
- **Docstrings**: Use triple quotes for module and class docstrings
- **Error handling**: Always catch specific exceptions (e.g., `ClientError`, `NoCredentialsError`)
- **Logging**: Use logging module, create logs/ directory, log to files not console
- **Environment**: Load .env files from multiple paths: `.env`, `/content/.env`, `~/.env`
- **AWS Integration**: Use boto3 with proper error handling for S3 and Rekognition
- **Threading**: Use ThreadPoolExecutor for concurrent operations
- **File organization**: All source code in src/ directory with __init__.py
- **Dependencies**: Core deps include boto3, openai, pandas, opencv-python, Pillow
- **No type hints**: Code doesn't use type annotations
- **Configuration**: Environment variables for AWS credentials and API keys