# Contributing to Model Audit Copilot

Thank you for your interest in contributing to Model Audit Copilot! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [issue tracker](https://github.com/yourusername/model-audit-copilot/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)

### Suggesting Features

1. Check the [roadmap](README.md#roadmap) and existing issues
2. Open a feature request issue with:
   - Use case description
   - Proposed solution
   - Alternative solutions considered
   - Potential impact

### Contributing Code

#### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/model-audit-copilot.git
cd model-audit-copilot

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

#### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Add tests for new functionality

4. Run tests locally:
   ```bash
   # Run all tests
   pytest tests/
   
   # Run with coverage
   pytest tests/ --cov=copilot --cov-report=html
   
   # Run specific test
   pytest tests/test_drift_detector.py -v
   ```

5. Run code quality checks:
   ```bash
   # Format code
   black copilot/ scripts/ tests/
   
   # Sort imports
   isort copilot/ scripts/ tests/
   
   # Lint
   flake8 copilot/ scripts/ tests/
   
   # Type checking
   mypy copilot/
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

7. Push and create a pull request

#### Commit Message Convention

We follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or modifications
- `chore:` Maintenance tasks

Examples:
```
feat: add support for categorical drift detection
fix: handle missing values in fairness audit
docs: update installation instructions
test: add edge cases for outlier detection
```

### Coding Standards

#### Python Style Guide

- Follow PEP 8
- Use type hints for function arguments and returns
- Maximum line length: 100 characters
- Use descriptive variable names

#### Code Organization

```python
"""Module docstring describing purpose."""

# Standard library imports
import os
from typing import Dict, List, Optional

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from copilot.config import get_config

# Module-level logger
logger = logging.getLogger(__name__)


class MyClass:
    """Class docstring with description.
    
    Attributes:
        attr1: Description of attribute 1
        attr2: Description of attribute 2
    """
    
    def __init__(self, param1: str, param2: Optional[int] = None):
        """Initialize MyClass.
        
        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2
            
        Raises:
            ValueError: If param1 is invalid
        """
        self.attr1 = param1
        self.attr2 = param2 or 42
```

#### Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include type hints
- Add usage examples for complex functions

#### Testing

- Write tests for all new functionality
- Maintain > 80% code coverage
- Use meaningful test names
- Include edge cases and error conditions
- Use fixtures for shared test data

### Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add entry to CHANGELOG.md if applicable
4. Ensure PR description clearly describes changes
5. Link related issues
6. Request review from maintainers
7. Address review feedback
8. Squash commits if requested

### Release Process

Maintainers handle releases:

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create release tag: `git tag v1.2.3`
4. Push tag: `git push origin v1.2.3`
5. GitHub Actions will handle PyPI deployment

## Development Tips

### Running the Dashboard Locally

```bash
streamlit run dashboard/main_app.py
```

### Debugging

1. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Use debugger:
   ```python
   import pdb; pdb.set_trace()
   ```

### Performance Testing

```python
# Profile code
import cProfile
cProfile.run('your_function()')

# Time execution
import timeit
timeit.timeit('your_function()', number=1000)
```

## Questions?

- Check the [FAQ](docs/FAQ.md)
- Ask in [discussions](https://github.com/yourusername/model-audit-copilot/discussions)
- Contact maintainers

Thank you for contributing to Model Audit Copilot! 🎉