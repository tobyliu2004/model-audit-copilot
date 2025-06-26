"""Setup configuration for Model Audit Copilot."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="model-audit-copilot",
    version="1.0.0",
    author="Toby Liu",
    author_email="",
    description="A comprehensive ML model auditing toolkit for detecting drift, fairness issues, and data quality problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tobyliu2004/model-audit-copilot",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audit-runner=scripts.audit_runner:main",
            "batch-audit=scripts.batch_runner:main",
            "audit-dashboard=scripts.launch_dashboard:main",
        ],
    },
    package_data={
        "model_audit_copilot": ["config/*.yaml", "templates/*.html"],
    },
    include_package_data=True,
    keywords="machine-learning ml-ops model-monitoring drift-detection fairness-ai data-quality",
    project_urls={
        "Bug Reports": "https://github.com/tobyliu2004/model-audit-copilot/issues",
        "Source": "https://github.com/tobyliu2004/model-audit-copilot",
        "Documentation": "https://model-audit-copilot.readthedocs.io",
    },
)