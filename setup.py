from setuptools import setup, find_packages

setup(
    name="player-performance-forecaster",
    version="0.1.0",
    author="Edgar Parra",
    description="NBA player performance forecasting with statistical modeling",
    url="https://github.com/EdgarParra565/player-performance-forecaster",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "nba-api",
        "matplotlib",
        "seaborn",
        "requests",
        "python-dotenv",
    ],
)
