"""Keep ``python -m unittest tests`` as the complete repository test command."""
from pathlib import Path


def load_tests(loader, tests, pattern):
    return loader.discover(str(Path(__file__).parent), pattern or 'test_*.py')
