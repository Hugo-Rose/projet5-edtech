"""Configuration pytest — pas de DB requise pour les tests unitaires."""
import sys
from pathlib import Path

# Assure que src/ est dans le PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))
