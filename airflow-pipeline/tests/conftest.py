import sys
import pathlib
# Adiciona raiz do projeto (airflow-pipeline) ao PYTHONPATH para imports de scripts/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))