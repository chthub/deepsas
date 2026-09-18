"""Phenotype-aware entry point for the current DeepSAS training pipeline.

This wrapper deliberately contains no copied model or refinement logic. It
enables phenotype-aware SnC selection and then executes the canonical
``deepsas_v1.py`` entry point, so model, convergence, logging, and checkpoint
behavior stay synchronized with the main pipeline.
"""

from pathlib import Path
import runpy
import sys


def main():
    project_dir = Path(__file__).resolve().parents[1]
    project_path = str(project_dir)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    if '--phenotype_aware' not in sys.argv:
        sys.argv.append('--phenotype_aware')
    runpy.run_path(str(project_dir / 'deepsas_v1.py'), run_name='__main__')


if __name__ == '__main__':
    main()
