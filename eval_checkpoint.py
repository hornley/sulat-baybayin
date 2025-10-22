"""
Top-level classification inference shim. Delegates to src.classification.eval_checkpoint.main().
"""

from src.classification.eval_checkpoint import main


if __name__ == '__main__':
    main()
