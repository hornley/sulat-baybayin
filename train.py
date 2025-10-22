"""
Top-level compatibility wrapper for the project. Delegate CLI handling to src.train.main()
so that new options added in src/train.py are always available when running `python train.py`.
"""

from src.train import main


if __name__ == '__main__':
    main()
