"""
Top-level compatibility wrapper for detection training.
Delegates to src.detection.train.main()
"""

from src.detection.train import main


if __name__ == '__main__':
    main()
