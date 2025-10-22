#!/usr/bin/env python
"""Thin wrapper delegating to src.generate_synthetic_sentences.main

This keeps existing CLI usage the same while the implementation lives under src/.
"""
from src.generate_synthetic_sentences import main


if __name__ == '__main__':
    main()
    # delegate to src.generate_synthetic_sentences
    from src.generate_synthetic_sentences import main as _main
    _main()
