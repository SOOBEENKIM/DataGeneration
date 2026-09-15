"""Compatibility entry point for the fail-closed aggregate-only CLI.

Direct validation selection is intentionally unavailable. Both entry points
require the four immutable model-worker terminals and matching authorization.
"""

from scripts.aggregate_candidate_selection_v2_6 import main


if __name__ == "__main__":
    main()
