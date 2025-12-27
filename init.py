# Main execution file for Heart Disease Analysis
# This file imports and runs functions from part_one.py

from part_one import (
    run_part_one
)

# =============================================================================
# PART 1 - DATA EXPLORATION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("        HEART DISEASE ANALYSIS PROJECT")
    print("=" * 60)
    print("\n>>> Starting Part 1: Data Exploration...\n")
    
    # Option 1: Run all Part 1 tasks at once
    df_cleaned = run_part_one('heart.csv')
    
