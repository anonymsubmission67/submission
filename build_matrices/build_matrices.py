#!/usr/bin/env python3
"""
Main script to run all build_matrices scripts.

This script can run all matrix building scripts or individual ones.
"""

import argparse
import sys
from pathlib import Path

# Import all processing functions
from api_format.no_persona import process_no_persona_api
from api_format.no_persona_compass import process_no_persona_compass_api
from api_format.personas import process_personas_api
from api_format.personas_compass import process_personas_compass_api


def main():
    parser = argparse.ArgumentParser(
        description="Build matrices from API output data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scripts
  python build_matrices.py --all
  
  # Run specific script
  python build_matrices.py --script no_persona_api
  
  # Run with specific prompt styles
  python build_matrices.py --prompt-styles simple chain_of_thought
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scripts (default if no other option is specified)"
    )
    
    parser.add_argument(
        "--script",
        choices=[
            "no_persona_api",
            "no_persona_compass_api",
            "personas_api",
            "personas_compass_api",
        ],
        help="Run a specific script"
    )
    
    
    parser.add_argument(
        "--prompt-styles",
        nargs="+",
        choices=["simple", "chain_of_thought"],
        help="Prompt styles to process (default: all available)"
    )
    
    args = parser.parse_args()
    
    # Determine what to run
    run_all = args.all or (not args.script)
    
    if args.script:
        scripts_to_run = [args.script]
    else:
        scripts_to_run = [
            "no_persona_api",
            "no_persona_compass_api",
            "personas_api",
            "personas_compass_api",
        ]
    
    # Map script names to functions
    script_map = {
        "no_persona_api": lambda: process_no_persona_api(prompt_styles=args.prompt_styles),
        "no_persona_compass_api": lambda: process_no_persona_compass_api(prompt_styles=args.prompt_styles),
        "personas_api": lambda: process_personas_api(prompt_styles=args.prompt_styles),
        "personas_compass_api": lambda: process_personas_compass_api(prompt_styles=args.prompt_styles),
    }
    
    # Run scripts
    for script_name in scripts_to_run:
        print(f"\n{'='*80}")
        print(f"Running: {script_name}")
        print(f"{'='*80}\n")
        
        try:
            script_map[script_name]()
            print(f"\n✓ Successfully completed: {script_name}\n")
        except Exception as e:
            print(f"\n✗ Error in {script_name}: {e}\n", file=sys.stderr)
            import traceback
            traceback.print_exc()
            if not run_all:
                # If running a single script, exit on error
                sys.exit(1)
    
    print(f"\n{'='*80}")
    print("All scripts completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

