from pathlib import Path

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Bread slice tester")
    subparsers = parser.add_subparsers(dest="command", required=True)
    test_parser = subparsers.add_parser("test", help="Run tests on a specific path")
    test_parser.add_argument("path", type=Path, help="Path to the test file or directory")

    args = parser.parse_args()
    if args.command == "build":
        print("Build!!")
        
    elif args.command == "test":
        print("test: ", args.path)
    
