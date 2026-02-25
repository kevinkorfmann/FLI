#!/usr/bin/env python3
"""Build the Likelihood-Based Inference book as a PDF via Sphinx + LaTeX."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
BUILD_LATEX = ROOT / "build" / "latex"


def run(cmd, cwd=None):
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Build the book PDF.")
    parser.add_argument("--clean", action="store_true", help="Remove build directory first")
    parser.add_argument("--latex-only", action="store_true", help="Generate LaTeX without compiling PDF")
    args = parser.parse_args()

    if args.clean and BUILD_LATEX.exists():
        print("Cleaning build directory...")
        shutil.rmtree(BUILD_LATEX)

    # Step 1: Sphinx LaTeX build
    print("Running Sphinx LaTeX builder...")
    BUILD_LATEX.mkdir(parents=True, exist_ok=True)
    run(["sphinx-build", "-b", "latex", str(DOCS), str(BUILD_LATEX)])

    if args.latex_only:
        print(f"LaTeX sources written to {BUILD_LATEX}")
        return

    # Step 2: Compile PDF
    tex_file = BUILD_LATEX / "LikelihoodInference.tex"
    if not tex_file.exists():
        print(f"Error: {tex_file} not found")
        sys.exit(1)

    compiler = shutil.which("latexmk")
    if compiler:
        print("Compiling PDF with latexmk...")
        run(["latexmk", "-pdf", "-interaction=nonstopmode",
             "LikelihoodInference.tex"], cwd=BUILD_LATEX)
    else:
        print("Compiling PDF with pdflatex (3 passes)...")
        for i in range(3):
            print(f"  Pass {i + 1}/3")
            run(["pdflatex", "-interaction=nonstopmode",
                 "LikelihoodInference.tex"], cwd=BUILD_LATEX)

    pdf = BUILD_LATEX / "LikelihoodInference.pdf"
    if pdf.exists():
        print(f"\nPDF built successfully: {pdf}")
    else:
        print("\nWarning: PDF file not found after compilation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
