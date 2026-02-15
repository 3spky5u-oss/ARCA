"""
Generate synthetic test PDFs for QA validation.

Creates PDFs at different quality tiers with known ground truth.
"""

import random
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError("PyMuPDF (fitz) is required: pip install pymupdf")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
QA_DIR = BASE_DIR / "data" / "qa_reference"
DOCS_DIR = QA_DIR / "docs"
EXPECTED_DIR = QA_DIR / "expected"


# Sample content for test documents
CLEAN_CONTENT = """Research Methodology and Data Analysis

1. Introduction

This report presents a comprehensive analysis of experimental data collected during the laboratory testing program. The methodology follows established protocols for measurement calibration, data acquisition, and statistical evaluation.

2. Data Classification

Measurements are classified based on their precision and repeatability characteristics. The Standard Classification Framework (SCF) is widely used in analytical practice.

2.1 High-Precision Measurements

High-precision measurements include calibrated instruments and reference standards. They are classified based on uncertainty levels:
- Grade A: uncertainty less than 0.01 units
- Grade B: uncertainty between 0.01 and 0.10 units

2.2 General Measurements

General measurements include field instruments and portable sensors. They are classified based on accuracy:
- Type I: accuracy within 5 percent of reference value
- Type II: accuracy within 10 percent of reference value

3. Statistical Analysis

The statistical significance of results is determined by the standard hypothesis testing framework. The general regression equation is:

Y = a + bX + cX^2 + epsilon

Where:
- a = intercept coefficient (dimensionless)
- b = linear coefficient (per unit)
- c = quadratic coefficient (per unit squared)
- X = independent variable
- epsilon = random error term

4. Error Analysis

Sources of measurement error include:
- Systematic error: instrument calibration drift
- Random error: environmental fluctuations
- Sampling error: non-representative selection

5. Conclusion

Rigorous methodology and careful error analysis are essential for reliable experimental results.
"""

MIXED_LAYOUT_CONTENT = """Laboratory Testing Summary Report

PROJECT INFORMATION
===================

Project Name: Materials Analysis Program
Client: Acme Research Ltd.
Reference: LAB-2024-0042
Date: January 2024

EXECUTIVE SUMMARY
=================

This report presents the results of a comprehensive materials testing program conducted on submitted specimens.

TENSILE TEST RESULTS
====================

Sample ID    | Load (kN) | Stress (MPa)    | Strain (%)      | Status
-------------|-----------|-----------------|-----------------|--------
SP-1-A       | 12.5      | 245.3           | 0.82            | Pass
SP-1-B       | 11.8      | 231.6           | 0.79            | Pass
SP-1-C       | 8.2       | 160.9           | 1.45            | FAIL
SP-2-A       | 13.1      | 257.0           | 0.75            | Pass
SP-2-B       | 12.9      | 253.1           | 0.77            | Pass

THERMAL ANALYSIS RESULTS
=========================

Specimen | Peak Temp (C)  | Threshold | Status
---------|----------------|-----------|--------
TH-1     | 142.5          | 200.0     | Pass
TH-2     | 218.3          | 200.0     | FAIL
TH-3     | 98.7           | 200.0     | Pass

CONCLUSIONS
===========

1. Tensile failure detected in specimen SP-1-C
2. Thermal threshold exceeded at TH-2
3. Additional testing recommended

RECOMMENDATIONS
===============

Based on the findings, we recommend:
- Repeat testing of failed specimens
- Extended thermal cycling analysis
- Statistical evaluation of variance
"""

SCANNED_CONTENT = """Inspection Record IR-1

Project: Structural Assessment
Reference: INS-2024-007
Date: March 2024

Zone    Description                          Rating
ID                                           Score
------  -----------------------------------  -------
Z-01    SURFACE COATING, intact, no defects   --
Z-02    PANEL SECTION A, minor wear          12
        trace discoloration, dry
Z-03    PANEL SECTION B, moderate wear       18
        occasional surface cracks
Z-04    JOINT ASSEMBLY, tight, no play       25
        minor oxidation present
Z-05    BASE PLATE, corroded, weakened       R
        requires replacement

Ambient temperature: 21.5 C during inspection
End of inspection zone: Section 5
"""

TERRIBLE_CONTENT = """LAB R3PORT SUMM4RY

Ref3rence: RPT-2024-00B
Date: Fe8ruary 2024

T3st Resu1ts:
- Sampl3 A: 12.5 kN
- Sampl3 B: 8.3 kN
- Sampl3 C: 15.1 kN

N0tes:
Calibrati0n drift det3cted
Retest recomm3nded
"""


def create_clean_digital_pdf(output_path: Path, content: str):
    """Create a clean, well-formatted digital PDF with searchable text."""
    doc = fitz.open()

    # Page settings
    page_width = 612  # Letter size
    page_height = 792
    margin = 72  # 1 inch

    # Split content by double newlines to create paragraphs
    paragraphs = content.split("\n\n")
    fontsize = 11
    line_height = fontsize * 1.4

    current_page = None
    y_pos = margin
    max_y = page_height - margin

    for para in paragraphs:
        # Create new page if needed
        if current_page is None or y_pos > max_y - line_height * 3:
            current_page = doc.new_page(width=page_width, height=page_height)
            y_pos = margin

        # Split paragraph into lines
        lines = para.split("\n")
        for line in lines:
            if y_pos > max_y:
                current_page = doc.new_page(width=page_width, height=page_height)
                y_pos = margin

            # Insert text at position (creates searchable text)
            text_point = fitz.Point(margin, y_pos)
            current_page.insert_text(text_point, line, fontname="helv", fontsize=fontsize)
            y_pos += line_height

        # Extra space between paragraphs
        y_pos += line_height * 0.5

    doc.save(str(output_path))
    doc.close()


def create_mixed_layout_pdf(output_path: Path, content: str):
    """Create a PDF with tables and multiple columns."""
    doc = fitz.open()

    page_width = 612
    page_height = 792
    margin = 54
    fontsize = 9
    line_height = fontsize * 1.3

    current_page = None
    y_pos = margin
    max_y = page_height - margin

    for line in content.split("\n"):
        if current_page is None or y_pos > max_y:
            current_page = doc.new_page(width=page_width, height=page_height)
            y_pos = margin

        text_point = fitz.Point(margin, y_pos)
        current_page.insert_text(text_point, line, fontname="cour", fontsize=fontsize)  # Monospace for table alignment
        y_pos += line_height

    doc.save(str(output_path))
    doc.close()


def create_scanned_pdf(output_path: Path, content: str):
    """Create a PDF that simulates a scanned document with noise."""
    doc = fitz.open()

    page_width = 612
    page_height = 792
    margin = 72
    fontsize = 10
    line_height = fontsize * 1.3

    page = doc.new_page(width=page_width, height=page_height)
    y_pos = margin + 3  # Slight offset to simulate scan skew

    for line in content.split("\n"):
        text_point = fitz.Point(margin + 5, y_pos)
        page.insert_text(text_point, line, fontname="cour", fontsize=fontsize)
        y_pos += line_height

    # Add some "noise" shapes to simulate scan artifacts
    for _ in range(20):
        x = random.randint(50, 560)
        y = random.randint(50, 740)
        size = random.uniform(0.5, 2)
        gray = random.uniform(0.7, 0.9)

        shape = page.new_shape()
        shape.draw_circle(fitz.Point(x, y), size)
        shape.finish(color=(gray, gray, gray), fill=(gray, gray, gray))
        shape.commit()

    doc.save(str(output_path))
    doc.close()


def create_terrible_pdf(output_path: Path, content: str):
    """Create a heavily degraded PDF simulating poor OCR quality."""
    doc = fitz.open()

    page_width = 612
    page_height = 792
    margin = 80
    fontsize = 9
    line_height = fontsize * 1.3

    page = doc.new_page(width=page_width, height=page_height)
    y_pos = margin

    for line in content.split("\n"):
        text_point = fitz.Point(margin, y_pos)
        page.insert_text(text_point, line, fontname="cour", fontsize=fontsize)
        y_pos += line_height

    # Heavy noise overlay
    for _ in range(100):
        x = random.randint(30, 580)
        y = random.randint(30, 760)
        size = random.uniform(1, 4)
        gray = random.uniform(0.5, 0.85)

        shape = page.new_shape()
        shape.draw_circle(fitz.Point(x, y), size)
        shape.finish(color=(gray, gray, gray), fill=(gray, gray, gray))
        shape.commit()

    # Add some lines to simulate creases/folds
    shape = page.new_shape()
    for _ in range(3):
        x1 = random.randint(0, 612)
        y1 = random.randint(0, 792)
        x2 = random.randint(0, 612)
        y2 = random.randint(0, 792)
        shape.draw_line(fitz.Point(x1, y1), fitz.Point(x2, y2))
        shape.finish(color=(0.8, 0.8, 0.8), width=0.5)
    shape.commit()

    doc.save(str(output_path))
    doc.close()


def generate_all_test_docs(force: bool = False):
    """Generate all test documents and ground truth files."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    # Document configurations
    docs = [
        ("clean_digital.pdf", CLEAN_CONTENT, create_clean_digital_pdf),
        ("mixed_layout.pdf", MIXED_LAYOUT_CONTENT, create_mixed_layout_pdf),
        ("scanned_poor.pdf", SCANNED_CONTENT, create_scanned_pdf),
        ("terrible_ocr.pdf", TERRIBLE_CONTENT, create_terrible_pdf),
    ]

    generated = []

    for filename, content, generator in docs:
        pdf_path = DOCS_DIR / filename
        txt_path = EXPECTED_DIR / filename.replace(".pdf", ".txt")

        if pdf_path.exists() and not force:
            print(f"Skipping {filename} (already exists)")
            continue

        # Generate PDF
        generator(pdf_path, content)
        print(f"Generated: {pdf_path}")

        # Write ground truth
        txt_path.write_text(content, encoding="utf-8")
        print(f"Generated: {txt_path}")

        generated.append(filename)

    return generated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA test documents")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    generated = generate_all_test_docs(force=args.force)

    if generated:
        print(f"\nGenerated {len(generated)} test documents")
    else:
        print("\nNo new documents generated (use --force to overwrite)")
