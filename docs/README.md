# Sign-Language Landmark Pipeline — Documentation

Technical documentation for the ASL sign-language landmark extraction and
verification system. Read in order, or jump to what you need.

| # | Document | What it covers |
|---|----------|----------------|
| 01 | [Overview](01_overview.md) | What the system does, the full directory tree, tech stack |
| 02 | [Architecture](02_architecture.md) | Layered design, module responsibilities, design principles |
| 03 | [Processing Pipeline](03_pipeline.md) | Step-by-step extraction logic and key algorithms |
| 04 | [File Reference](04_file_reference.md) | Every `.py` file explained function by function |
| 05 | [Data Formats](05_data_formats.md) | `.npy` shapes, feature layout, folder structure, JSON schemas |
| 06 | [Developer Guide](06_developer_guide.md) | How to run, extend, and debug |
| 07 | [Future System Design](07_future_design.md) | Suggested design for training + real-time inference |

Diagrams live in [`diagrams/`](diagrams). The combined PDF is
`TECHNICAL_DOCUMENTATION.pdf` (regenerate with `python docs/build_pdf.py`).

## Regenerating the docs assets
```
python docs/make_diagrams.py    # rebuild architecture / data-flow / verification diagrams
python dataset_report.py        # rebuild the verification infographics (processed_dataset/reports/)
python docs/build_pdf.py        # rebuild TECHNICAL_DOCUMENTATION.pdf from these markdown files
```
