# Workflow Automation Prototype

## Overview
This Python-based workflow automation prototype consolidates BCM incident inputs from emails, Teams messages, screenshots, and text files into a standardised Excel report.

## Problem
The BCM team previously had to manually read emails, copy details into Excel, validate missing fields, and prepare summary reporting.

## Solution
The tool automates incident intake by extracting text, applying OCR for screenshots, categorising incident details, validating missing fields, and generating a structured Excel report.

## Key Features
- OCR extraction from screenshots
- Local AI-assisted incident summarisation
- Standardised incident intake table
- Missing field validation
- Executive summary tab
- Excel report generation

## Technologies Used
- Python
- Pandas
- OpenPyXL / XlsxWriter
- Tesseract OCR
- Ollama local LLM
- Logistic regression classifier

## How To Run
1. Install Python dependencies.
2. Install Tesseract OCR if using screenshots.
3. Install Ollama if using AI summarisation.
4. Place files into `input/raw_incidents`.
5. Run the tool.
6. Open the generated Excel report in `output`.

## Limitations
- OCR accuracy depends on screenshot quality.
- AI summaries require human review.
- Not yet connected directly to Outlook, Teams, or SharePoint.
- Prototype is not production-governed.

## Future Enhancements
- Direct Outlook/Teams integration
- Improved classifier training dataset
- Web dashboard
- Approval workflow
- Audit logging
