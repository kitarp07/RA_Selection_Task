## Overview
This project performs bone segmentation on 3D knee CT volumes, including contour expansion, randomized mask adjustment, and anatomical landmark detection on the tibia.

## Repository Structure
- src/ — Scripts and modules for segmentation, expansion, adjustment, and landmark detection.
- notebooks/ — Data exploration and analysis notebooks.
- results/ — Saved segmentation masks generated from tasks 1 to 4.
- main.py — Full pipeline script to run all tasks sequentially.

## Installation
Install required packages with:

pip install -r requirements.txt

## Usage
Run the full processing pipeline with:
python main.py

## Report
See report.pdf for a detailed explanation of the approach and results.
