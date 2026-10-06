# CVD Synthetic Data Lab

A Python-based synthetic cardiovascular patient data generation and analytics application.

This project demonstrates how synthetic patient records can be generated from the statistical characteristics of a real cardiovascular disease dataset and evaluated against the original data.

The implementation was developed as part of **Smart Computing Using Advanced Python** and is directly connected to the project:

> **An Intelligent Clinical Decision Support System for Cardiovascular Disease Risk Assessment Using Visual Analytics Dashboard**

---

## Overview

Healthcare datasets can contain sensitive patient information and may also be limited in size or availability.

Synthetic data generation provides a way to create new patient-like records while preserving important statistical characteristics of an original dataset.

This project uses a **Gaussian Copula-based synthetic data generation approach** to learn relationships between cardiovascular health variables and generate new synthetic patient records.

The generated data can then be used for:

- Model development
- Testing
- Data analysis
- Visualization
- Demonstration
- Experimental clinical analytics

---

## How It Works

```text
Original CVD Dataset
        │
        ▼
Data Analysis & Preprocessing
        │
        ▼
Learn Feature Distributions
        │
        ▼
Learn Correlation Structure
        │
        ▼
Gaussian Copula
        │
        ▼
Generate Synthetic Records
        │
        ▼
Real vs Synthetic Evaluation