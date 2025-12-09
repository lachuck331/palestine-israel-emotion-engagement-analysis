# Reddit Engagement Analysis: Israel-Palestine Conflict

**Course:** DSC 161 / LIGN 167 - Final Project  
**Authors:** Ishaan Bal, Lacha Barton-Gluzman, Kartik Gopalani  
**Date:** December 8, 2025  

## Project Overview
This project investigates how emotional content (e.g., Joy, Anger, Fear) and topical themes (e.g., Violence, Politics) influence user engagement on Reddit discussions surrounding the Israel-Palestine conflict.

Using a dataset of 30,000 sampled posts, we applied automated feature extraction and LLM benchmarking to determine if specific sentiments or topics drive upvotes and comments.

## Repository Structure

| File | Description |
| :--- | :--- |
| `DSC161_FinalProject.ipynb` | **Main Analysis:** Performs feature extraction (NRCLex, Empath) on the 30k dataset, runs ANOVA tests for significance, and attempts engagement prediction using Random Forest. |
| `handcoding_analysis.ipynb` | **Validation:** Compares human-coded ground truth (100 posts) against **Gemini 3 Pro** predictions. Includes Inter-Coder Reliability (Cohen's Kappa) calculations. |
| `reddit_emotion_engagement.ipynb` | **Data Prep:** Initial exploration and cleaning of the raw Reddit corpus. |
| `HandCoding_human_vs_AI.csv` | The validation dataset containing 100 posts with Human vs. AI labels. |

## Workflow & Methodology

### 1. Human vs. AI Validation (`handcoding_analysis.ipynb`)
Before scaling up, we established a ground truth using a custom codebook for **8 Emotions** (Plutchik's Wheel) and **5 Topics** (Violence, Politics, Religion, Humanitarian, Personal Narrative).
* **Process:** 100 posts were hand-coded by two independent researchers.
* **Metric:** Calculated Cohen's Kappa for inter-coder reliability (~0.57).
* **AI Benchmarking:** We prompted **Gemini 3 Pro** to classify these posts and compared its accuracy (Emotion: 36%, Topic: 27%) against the human consensus.

### 2. Large-Scale Engagement Analysis (`DSC161_FinalProject.ipynb`)
We analyzed a random sample of 30,000 posts using automated tools to scale the analysis.
* **Feature Extraction:**
    * **Emotion:** Used `NRCLex` to detect dominant emotions.
    * **Topic:** Used `Empath` to categorize content into thematic clusters.
* **Statistical Analysis:** Performed ANOVA tests to check for significant differences in engagement between groups (e.g., Joy vs. Anger).
* **Modeling:** Trained a **Random Forest Regressor** to predict engagement scores based on emotion/topic features.

## Installation & Requirements

To replicate this analysis, you will need the following Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nrclex empath
