# Symbolic-Migraine-Engine
Symbolic Reasoning for ICHD-3 Migraine Diagnosis: Integrating Patient Diary with Clinical Criteria
A hybrid neural-symbolic AI system that processes patient diary text to extract clinical entities and applies ICHD-3 diagnostic criteria for migraine subtype classification.

## Overview

This project develops a multi-stage pipeline that:
1. Generates synthetic 30-day patient diaries using LLMs (Qwen 2.5 and LLaMA 3.1)
2. Extracts clinical entities using fine-tuned ClinicalBERT
3. Applies ICHD-3 diagnostic rules through symbolic reasoning
4. Classifies patients into six migraine subtypes
5. Comparison to CatBoost baseline and ground truth labels

**Research Question:** If a patient writes naturally about their symptoms, can our AI system extract the right information and reach the correct diagnosis?

## Repository Structure

```
├── Data info.rtf              # Original Kaggle dataset information
├── migraine_data.csv          # Raw patient data (400 patients, 24 features)
├── migraine_with_id.csv       # Processed data with Patient_ID
├── criteriaof6.txt            # ICHD-3 diagnostic criteria for 6 subtypes
├── data_preprocess.ipynb      # Data preprocessing and Patient_ID assignment
├── prompt1.ipynb              # LLM diary generation - Prompt variation 1
├── prompt2.ipynb              # LLM diary generation - Prompt variation 2
├── prompt3.ipynb              # LLM diary generation - Prompt variation 3
├── prompt4.ipynb              # LLM diary generation - Prompt variation 4
├── diary_preprocess.ipynb     # Diary text cleaning and normalization
├── clinicalBERT.ipynb         # ClinicalBERT fine-tuning for NER + post-processing
├── eval_ner.ipynb             # NER performance evaluation
├── symbolic_engine.ipynb      # Experta-based diagnostic reasoning engine
├── eval_symbolic.ipynb        # Symbolic reasoning evaluation
├── catboost.ipynb             # CatBoost ML baseline model
├── modelcompare.ipynb         # Comparison between symbolic and ML approaches
├── engine_output.csv          # Diagnostic outputs from symbolic engine
└── README.md                  # This file
```

## Pipeline

### 1. Data Preprocessing
- **Input:** Kaggle migraine dataset (400 patients, 24 clinical features)
- **Process:** Assign Patient_ID, split 200/200 for Qwen and LLaMA
- **Output:** `migraine_with_id.csv`

### 2. Synthetic Diary Generation
- **Models:** Qwen 2.5 and LLaMA 3.1
- **Process:** Generate 30-day patient diaries using 4 prompt variations
- **Evaluation:** TTR, similarity, perplexity, F-K grade, typo rate
- **Output:** Realistic first-person narrative diaries

### 3. Diary Preprocessing
- **Process:** Text cleaning, typo correction, synonym normalization, sentence segmentation
- **Output:** Preprocessed JSON diaries for NER

### 4. Named Entity Recognition
- **Model:** Fine-tuned ClinicalBERT
- **Features:** Automatic BIO tagging, negation handling, pattern matching
- **Entities:** 23 clinical features (pain characteristics, aura symptoms, associated symptoms)
- **Performance:** 68% mean information preservation

### 5. Symbolic Reasoning Engine
- **Rules:** ICHD-3 diagnostic criteria for 6 migraine subtypes
- **Output:** Rule-based diagnosis with explicit reasoning
- **Performance:** 49% accuracy on extracted features, 79% on structured data

### 6. Machine Learning Baseline
- **Model:** CatBoost gradient boosting classifier
- **Features:** 23 clinical features from original structured data
- **Performance:** 87% accuracy

## Key Results

- **ClinicalBERT NER:** 68% mean accuracy, >90% on common symptoms, struggles with frequency (93.5% mismatch) and rare features (40-50% missing)
- **Symbolic Engine:** 49% accuracy (extracted) vs 79% (structured) - 30-point information loss gap
- **CatBoost ML:** 87% accuracy on structured data
- **Trade-off:** ML achieves higher accuracy, symbolic reasoning provides interpretability

## Requirements

```
python>=3.10
torch
transformers
catboost
pandas
numpy
scikit-learn
```

## Usage

1. **Data Preparation:**
   ```bash
   jupyter notebook data_preprocess.ipynb
   ```

2. **Generate Diaries:**
   ```bash
   jupyter notebook prompt1.ipynb  # Repeat for prompt2-4
   jupyter notebook diary_preprocess.ipynb
   ```

3. **Train NER Model and Perform post-processing:**
   ```bash
   jupyter notebook clinicalBERT.ipynb
   ```

4. **Run Symbolic Engine and Catboost:**
   ```bash
   jupyter notebook symbolic_engine.ipynb
   jupyter notebook catboost.ipynb
   ```

5. **Compare Models:**
   ```bash
   jupyter notebook modelcompare.ipynb
   ```

## Limitations

- Synthetic diaries may not capture full patient language variability
- Small dataset (400 patients)
- Not validated on real patient diaries
- Frequency extraction requires significant improvement
- Rare symptoms show poor extraction rates
  

## Acknowledgments

This work was completed as part of the Symbolic AI course in the Department of Biomedical Informatics at Columbia University. Special thanks to Professor Weng and teaching assistants Fang Yi and James for their guidance and support.

## License

This project is for academic research purposes.
