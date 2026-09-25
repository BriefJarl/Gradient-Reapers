# Gradient Reapers — Business Entity Resolution

> Amazon ML Challenge 2026

## Problem

Build an ML-based **Business Entity Resolution (ER)** system that identifies which records from noisy Source 2 and Source 3 correspond to each business in the clean Source 1 reference dataset.

The system receives only:

- Business name
- Business address

There are no shared identifiers between sources.

For every Source 1 entity, the system predicts **all matching Source 2 and Source 3 IDs**, including the possibility of no match.

---

## Objective

The solution is optimized for **Macro F0.5**, where precision is weighted more heavily than recall.

Therefore, the system follows a conservative matching strategy:

**High-confidence matches → accept**

**Uncertain matches → reject**

At the same time, blocking is designed for high recall so that true matches are not removed before ML scoring.

---

# System Architecture

```text
                         ALL RECORDS
                              │
                              ▼
                 ┌─────────────────────────┐
                 │      NORMALIZATION       │
                 │  Name + Address         │
                 │  multiple representations│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │        BLOCKING          │
                 │ Name / Address / Hybrid │
                 │ High-recall candidate   │
                 │ generation               │
                 └────────────┬────────────┘
                              │
                              ▼
                    CANDIDATE PAIRS
                              │
                              ▼
                 ┌─────────────────────────┐
                 │   FEATURE ENGINEERING   │
                 │ Name similarity         │
                 │ Address similarity      │
                 │ Token / character       │
                 │ Exact / partial signals │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    MATCH CLASSIFIER     │
                 │   XGBoost / LightGBM    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    DECISION LAYER       │
                 │ Conservative threshold  │
                 │ Entity-level matching   │
                 └────────────┬────────────┘
                              │
                              ▼
                    FINAL MATCHES
                              │
                              ▼
                  matching_results.tsv
```

---

# ML Pipeline

```text
TRAINING DATA
      │
      ▼
Data Exploration
      │
      ▼
Normalization
      │
      ▼
Multiple Blocking Strategies
      │
      ▼
Candidate Generation
      │
      ▼
Candidate Recall Evaluation
      │
      ▼
Pair Feature Engineering
      │
      ▼
Supervised Pair Classifier
      │
      ▼
Threshold Optimization
      │
      ▼
Macro F0.5 Validation
      │
      ├───────────────┐
      │               │
 Improve Blocking   Improve Model
      │               │
      └───────┬───────┘
              │
              ▼
       FINAL PIPELINE
              │
              ▼
          TEST DATA
              │
              ▼
        Predictions
              │
       ┌──────┴──────┐
       ▼             ▼
candidate_pairs.tsv  matching_results.tsv
```

---

# Target Architecture

The final system is designed as a multi-stage entity-resolution pipeline:

```text
                         INPUT
                           │
             ┌─────────────┴─────────────┐
             │                           │
         Source 1                    Source 2 / 3
             │                           │
             └─────────────┬─────────────┘
                           │
                     NORMALIZATION
                           │
             ┌─────────────┼─────────────┐
             │             │             │
         Name Block    Address Block   Hybrid Block
             │             │             │
             └─────────────┼─────────────┘
                           │
                    UNION CANDIDATES
                           │
                     HARD FILTERS
                           │
                   FEATURE ENGINE
                           │
              ┌────────────┴────────────┐
              │                         │
        Name Similarity           Address Similarity
              │                         │
              └────────────┬────────────┘
                           │
                     META FEATURES
                           │
                           ▼
                   XGBoost / LightGBM
                           │
                      Probability
                           │
                           ▼
                Entity-Level Decision
                           │
                ┌──────────┴──────────┐
                │                     │
          High Confidence        Low Confidence
                │                     │
              MATCH                NO MATCH
                │                     │
                └──────────┬──────────┘
                           │
                           ▼
                    FINAL RESULTS
```

---

# Key Design Principles

### 1. High-Recall Blocking

Blocking creates the candidate set before ML scoring.

A true match that is removed during blocking can never be recovered later.

Therefore:

**Blocking → maximize candidate recall**

---

### 2. Precision-Aware Matching

Because F0.5 gives greater importance to precision, the final decision layer is conservative.

We prefer:

```text
Uncertain pair → NO MATCH
```

rather than creating a potentially incorrect merge.

---

### 3. Multiple Representations

Names and addresses will be represented using multiple normalized forms rather than relying on a single string transformation.

Examples include:

- Lowercasing
- Punctuation normalization
- Whitespace normalization
- Token normalization
- Alphanumeric representations
- Character-level representations
- Address component/token representations

---

### 4. Multiple Blocking Strategies

Candidate generation will combine several high-recall blocking rules.

Examples:

```text
Name-based blocking
Address-based blocking
Token-based blocking
Hybrid name + address blocking
```

Candidate sets are then combined before pair scoring.

---

### 5. Pair-Level ML

Each candidate pair is converted into numerical similarity features.

Example feature groups:

```text
NAME FEATURES
├── Exact similarity
├── Token similarity
├── Character similarity
└── Edit / fuzzy similarity

ADDRESS FEATURES
├── Token similarity
├── Character similarity
├── Numeric-token agreement
└── Address component similarity

META FEATURES
├── Blocking rule agreement
├── Missing-field indicators
└── Other pair-level signals
```

The resulting features are passed to a supervised classifier.

---

# Outputs

The final solution produces:

```text
output/
├── candidate_pairs.tsv
└── matching_results.tsv
```

### `candidate_pairs.tsv`

Contains the generated candidate pairs after blocking.

### `matching_results.tsv`

Contains the final predicted Source 2 and Source 3 matches for every Source 1 entity.

---

# Validation

Before submission, the generated files are checked using:

```text
utils/validate_submission.py
```

The complete pipeline must be reproducible from the project code and configuration.

---

# Team Workflow

The repository uses GitHub for collaboration.

Recommended workflow:

```text
main
 │
 ├── feature/normalization
 ├── feature/blocking
 ├── feature/features
 ├── feature/model
 └── feature/evaluation
```

Changes should be developed on feature branches and merged into `main` after testing.

---

# Project Goal

Build a robust, scalable and precision-aware Business Entity Resolution system combining:

**Normalization → High-Recall Blocking → Candidate Generation → Feature Engineering → Supervised Matching → Conservative Decision Layer → Validated Submission**
