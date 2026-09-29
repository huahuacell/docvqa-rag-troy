# InfographicsVQA Multimodal RAG

A multimodal Retrieval-Augmented Generation (RAG) system for **InfographicsVQA**, comparing an OCR-based Text RAG baseline, a SigLIP visual retrieval baseline, and a ColPali multi-vector retrieval optimization.

This project evaluates both retrieval quality and final visual question answering performance on the full **InfographicsVQA validation set (500 infographics, 2,801 questions)**.

---

## Overview

Infographics contain not only text, but also charts, spatial layouts, icons, tables, and visual relationships. A standard OCR-only RAG system may retrieve relevant text while losing important layout information.

This project implements and compares three retrieval pipelines.

### 1. Text RAG

```text
Official OCR
    ↓
OCR Text Chunks
    ↓
MiniLM Embeddings
    ↓
FAISS
    ↓
Top-K Text Context
    ↓
VLM Generator
```

Retriever:
- `sentence-transformers/all-MiniLM-L6-v2`
- FAISS `IndexFlatIP`
- OCR text chunks

### 2. SigLIP Visual RAG

```text
Infographic Images
    ↓
Overlapping Image Chunks
    ↓
SigLIP Embeddings
    ↓
FAISS
    ↓
Top-K Visual Chunks
    ↓
VLM Generator
```

Retriever:
- `google/siglip-base-patch16-224`
- FAISS `IndexFlatIP`
- 768 × 768 overlapping visual chunks

### 3. ColPali Multimodal RAG

```text
Infographic Images
    ↓
Visual Chunks
    ↓
ColPali Multi-Vector Representation
    ↓
Late Interaction Retrieval
    ↓
Top-K Visual Chunks
    ↓
VLM Generator
```

Retriever:
- `vidore/colpali-v1.3-hf`
- Multi-vector visual representation
- Late interaction scoring

ColPali is used as the main retrieval optimization over the single-vector SigLIP baseline.

---

## Architecture

The complete pipeline consists of three stages:

```text
Data Processing
      ↓
Indexing
      ↓
Retrieval
      ↓
Top-K Context
      ↓
Generation
      ↓
Evaluation
```

Three retrieval branches share the same final generation stage:

```text
                    ┌─ OCR + MiniLM + FAISS
InfographicsVQA ────┼─ SigLIP + FAISS
                    └─ ColPali Late Interaction
                              ↓
                         Top-K Context
                              ↓
                         VLM Generator
                              ↓
                      Predicted Answer
```

The generator is instructed to answer using only the retrieved context and return a short answer without explanation.

---

## Dataset

The experiments use the **InfographicsVQA validation split**.

| Item | Size |
|---|---:|
| Infographic images | 500 |
| Validation questions | 2,801 |
| Visual chunks | 4,199 |

Each question contains information such as:

```text
questionId
question
image_local_name
answers
answer_type
evidence
operation/reasoning
```

---

## Evaluation Metrics

### Document Recall@5

For each question, retrieval is counted as successful if at least one of the Top-5 retrieved chunks comes from the correct infographic.

The metric is calculated over all **2,801 validation questions**.

Evidence-level Recall@5 is not used as the main retrieval metric because InfographicsVQA does not provide complete chunk-level ground-truth evidence. Approximating evidence locations using OCR bounding boxes introduces additional localization errors.

### ANLS

Final answer quality is evaluated using the official **Average Normalized Levenshtein Similarity (ANLS)** evaluation script.

ANLS measures the similarity between generated answers and acceptable Ground Truth answers and is evaluated over all **2,801 validation questions**.

---

## Main Results

| Method | Document Recall@5 | ANLS |
|---|---:|---:|
| OCR + MiniLM + FAISS | **0.771867** | **0.311300** |
| SigLIP + FAISS | **0.661192** | **0.514900** |
| **ColPali** | **0.917172** | **0.674400** |

ColPali achieves the strongest overall performance.

Compared with SigLIP:

```text
Document Recall@5
0.661192 → 0.917172

ANLS
0.514900 → 0.674400
```

The results also show that higher document-level recall does not always imply higher final answer quality. For example, Text RAG retrieves the correct infographic more frequently than SigLIP, but its ANLS is substantially lower because OCR chunks may lose spatial and visual relationships.

---

## Case Study

Case studies are used to analyze the full evidence chain instead of evaluating only the final answer.

### Type 1 — Good Case

Effective answer evidence appears in the retrieved Top-K context and the generated answer is correct.

```text
Retrieval ✓
Evidence ✓
Answer ✓
```

### Type 2 — Bad Case: Retrieval Failure

The correct document or answer evidence is not retrieved, leading to an incorrect answer.

```text
Retrieval ✗
      ↓
Wrong Context
      ↓
Answer ✗
```

### Type 3 — Faithfulness / Overconfidence

Retrieved evidence is insufficient, but the model still produces a specific and confident-looking answer instead of expressing uncertainty.

```text
Insufficient Context
        ↓
Specific Answer
        ↓
Potential Faithfulness Risk
```

### Type 4 — Optimization Case

Text RAG or SigLIP fails, while ColPali retrieves useful evidence and answers correctly.

This category is used to analyze why multi-vector late interaction improves retrieval.

### Type 5 — Correct but Retrieval-Misaligned

The final answer is correct even though the target infographic is not retrieved or explicit supporting evidence is missing.

This demonstrates that:

```text
Correctness ≠ Faithfulness
```

A correct answer does not necessarily imply a successful or evidence-grounded RAG pipeline.

---

## Representative Optimization Example

**Question**

```text
Which platform is suitable for software providers?
```

**Ground Truth**

```text
LinkedIn
```

| Method | Retrieval | Prediction |
|---|---|---|
| Text RAG | Miss | Hadoop |
| SigLIP | Miss | Microsoft Azure |
| ColPali | Hit (Rank 1) | LinkedIn |

ColPali retrieves the correct visual region containing the relationship between **LinkedIn** and **Software providers**, while the baseline retrievers are distracted by unrelated software-related content.

This example illustrates that the improvement comes not only from retrieving similar keywords, but from retrieving the correct visual relationship.

---

## Project Structure

```text
docvqa-rag-troy/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── annotations/
│   ├── images/
│   ├── ocr/
│   └── chunks/
│
├── src/
│   ├── data_processor.py
│   ├── evaluator.py
│   ├── generator.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── text_indexer.py
│   ├── text_retriever.py
│   ├── colpali_indexer.py
│   └── colpali_retriever.py
│
├── scripts/
│   ├── run_text_rag.py
│   ├── run_visual_rag.py
│   ├── run_colpali.py
│   ├── make_submission.py
│   ├── find_cases.py
│   └── export_case_studies.py
│
├── results/
│   ├── text_document_recall.json
│   ├── text_document_recall_details.json
│   ├── text_predictions.json
│   ├── siglip_recall.json
│   ├── siglip_document_recall_details.json
│   ├── siglip_predictions.json
│   ├── colpali_recall.json
│   ├── colpali_document_recall_details.json
│   ├── colpali_predictions.json
│   └── case_studies/
│
└── eval/
    ├── gt/
    ├── submissions/
    └── evaluate.py
```

Large images, embeddings, model weights, and FAISS indexes may be excluded from Git using `.gitignore`.

---

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / Kaggle

```bash
pip install -r requirements.txt
```

If FAISS is unavailable:

```bash
pip install faiss-cpu
```

---

## Environment Variables

The generation stage uses an OpenAI-compatible VLM API.

### PowerShell

```powershell
$env:ECNU_API_KEY="YOUR_API_KEY"
$env:ECNU_BASE_URL="YOUR_API_BASE_URL"
```

Do **not** commit API keys to GitHub.

---

## Running the Experiments

### Text RAG

```powershell
python -m scripts.run_text_rag
```

### SigLIP Visual RAG

Build the visual index:

```powershell
python -m scripts.run_visual_rag --mode index --force
```

Evaluate Document Recall@5:

```powershell
python -m scripts.run_visual_rag --mode recall --force
```

GPU acceleration is recommended for visual indexing.

### ColPali Retrieval

```powershell
python -m scripts.run_colpali --mode index
```

```powershell
python -m scripts.run_colpali --mode retrieve
```

Evaluate retrieval:

```powershell
python -m scripts.run_colpali --mode recall --force
```

Generate answers:

```powershell
python -m scripts.run_colpali --mode generate
```

---

## Official ANLS Evaluation

Convert predictions into the official submission format:

```powershell
python -m scripts.make_submission `
  --predictions results/colpali_predictions.json `
  --output eval/submissions/colpali_rag.json
```

Run the official evaluator:

```powershell
python eval/evaluate.py `
  -g eval/gt/infographicVQA_val_v1.0.json `
  -s eval/submissions/colpali_rag.json
```

Expected final ColPali result:

```text
Overall ANLS: 0.6744
```

---

## Case Study Analysis

Generate candidate cases:

```powershell
python scripts/find_cases.py
```

Candidate results are stored in:

```text
results/case_study_candidates.txt
```

Export retrieved visual chunks for selected examples:

```powershell
python scripts/export_case_studies.py
```

The exported cases are stored under:

```text
results/case_studies/
```

Each case contains retrieval information and visual evidence that can be used to analyze retrieval success, failure, optimization effects, and faithfulness.

---

## Key Findings

The experiments highlight three important observations.

**First**, retrieving the correct infographic does not guarantee that the relevant answer evidence is included in the retrieved chunk.

**Second**, visual retrieval preserves layout and graphical relationships that may be lost when OCR text is flattened into text chunks.

**Third**, ColPali's multi-vector late-interaction retrieval substantially improves document retrieval and final question answering performance, but retrieval failures and generation errors still remain.

Reliable multimodal RAG therefore requires evaluation of the complete chain:

```text
Correct Document
        ↓
Sufficient Evidence
        ↓
Evidence-Grounded Answer
```

Recall@5 and ANLS alone cannot fully characterize this process.

---

## Limitations

- The experiments use a fixed validation document collection of 500 infographics.
- Document Recall@5 measures document-level retrieval rather than exact answer-evidence retrieval.
- Case studies are selected for diagnostic analysis and should not be interpreted as an estimate of the overall hallucination rate.
- The current system does not include an explicit evidence citation mechanism or confidence-based refusal strategy.

---

## Future Work

Potential improvements include:

- Hybrid OCR + visual retrieval
- Better visual chunk boundary handling
- Document-level retrieval followed by local evidence retrieval
- Reranking of retrieved chunks
- Explicit evidence citation
- Faithfulness judging
- Evidence-aware answer generation
- Refusal when retrieved evidence is insufficient

---

## Report

The full experimental report contains:

```text
System Architecture
Evaluation Metrics
Quantitative Results
Case Studies
Retrieval / Generation Failure Analysis
Faithfulness Analysis
Optimization Analysis
Limitations
Future Improvements
```

For detailed methodology, quantitative results, and qualitative case analysis, see the accompanying experimental report.

---

## References

1. Mathew, M., Bagal, V., Tito, R., et al.  
   **InfographicVQA.**  
   IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.

2. Zhai, X., Mustafa, B., Kolesnikov, A., et al.  
   **Sigmoid Loss for Language Image Pre-Training.**

3. Faysse, M., Sibille, H., Wu, T., et al.  
   **ColPali: Efficient Document Retrieval with Vision Language Models.**

---

## License

This repository is intended for academic and educational use.

Please follow the licenses and terms of the original InfographicsVQA dataset and all pretrained models used in this project.
