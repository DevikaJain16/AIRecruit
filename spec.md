Here is the comprehensive technical specification to guide our development sprint.

---

 **Technical Specification: Local AI Recruitment Ranker**
**1. System Architecture Overview**

The system is divided into two strict phases to comply with the $\le$ 5-minute, CPU-only runtime limit.

* **Phase 1: Offline Pre-computation (Unrestricted)**
* We will parse the raw JSON data, run heavy Natural Language Processing (NLP) to extract features, and use a local embedding model to generate dense vectors for all 100K candidates. This data will be saved to disk.


* **Phase 2: Local Online Ranker (The 5-Minute Sprint)**
* This is the script that gets evaluated. It will load the pre-computed vectors, execute a fast mathematical search (Cosine Similarity), apply heuristic penalties (e.g., dropping honeypots), generate the text reasoning, and output the exact 100-row CSV.



**2. Tech Stack Requirements**

* **Language:** Python 3.11+
* **Vector Search & Math:** `faiss-cpu` (extremely fast local vector indexing), `numpy`, `pandas`.
* **Embeddings:** `sentence-transformers` (specifically the `all-MiniLM-L6-v2` model, which is highly accurate but small enough to run fast on a CPU).
* **Sandbox Deployment:** Streamlit (easy to wrap a Python script into a UI for the required demo link).

---

 **3. Data Processing & Feature Engineering (Phase 1)**

Before the clock starts, we must shape the data.

* **Semantic Concatenation:** We will merge the candidate's `headline`, `summary`, and `description` from their `career_history` into a single text block. We will do the same for the Job Description.
* **Vectorization:** We will pass these text blocks through `all-MiniLM-L6-v2` to create a 384-dimensional vector for each candidate.
* **Behavioral Extraction:** We will extract `profile_completeness_score`, `recruiter_response_rate`, and `last_active_date` to calculate an offline "Engagement Multiplier" (ranging from 0.5 to 1.0).

 **4. Core Ranking Algorithm (Phase 2)**

When `rank.py` is executed, it will perform the following pipeline:

**Step A: Honeypot Filtering (The Trapdoor)**
The system will immediately drop candidates who fail logical constraints to keep our honeypot rate at 0%:

* *Logic 1:* If `years_of_experience` $>$ candidate age (inferred from education start year).
* *Logic 2:* If skill `proficiency` == "expert" but `duration_months` == 0.
* *Logic 3:* If `github_activity_score` $>$ 0 but `linkedin_connected` == false (or similar platform contradictions).

**Step B: The Hybrid Scoring Engine**
Calculate the final score for the remaining candidates using this equation:
`Final Score = (Semantic Vector Similarity) * (Engagement Multiplier) * (Job Hopper Penalty)`

* *Semantic Vector Similarity:* FAISS calculates the distance between the JD vector and the candidate vector.
* *Job Hopper Penalty:* If the average duration in `career_history` is $< 18$ months, apply a 0.85x penalty to align with the JD's strict "No Title-chasers" rule.

**Step C: Deterministic Reasoning Generation**
Because we cannot call an LLM API to write the reasoning column, we will build a dynamic string constructor based on the candidate's highest-scoring attributes:

* *Template Logic:* "Target role fit with [X] years of experience. Strong background in [Top Skill 1] and [Top Skill 2]. [Engagement condition]."
* *Output Example:* "Target role fit with 6.0 years of experience. Strong background in Pinecone and FAISS. High recruiter engagement (0.91 response rate)."

---

 **5. Delivery Checklist**

To ensure the submission is valid and passes `validate_submission.py`:

* [ ] `team_xxx.csv` output containing exactly 101 rows (1 header + 100 data rows).
* [ ] Scores strictly non-increasing by rank.
* [ ] GitHub repository containing the offline scripts, the `rank.py` script, and the `submission_metadata.yaml`.
* [ ] Hosted Streamlit app allowing the user to upload a mini-JSON file and download the ranked CSV.

---

This architecture guarantees we stay within the compute and network constraints while delivering semantic-level accuracy that crushes standard keyword matching.

