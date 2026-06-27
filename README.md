# Redrob Intelligent Candidate Discovery

## Architecture
This system uses a **Two-Stage Semantic Ranker with Behavioral Multipliers**. 
To comply with the strict ≤ 5-minute CPU-only rule and zero-network constraint, this solution separates heavy AI embedding into an offline pre-computation phase, and utilizes highly optimized local vector search (FAISS) for the online ranking.

## Setup & Reproduction
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
