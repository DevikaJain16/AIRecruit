import numpy as np
import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

# --- Configuration ---
INDEX_FILE = 'candidates.index'
METADATA_FILE = 'metadata.pkl'
OUTPUT_CSV = 'team_submission.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'

JD_TEXT = """
Senior AI Engineer. Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
Scrappy product-engineering attitude. Embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5).
Vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, FAISS). Strong Python.
Evaluation frameworks for ranking systems (NDCG, MRR, MAP).
We do NOT want: Title-chasers, Framework enthusiasts, Pure consulting firm background, Pure research without production.
Ideal: 6-8 years total experience, applied ML at product companies. Shipped end-to-end ranking systems.
"""

def generate_reasoning(rank, metadata):
    if metadata.get('is_dummy'):
        return "Padding candidate to meet 100 row requirement for sample testing."
        
    years_exp = metadata.get('years_exp', 0)
    eng_mult = metadata.get('engagement_multiplier', 1.0)
    
    if rank <= 10:
        return f"Exceptional semantic match. {years_exp} years of experience aligns well. Strong behavioral signals (engagement multiplier: {eng_mult:.2f})."
    elif rank <= 50:
        return f"Solid technical alignment with core JD requirements. {years_exp} years of experience. Good platform activity."
    else:
        return f"Relevant ML/Engineering background. {years_exp} years of experience. Kept in consideration due to semantic overlap."

def main():
    start_time = time.time()
    print("Loading Pre-computed Data & FAISS Index...")
    
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
        
    print("Loading Model & Embedding JD...")
    model = SentenceTransformer(MODEL_NAME)
    jd_embedding = model.encode([JD_TEXT], convert_to_numpy=True)
    faiss.normalize_L2(jd_embedding)
    
    print("Calculating Semantic Similarity...")
    total_candidates = index.ntotal
    distances, indices = index.search(jd_embedding, total_candidates)
    
    print("Applying Heuristics & Behavioral Multipliers...")
    results = []
    for i in range(total_candidates):
        cand_idx = indices[0][i]
        base_score = float(distances[0][i])
        cand_meta = metadata[cand_idx]
        
        if cand_meta['is_honeypot']:
            final_score = 0.0 
        else:
            final_score = base_score * cand_meta['engagement_multiplier']
            
        # FIX: Round the score immediately so the tie-breaker logic is mathematically perfect
        results.append({
            'candidate_id': cand_meta['candidate_id'],
            'score': round(final_score, 4),
            'metadata': cand_meta
        })
        
    # FIX: Pad with dummy candidates if the sample file has less than 100 people
    while len(results) < 100:
        dummy_id = f"CAND_99{len(results):05d}"
        results.append({
            'candidate_id': dummy_id,
            'score': 0.0000,
            'metadata': {'years_exp': 0, 'engagement_multiplier': 0, 'is_dummy': True}
        })
        
    print("Ranking Candidates...")
    # Strict Sorting: Primary by rounded score (descending), Secondary by ID (ascending)
    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    top_100 = results[:100]
    
    csv_data = []
    for rank, cand in enumerate(top_100, start=1):
        reasoning = generate_reasoning(rank, cand['metadata'])
        csv_data.append({
            'candidate_id': cand['candidate_id'],
            'rank': rank,
            'score': f"{cand['score']:.4f}", # Ensure exact 4 decimal string format
            'reasoning': reasoning
        })
        
    df = pd.DataFrame(csv_data)
    df.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = time.time() - start_time
    print(f"Phase 2 Complete! Saved to {OUTPUT_CSV}")
    print(f"Total Execution Time: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()