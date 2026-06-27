import json
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import time

# Configuration
CANDIDATES_FILE = 'India_runs_data_and_ai_challenge/sample_candidates.json'
OUTPUT_INDEX = 'candidates.index'
OUTPUT_METADATA = 'metadata.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2'

def extract_candidate_text(candidate):
    """Merges candidate profile and experience into a single semantic block."""
    profile = candidate.get('profile', {})
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    
    experience = []
    for job in candidate.get('career_history', []):
        experience.append(f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}")
    
    skills = [skill.get('name', '') for skill in candidate.get('skills', [])]
    
    # Combine into one rich text string
    full_text = f"Headline: {headline}. Summary: {summary}. Experience: {' | '.join(experience)}. Skills: {', '.join(skills)}."
    return full_text

def calculate_multipliers(candidate):
    """Calculates engagement multipliers and flags honeypots."""
    signals = candidate.get('redrob_signals', {})
    
    # 1. Engagement Multiplier (Penalize low response rates)
    response_rate = signals.get('recruiter_response_rate', 0.5)
    engagement_multiplier = 0.80 + (0.20 * response_rate) 
    
    # 2. Honeypot check (Example: Expert skill with 0 months duration)
    is_honeypot = False
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            is_honeypot = True
            break
            
    return engagement_multiplier, is_honeypot

def main():
    print(f"Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Reading candidates from {CANDIDATES_FILE}...")
    with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
        candidates = json.load(f) 
        
    texts = []
    metadata = {}
    
    print("Processing candidate text and signals...")
    for idx, cand in enumerate(candidates):
        cid = cand['candidate_id']
        texts.append(extract_candidate_text(cand))
        
        eng_mult, is_trap = calculate_multipliers(cand)
        metadata[idx] = {
            'candidate_id': cid,
            'engagement_multiplier': eng_mult,
            'is_honeypot': is_trap,
            'years_exp': cand.get('profile', {}).get('years_of_experience', 0)
        }

    print("Generating Embeddings (This will take a moment)...")
    start_time = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(embeddings)
    
    print(f"Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings)
    
    print(f"Saving artifacts to disk...")
    faiss.write_index(index, OUTPUT_INDEX)
    with open(OUTPUT_METADATA, 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"Phase 1 Complete! Took {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()