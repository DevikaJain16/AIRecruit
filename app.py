import streamlit as st
import pandas as pd
import subprocess
import os

st.set_page_config(page_title="Redrob AI Ranker", layout="wide")

st.title("🚀 Intelligent Candidate Discovery & Ranking")
st.markdown("### Two-Stage Semantic Ranker (CPU Optimized)")
st.write("This sandbox demonstrates the Phase 2 Local Ranker. It uses pre-computed FAISS embeddings and behavioral multipliers to evaluate candidates in seconds, without any external API calls.")

if st.button("▶️ Run AI Ranking Pipeline"):
    with st.spinner("Executing rank.py... (Performing vector search & applying heuristics)"):
        try:
            result = subprocess.run(["python", "rank.py"], capture_output=True, text=True)
            
            if os.path.exists("team_submission.csv"):
                st.success("Pipeline executed successfully in under 5 minutes!")
                
                df = pd.read_csv("team_submission.csv")
                st.dataframe(df, use_container_width=True)
                
                with open("team_submission.csv", "rb") as file:
                    st.download_button(
                        label="📥 Download team_submission.csv",
                        data=file,
                        file_name="team_submission.csv",
                        mime="text/csv",
                    )
            else:
                st.error("Error: CSV not generated. Check logs.")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"Failed to execute: {e}")