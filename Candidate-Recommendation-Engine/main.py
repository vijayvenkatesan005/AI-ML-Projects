# Vijay Venkatesan

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.llms import OpenAI
import streamlit as st
import os
import numpy as np
import pandas as pd
import tempfile

r_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=500,
    chunk_overlap=50
)

encoder = SentenceTransformer("all-mpnet-base-v2")

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = openai_api_key

llm = OpenAI(temperature=0.6)

def extract_document_text(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(pdf_file.read())
        tmpfile_path = tmpfile.name
     
    loader = PyPDFLoader(tmpfile_path)
    pages = loader.load_and_split()
    page_content_list = [doc.page_content for doc in pages]
    text = "\n".join(page_content_list)

    os.remove(tmpfile_path)

    return text

def chunk_text(text):
    chunks = r_splitter.split_text(text)
    return chunks

def create_embedding(sections):
    vectors = encoder.encode(sections)
    return vectors

def compute_similarity(resume_vector, job_description_vector):
    dot_product_vectors = np.dot(resume_vector, job_description_vector)
    product_vector_norms = np.linalg.norm(resume_vector) * np.linalg.norm(job_description_vector)
    cosine_similarity = dot_product_vectors / product_vector_norms
    return cosine_similarity

def generate_fit_explanation(job_desc, resume_text, llm):
    prompt = (
        f"Job Description:\n{job_desc}\n\n"
        f"Candidate Resume:\n{resume_text}\n\n"
        "Based on the job description and this resume, "
        "provide a concise 1 sentence explanation of why this candidate is a strong fit for the role."
    )

    explanation = llm(prompt)
    return explanation.strip()

def rank_candidates(similarity_scores, resume_texts, N):
    ranked_candidates = sorted(
        similarity_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_n = ranked_candidates[:N] if N < len(ranked_candidates) else ranked_candidates

    results = []
    for full_name, similarity_score in top_n:
        resume = resume_texts[full_name]
        explanation = generate_fit_explanation(job_description, resume, llm)
        results.append({
            "Candidate": full_name,
            "Similarity Score": similarity_score,
            "Explanation": explanation
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

st.title("AI Resume Screening & Fit Summary")

job_description = st.text_area("Paste the Job Description here", height=200)
uploaded_files = st.file_uploader(
    "Upload candidate resumes (PDF, named as firstName_lastName_resume.pdf)",
    type=["pdf"],
    accept_multiple_files=True
)

no_candidates_to_rank = st.number_input(
    "How many candidates to display?",
    min_value=1,
    max_value=100,
    value=5,
    step=1,
    format="%d"
) 

job_description_chunks = chunk_text(job_description)
job_description_matrix = create_embedding(job_description_chunks)
job_description_vector = np.mean(job_description_matrix, axis=0)

target_no_components = 3

all_valid = True
job_description_processed = False

if job_description and job_description.strip():
    job_description_processed = True
else:
    job_description_processed = False

if not job_description_processed:
    st.warning("Please enter a job description before proceeding.")

if job_description_processed:
    for pdf_file in uploaded_files:
        file_name = pdf_file.name
        base_name = os.path.splitext(file_name)[0]
        components = base_name.split("_")
        if len(components) == target_no_components:
            first_name, last_name = components[0].capitalize(), components[1].capitalize()
            st.write(f"Processing {first_name} {last_name} ({file_name})")
        else:
            st.warning(f"File '{file_name}' does not conform to 'firstName_lastName_resume.pdf' naming convention. Please rename and re-upload.")
            all_valid = False
            break

if all_valid and job_description_processed:
    resume_texts = {}
    resume_chunks = {}
    resume_vectors = {}
    similarity_scores = {}

    for pdf_file in uploaded_files:
        file_name = pdf_file.name
        base_name = os.path.splitext(file_name)[0]
        components = base_name.split("_")
        first_name, last_name = components[0].capitalize(), components[1].capitalize()
        full_name = f"{first_name} {last_name}"
        resume_text = extract_document_text(pdf_file)
        resume_texts[full_name] = resume_text
    
    for full_name, full_text in resume_texts.items():
        chunks = chunk_text(full_text)
        resume_chunks[full_name] = chunks
    
    for full_name, chunks in resume_chunks.items():
        resume_matrix = create_embedding(chunks)
        resume_vector = np.mean(resume_matrix, axis=0)
        resume_vectors[full_name] = resume_vector
        similarity_score = compute_similarity(
            resume_vector,
            job_description_vector
        )
        similarity_scores[full_name] = similarity_score
    
    rank_candidates(similarity_scores, resume_texts, no_candidates_to_rank)
    

    





    

    






