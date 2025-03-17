import json

from langchain_ollama.embeddings import OllamaEmbeddings
import numpy as np

def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    return dot_product / (magnitude_A * magnitude_B)

if __name__ == "__main__":
    ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    with open("data/job-1.json", "r") as f:
        job1 = f.read()
    with open("data/job-2.json", "r") as f:
        job2 = f.read()
    prompt = "Cloud Architect"
    job_embeddings = ollama_embeddings.embed_documents([job1, job2])
    prompt_embeddings = ollama_embeddings.embed_query(prompt)
    print(f"""
    Cosine similarity to job1: {cosine_similarity(prompt_embeddings, job_embeddings[0])}
    Cosine similarity to job2: {cosine_similarity(prompt_embeddings, job_embeddings[1])}
    """)

    job1_json = json.loads(job1)
    job2_json = json.loads(job2)
    job_embeddings = ollama_embeddings.embed_documents([job1_json["title"], job2_json["title"]])
    print(f"""
    Cosine similarity to job1 title: {cosine_similarity(prompt_embeddings, job_embeddings[0])}
    Cosine similarity to job2 title: {cosine_similarity(prompt_embeddings, job_embeddings[1])}
    """)
