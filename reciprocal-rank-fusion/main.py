from pathlib import Path
import os
from dotenv import load_dotenv
from collections import defaultdict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

SYSTEM_PROMPT = """
You are a smart PDF assistant. Answer user queries using only the provided PDF excerpts.

- For summaries, give a brief overview of key points.
- For specific questions, extract and present relevant info directly.
- For explanations, start with a simple overview, then add detail if needed.
- If the info isn't in the excerpts, reply: "The PDF does not contain this information."

Be clear, concise, and avoid unnecessary jargon. Structure your answers to match the user's intent.
If the query is unclear, ask the user to clarify the question once again.
"""

def load_pdf_documents(pdf_file_path):
    loader = PyPDFLoader(file_path=pdf_file_path)
    return loader.load()

def split_into_chunks(documents, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def generate_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )

def store_chunks_in_qdrant(chunks, embedding_model):
    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url="http://localhost:6333",
        collection_name="pdf_chunks"
    )

def load_chat_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY
    )

def generate_query_variations(original_query, model, num_variations=3):
    prompt = f"Generate {num_variations} different ways to ask this question: {original_query}"
    response = model.invoke(prompt)
    variations = response.content.split("\n")
    return [original_query] + [v.strip() for v in variations if v.strip()]

def retrieve_parallel_with_rrf(vector_store, queries, k=3):
    docs_per_query = []
    for query in queries:
        docs = vector_store.similarity_search(query, k=k)
        docs_per_query.append(docs)
    return docs_per_query

def rank_the_queries(docs_per_query, k=60):
    scores = defaultdict(float)
    doc_map = {}
    source_info = defaultdict(list)

    for i, query_docs in enumerate(docs_per_query):
        for rank, doc in enumerate(query_docs, start=1):
            content = doc.page_content
            score = 1 / (k + rank)
            scores[content] += score
            doc_map[content] = doc
            source_info[content].append(f"Variation {i+1} (Rank {rank}, +{score:.4f})")

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    unique_docs = [doc_map[doc_text] for doc_text, _ in sorted_docs]
    return unique_docs

def chat_with_rrf(query, vector_store, model):
    queries = generate_query_variations(query, model)
    docs_per_query = retrieve_parallel_with_rrf(vector_store, queries)
    fused_docs = rank_the_queries(docs_per_query)

    context = "\n\n...\n\n".join([doc.page_content for doc in fused_docs[:5]])
    full_prompt = (
        SYSTEM_PROMPT +
        f"\n\nRelevant excerpts from the PDF:\n{context}\n\nUser's question: {query}\n\nAssistant:"
    )
    response = model.invoke(full_prompt)
    return response.content

def main():
    pdf_path = Path("reciprocal-rank-fusion") / "resume_analyst.pdf"
    documents = load_pdf_documents(pdf_path)
    chunks = split_into_chunks(documents)
    embeddings = generate_embeddings()
    vector_store = store_chunks_in_qdrant(chunks, embeddings)
    chat_model = load_chat_model()

    while True:
        query = input("Ask a question about the PDF (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            print("Please enter a valid question.")
            continue

        try:
            answer = chat_with_rrf(query, vector_store, chat_model)
            print("\nAnswer:\n", answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
