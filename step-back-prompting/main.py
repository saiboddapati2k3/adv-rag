from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
import os

load_dotenv()


def load_pdf(file_name: str):
    pdf_path = Path(__file__).parent / "step-back-prompting" / file_name
    loader = PyPDFLoader(file_path=pdf_path)
    return loader.load()


def split_text(docs, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents=docs)


def create_embedder(api_key: str):
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )


def create_vector_store(docs, embedder):
    return QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedder,
        url="http://localhost:6333",
        collection_name="resume_analyst"
    )


def initialize_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key
    )


SYSTEM_PROMPT = """
You are a smart PDF assistant. You help users understand Resume content by reasoning broadly using Step-Back Prompting.

Guidelines:
1. First consider a broader view of the user's question.
2. Then answer the specific query based on the provided excerpts only.
3. If the answer is not found, say: "The PDF does not contain this information."
4. Use clear, simple language. Always show your reasoning process.
"""


def get_broader_question(llm, specific_query):
    prompt = f"Generate a broader question related to: {specific_query}"
    response = llm.invoke(prompt)
    return response.content.strip()


def retrieve_relevant_chunks(query, broader_query, vector_store, k=3):
    specific_chunks = vector_store.similarity_search(query, k=k)
    broad_chunks = vector_store.similarity_search(broader_query, k=k)

    all_chunks = specific_chunks + broad_chunks
    unique = []
    seen = set()
    for doc in all_chunks:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique


def build_step_back_prompt(original_query, broader_query, chunks_text):
    return (
        SYSTEM_PROMPT + "\n\n"
        "Based on these excerpts from the PDF, answer the question using Step-Back Prompting.\n\n"
        f"Step-Back Query: {broader_query}\n\n"
        f"Excerpts:\n{chunks_text}\n\n"
        f"Original Question: {original_query}\n\n"
        "Let's think step-by-step:\n"
        "1. Think about the broader topic (from the step-back query).\n"
        "2. Connect it to the specific question.\n"
        "3. Provide a clear and concise answer.\n\n"
        "So, the answer is:"
    )


def answer_with_step_back(query, vector_store, llm):
    print("\n[1] Generating broader query...")
    broader_query = get_broader_question(llm, query)
    print("→ Broader query generated:", broader_query)

    print("\n[2] Retrieving relevant document chunks...")
    retrieved_docs = retrieve_relevant_chunks(query, broader_query, vector_store)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"→ Retrieved {len(retrieved_docs)} unique chunks.")

    print("\n[3] Constructing step-back prompt...")
    full_prompt = build_step_back_prompt(query, broader_query, context_text)


    print("\n[4] Sending prompt to Gemini LLM...")
    response = llm.invoke(full_prompt)

    return response.content


def run_pdf_assistant():
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Check your .env file.")

    print("Loading and indexing Resume PDF...")
    docs = load_pdf("resume_analyst.pdf")
    chunks = split_text(docs)
    embedder = create_embedder(GOOGLE_API_KEY)
    vector_store = create_vector_store(chunks, embedder)
    llm = initialize_llm(GOOGLE_API_KEY)

    print("PDF loaded and system initialized!")
    print("\nWelcome to the Resume Analyzer with Step-Back Prompting!")
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        try:
            answer = answer_with_step_back(query, vector_store, llm)
            print("\ Assistant:\n", answer)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    run_pdf_assistant()