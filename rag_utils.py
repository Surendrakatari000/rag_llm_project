import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from a .env file
load_dotenv()

# Paths
VECTORSTORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTORSTORE_DIR, "faiss_index.pkl")
EMBEDDINGS_FILE = os.path.join(VECTORSTORE_DIR, "embeddings.pkl")

# --- Ensure the vectorstore directory exists ---
if not os.path.exists(VECTORSTORE_DIR):
    print(f"Error: The '{VECTORSTORE_DIR}' directory does not exist.")
    print("Please run the script to create the embeddings and FAISS index first.")
    exit()

# Load embeddings and FAISS index
def load_vectorstore():
    """Loads the FAISS index and corresponding text chunks from disk."""
    try:
        with open(INDEX_FILE, "rb") as f:
            index = pickle.load(f)
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
            texts = data["texts"]
        return index, texts
    except FileNotFoundError as e:
        print(f"Error loading vectorstore files: {e}")
        print("Please ensure both 'faiss_index.pkl' and 'embeddings.pkl' exist in the 'vectorstore' directory.")
        exit()

# Encode query using the same embedding model
def embed_query(query: str):
    """Encodes the user's query into a vector."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([query])

# Retrieve top-k relevant chunks
def retrieve_chunks(query: str, top_k: int = 3):
    """Retrieves the top-k most relevant text chunks for a given query."""
    index, texts = load_vectorstore()
    query_vector = embed_query(query)
    distances, indices = index.search(query_vector, top_k)
    # Filter out invalid indices (-1) that FAISS might return
    retrieved = [texts[i] for i in indices[0] if i != -1 and i < len(texts)]
    return retrieved

# Generate answer using Hugging Face Inference Client
def generate_answer(query: str):
    """Generates an answer by calling the Hugging Face Inference API via the client."""
    context_chunks = retrieve_chunks(query)
    context = "\n".join(context_chunks)

    # --- Hugging Face Inference Client Configuration ---
    model_repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(model=model_repo_id, token=os.getenv("hugging_face"))

    print("Querying the Inference Client...")
    try:
        # Use chat.completions API
        response = client.chat.completions.create(
            model=model_repo_id,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ],
            max_tokens=200,
            temperature=0.7,
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        print(f"Inference Client request failed: {e}")
        return "Sorry, there was an error communicating with the Inference Client."

# --- Main execution block ---
if __name__ == "__main__":
    user_query = "what is AI"
    final_answer = generate_answer(user_query)
    print("\n---")
    print(f"Question: {user_query}")
    print(f"Answer: {final_answer}")
