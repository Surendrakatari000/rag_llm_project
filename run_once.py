import os 
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import pickle

DOCS_DIR = "docs"
VECTORESTORE_DIR = "vectorstore"
INDEX_FILE = os.path.join(VECTORESTORE_DIR, "faiss_index.pkl")
EMBEDDINGS_FILE = os.path.join(VECTORESTORE_DIR,"embeddings.pkl")

os.makedirs(VECTORESTORE_DIR,exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_pdfs(doc_folder) :
    texts = []
    for file in os.listdir(doc_folder) :
        if file.endswith(".pdf") :
            pdf_path = os.path.join(doc_folder,file) 
            render = PdfReader(pdf_path)
            for page in render.pages :
                text = page.extract_text()
                if text :
                    for chunk in text.split(". ") :
                        if chunk.strip()  :
                            texts.append(chunk.strip())
    return texts
    
documents = load_pdfs(DOCS_DIR) 

embeddings = model.encode(documents,convert_to_numpy=True)

dimensions = embeddings.shape[1] 

index = faiss.IndexFlatL2(dimensions)

index.add(embeddings) 


with open(INDEX_FILE,"wb") as f :
    pickle.dump(index,f) 
    print("done")

with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump({"texts": documents}, f) 
    print("both")
