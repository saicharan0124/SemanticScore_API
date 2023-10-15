from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class StringInput(BaseModel):
    sentence1: str
    sentence2: str

class ListInput(BaseModel):
    sentences1: list
    sentences2: list

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You should restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/calculate-cosine-similarity-tabulated")
async def calculate_cosine_similarity_tabulated(data: ListInput):
    try:
        # Encode the sentences
        embeddings1 = model.encode(data.sentences1)
        embeddings2 = model.encode(data.sentences2)

        # Calculate cosine similarity scores and organize them in a structured format
        similarity_table = []
        for sentence1 in embeddings1:
            row_scores = [util.pytorch_cos_sim(sentence1, sentence2).item() for sentence2 in embeddings2]
            similarity_table.append(row_scores)

        return {"cosine_similarity_table": similarity_table}
    except Exception as e:
        # Handle any exceptions that might occur during calculation
        raise HTTPException(status_code=500, detail="An error occurred during calculation")

@app.post("/calculate-cosine-similarity")
async def calculate_cosine_similarity(data: StringInput):
    try:
        # Encode the sentences
        my_embedding = model.encode(data.sentence1)
        embeddings = model.encode(data.sentence2)

        # Calculate cosine similarity
        cos_sim = util.pytorch_cos_sim(my_embedding, embeddings).item()

        return {"cosine_similarity": cos_sim}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during calculation")
