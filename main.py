from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

class InputData(BaseModel):
    sentence1: str
    sentence2: str

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You should restrict this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/calculate-cosine-similarity")
async def calculate_cosine_similarity(data: InputData):
    try:
        # Encode the sentences
        my_embedding = model.encode(data.sentence1)
        embeddings = model.encode(data.sentence2)

        # Calculate cosine similarity
        cos_sim = util.cos_sim(my_embedding, embeddings)

        # Convert cos_sim to a float for JSON serialization
        return {"cosine_similarity": float(cos_sim.numpy())}
    except Exception as e:
        # Handle any exceptions that might occur during calculation
        raise HTTPException(status_code=500, detail="An error occurred during calculation")
