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

        # Calculate cosine similarity scores
        similarity_table = []

        # Add the first row with the specified headers
        first_row = [""] + ["PO1", "PO2", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "PO9", "PO10", "PO11", "PO12", "PSO1", "PSO2", "PSO3"]
        similarity_table.append(first_row)

        # Iterate through the sentences and calculate similarity scores
        for i, sentence1 in enumerate(embeddings1):
            row = ["S" + str(i + 1)]  # Start the row with "S1", "S2", ...

            for sentence2 in embeddings2:
                score = util.pytorch_cos_sim(sentence1, sentence2).item()
                formatted_score = f"{score:.3f}"
                row.append(formatted_score)

            # Add the row to the similarity table
            similarity_table.append(row)

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
