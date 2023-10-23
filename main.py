from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import base64

app = FastAPI()

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class StringInput(BaseModel):
    sentence1: str
    sentence2: str

class ListInput(BaseModel):
    sentences1: list
    sentences2: list

class MatrixInput(BaseModel):
    matrix: list

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
    
@app.post("/convert-matrix-to-image")
async def convert_matrix_to_image(data: MatrixInput):
    try:
        # Convert the matrix to a NumPy array
        matrix_data = data.matrix
        matrix = np.array(matrix_data)

        # Create a tabulated image
        fig_width = 10  # Set an initial figure width
        num_columns = len(matrix[0])
        cell_width = 1.0 / num_columns
        fig_height = 1.5

        # Calculate the required figure width based on the number of columns and cell size
        if num_columns * cell_width > fig_width:
            fig_width = num_columns * cell_width

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')

        table = ax.table(cellText=matrix, cellLoc='center', loc='center')
        table.scale(1, fig_height / 1.5)  # Adjust the height proportionally

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()


        

        # Encode the image to base64
        encoded_image = base64.b64encode(buf.read()).decode('utf-8')

        return {"tabulated_image": encoded_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during image conversion")
