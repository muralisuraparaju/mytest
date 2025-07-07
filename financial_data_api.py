import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import your existing modules
import datagenerator
import embeddings_generator
import embedding_wordcloud
import data_product_recommender

app = FastAPI(title="Financial Data API", 
              description="API for financial data embeddings and recommendations")

# Create directories if they don't exist
os.makedirs("wordclouds", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static directory to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")


class UserPrompt(BaseModel):
    """Model for user prompt requests"""
    query: str
    top_k: int = 5


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Financial Data API",
        "endpoints": {
            "/generate-data": "Generate sample financial data",
            "/generate-embeddings": "Generate embeddings for financial data",
            "/generate-wordclouds": "Create word clouds from embeddings",
            "/wordclouds": "List all available word cloud images",
            "/wordcloud/{image_name}": "Get a specific word cloud image",
            "/recommend": "Get recommendations based on user prompt"
        }
    }


@app.post("/generate-data")
async def generate_data():
    """Generate sample financial data using datagenerator module"""
    try:
        # Call the main function of your datagenerator module
        datagenerator.main()
        return {"status": "success", "message": "Financial data generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")


@app.post("/generate-embeddings")
async def generate_embeddings(
    max_features: int = Query(1000, description="Maximum number of TF-IDF features"),
    n_components: int = Query(100, description="Number of components for dimensionality reduction")
):
    """Generate embeddings for financial data"""
    try:
        # Create an instance of the embeddings generator and generate embeddings
        embeddings_generator.main()
        return {"status": "success", "message": "Embeddings generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embeddings generation failed: {str(e)}")


@app.post("/generate-wordclouds")
async def generate_wordclouds(
    max_features: int = Query(1000, description="Maximum number of TF-IDF features"),
    n_components: int = Query(100, description="Number of components for dimensionality reduction"),
    n_clusters: int = Query(0, description="Number of clusters for clustering (0 for auto)"),
    min_df: int = Query(2, description="Minimum document frequency for TF-IDF")
):
    """Generate word clouds from embeddings"""
    try:
        # Call the main function of your embedding_wordcloud module
        embedding_wordcloud.main()
        
        # List all generated word cloud files
        wordcloud_files = []
        for file in os.listdir("wordclouds"):
            if file.endswith(".png"):
                # Copy to static dir for serving
                src_path = os.path.join("wordclouds", file)
                dst_path = os.path.join("static", file)
                with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
                    dst.write(src.read())
                wordcloud_files.append(file)
                
        return {
            "status": "success", 
            "message": f"Word clouds generated successfully. {len(wordcloud_files)} files created.",
            "files": wordcloud_files,
            "urls": [f"/wordcloud/{file}" for file in wordcloud_files]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.get("/wordclouds")
async def list_wordclouds():
    """List all available word cloud images"""
    try:
        wordcloud_files = []
        for file in os.listdir("wordclouds"):
            if file.endswith(".png"):
                wordcloud_files.append({
                    "filename": file,
                    "url": f"/wordcloud/{file}"
                })
        return {"wordclouds": wordcloud_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list word clouds: {str(e)}")


@app.get("/wordcloud/{image_name}")
async def get_wordcloud(image_name: str):
    """Get a specific word cloud image"""
    image_path = os.path.join("wordclouds", image_name)
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail=f"Image {image_name} not found")
    
    return FileResponse(image_path, media_type="image/png")


@app.post("/recommend")
async def get_recommendations(prompt: UserPrompt):
    """Get recommendations based on user prompt"""
    try:
        # Create an instance of the recommender
        recommender = data_product_recommender.DataProductRecommender()
        
        # Get recommendations based on user query
        recommendations = recommender.recommend_products(prompt.query, top_k=prompt.top_k)
        
        # Extract intent analysis
        intent = recommender.analyze_query_intent(prompt.query)
        
        return {
            "query": prompt.query,
            "intent": intent,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
