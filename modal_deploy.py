import modal

app = modal.App("cited-rx-rag")


def download_models():
    import gradio
    print(f"Gradio version in container: {gradio.__version__}")
    from sentence_transformers import SentenceTransformer, CrossEncoder
    print("Downloading models into container image...")
    SentenceTransformer("BAAI/bge-m3") 
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
    print("Downloads complete!")

# 1. Define the cloud environment
# 1. Define the cloud environment
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "gradio==5.23.0",
        "fastapi", "uvicorn", "qdrant-client", 
        "sentence-transformers", "anthropic", "httpx", 
        "pypdf", "python-multipart", "rank-bm25","python-dotenv","langchain-text-splitters"
    )
    .run_function(download_models)
    .add_local_dir("./backend", remote_path="/root/backend")
    .add_local_dir("./data", remote_path="/root/data")
    
)

# 2. Configure the server hardware
@app.function(
    image=image,
    secrets=[modal.Secret.from_dotenv()],
    timeout=1800,       # Allow up to 30 mins for heavy PDF uploads
    memory=4096,
    min_containers=1,
    max_containers=3,       
)

@modal.concurrent(max_inputs=1000)

@modal.asgi_app()
def serve():
    # Tell the cloud server where to look for your files
    import sys
    sys.path.append("/root")
    
    # Import your merged FastAPI + Gradio app
    from backend.api import app as fastapi_app
    return fastapi_app