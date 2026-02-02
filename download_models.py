import os
from huggingface_hub import hf_hub_download, snapshot_download

def download_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading Mistral-7B-Instruct-v0.2-GGUF...")
    repo_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Mistral model downloaded to: {model_path}")
    except Exception as e:
        print(f"Error downloading Mistral: {e}")
    
    print("Downloading Embedding model (all-MiniLM-L6-v2)...")
    try:
        # Download the entire model repository for the embedding model
        embedding_model_path = snapshot_download(
            repo_id='sentence-transformers/all-MiniLM-L6-v2',
            local_dir=os.path.join(models_dir, "embedding_model"),
            local_dir_use_symlinks=False
        )
        print(f"Embedding model downloaded to: {embedding_model_path}")
    except Exception as e:
        print(f"Error downloading Embedding model: {e}")

if __name__ == "__main__":
    download_models()
