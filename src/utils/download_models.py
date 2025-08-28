import os
from huggingface_hub import snapshot_download


def download_models():
    print("Checking download progress...")

    # Check if models are already cached
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"HF cache location: {cache_dir}")

    # Try downloading with progress tracking
    print("Downloading Qwen3-4B...")
    try:
        snapshot_download("Qwen/Qwen3-4B", cache_dir=cache_dir)
        print("Qwen3-4B download complete")
    except Exception as e:
        print(f"Download error: {e}")

    print("Downloading your model...")
    try:
        snapshot_download("PetarKal/qwen3-4b-EM-finetuned", cache_dir=cache_dir)
        print("Your model download complete")
    except Exception as e:
        print(f"Download error: {e}")


if __name__ == "__main__":
    download_models()