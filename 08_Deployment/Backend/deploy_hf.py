from huggingface_hub import HfApi
import sys

# Get credentials
token = input("Paste your HF Token (hf_...): ").strip()
repo_id = input("Paste your HF Space Repo ID (e.g. your_username/BirdSense-Backend): ").strip()

print(f"\nUploading the 08_Deployment/Backend folder to {repo_id}...")

try:
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space"
    )
    print("\n✅ SUCCESS! Backend deployed to Hugging Face!")
except Exception as e:
    print(f"\n❌ ERROR: {e}")
