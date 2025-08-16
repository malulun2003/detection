from huggingface_hub import snapshot_download

download_path = snapshot_download(repo_id="IDEA-Research/grounding-dino-base")
# print(f"Model downloaded to: {download_path}")
print(download_path)
