from pathlib import Path
from huggingface_hub import hf_hub_download

filename = (
    "machine_learning/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
)

path = hf_hub_download(
    repo_id="bvsam/cic-ids-2017",
    repo_type="dataset",
    filename=filename,
    local_dir="data",
)

print("External dataset downloaded successfully!")
print("Saved to:", path)