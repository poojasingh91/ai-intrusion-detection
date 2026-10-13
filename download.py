from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="bvsam/cic-ids-2017",
    repo_type="dataset",
    filename="machine_learning/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet",
    local_dir="data",
)

print("Downloaded:", path)