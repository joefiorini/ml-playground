from typing import Optional
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, hf_hub_url
from tqdm.notebook import tqdm
import requests


def download_model_from_huggingface(
    model_identifier: str, update: bool = False, subfolder: Optional[str] = None
) -> None:
    """
    Downloads a model from Hugging Face and stores it in a local directory.

    :param model_identifier: The model identifier on Hugging Face in the format "username/repository".
    :param update: Whether to re-download the model files even if they already exist locally.
    :param subfolder: Optional subfolder under "models" where to store the model files.
    """
    # Create the models directory if it doesn't exist
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    # Parse the username and repository from the model identifier
    username, repository = model_identifier.split("/")

    # Create a directory for the model under the models directory
    model_dir = models_dir / f"{username}_{repository}"
    model_dir.mkdir(exist_ok=True)

    if subfolder is not None:
        model_dir = model_dir / subfolder
        model_dir.mkdir(exist_ok=True)

    # Get a list of all files in the model repository
    model_files = HfApi().list_repo_files(model_identifier)
    model_files = map(lambda x: Path(x), model_files)

    # Define the allowed extensions
    allowed_extensions = [".bin", ".safetensors", ".json"]

    # Download the model files into the model directory
    for file in model_files:
        file_name = file.name
        file_path = model_dir / file_name
        if file_path.suffix in allowed_extensions and (
            not file_path.exists() or update
        ):
            # Get the URL of the file
            file_url = hf_hub_url(repo_id=model_identifier, filename=file_name)

            # Get the response from the URL
            response = requests.get(file_url, stream=True)

            # Total size in bytes
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte

            t = tqdm(total=total_size, unit="iB", unit_scale=True)

            with open(file_path, "wb") as f:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    f.write(data)
            t.close()

            if total_size != 0 and t.n != total_size:
                print("ERROR, something went wrong")
