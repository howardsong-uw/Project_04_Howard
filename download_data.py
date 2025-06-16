from kaggle.api.kaggle_api_extended import KaggleApi
import os, zipfile

def download_spaceship_titanic(dest="data"):
    api = KaggleApi()
    api.authenticate()

    os.makedirs(dest, exist_ok=True)
    api.competition_download_files(
        competition="spaceship-titanic",
        path=dest,
        quiet=False
    )

    zip_path = os.path.join(dest, "spaceship-titanic.zip")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest)
    os.remove(zip_path)

if __name__ == "__main__":
    download_spaceship_titanic()
    print("The data has been downloaded and extracted to the data/ directory.")
