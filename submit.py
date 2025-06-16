import os
import subprocess
import datetime

def submit_to_kaggle(file_path="submission_ensemble.csv", competition="spaceship-titanic", message=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"no such file or directory: {file_path}")

    if message is None:
        message = f"Auto submission at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    print(f" submitting {file_path} to Kaggle...")
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c", competition,
        "-f", file_path,
        "-m", message
    ]
    try:
        subprocess.run(cmd, check=True)
        print(" submission successful!")
    except subprocess.CalledProcessError as e:
        print("Submission failed:", e)

if __name__ == "__main__":
    submit_to_kaggle("submission_ensemble.csv")
