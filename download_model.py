import os
import gdown

URL = "https://drive.google.com/uc?id=1O7aWq96k-5COWldEUDVwdRn3UVLIxKrF"
OUTPUT = "model.h5"  # Rename according to your model type

def download_model():
    if os.path.exists(OUTPUT):
        print(f"{OUTPUT} already exists. Skipping download.")
        return

    print(f"Downloading model from Google Drive...")
    gdown.download(URL, OUTPUT, quiet=False)
    print("Download completed!")

if __name__ == "__main__":
    download_model()
