import tarfile
import os

def make_tarfile(output_filename, source_files):
    with tarfile.open(output_filename, "w:gz") as tar:
        for f in source_files:
            tar.add(f, arcname=os.path.basename(f))

if __name__ == "__main__":
    files = [
        "aiml_model.pkl",
        "sample_submission.csv",
        "inference.py"
    ]
    make_tarfile("model.tar.gz", files)
