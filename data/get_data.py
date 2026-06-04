import os
from tdc.single_pred import Tox

def download_and_split():
    os.makedirs("data", exist_ok=True)
    print("[INFO] Mengunduh dataset ClinTox dari Therapeutics Data Commons...")
    data_loader = Tox(name='ClinTox')
    split = data_loader.get_split(method='scaffold')
    
    # Simpan hasil pemisahan data ke dalam sub-folder data/
    split['train'].to_csv("data/train.csv", index=False)
    split['valid'].to_csv("data/valid.csv", index=False)
    split['test'].to_csv("data/test.csv", index=False)
    print("[SUCCESS] Dataset ClinTox berhasil disimpan di folder data/ !")

if __name__ == "__main__":
    download_and_split()
