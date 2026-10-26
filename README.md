# OralScan: Oral Cancer Image Classification

A Flask web application for classifying uploaded oral images with a fine-tuned Swin Transformer.

> **Research use only.** This project is not a medical device and must not be used to diagnose, treat, or make decisions about a person's health.

## Image
<img width="1907" height="817" alt="Screenshot 2026-10-26 141257" src="https://github.com/user-attachments/assets/cad422f9-f927-47ed-8e54-f43c35de2b8f" />

## Requirements

- Python 3.10 or newer
- A fine-tuned checkpoint named `best_swin_model.pth`
- Internet access on first model load to download the base Swin model from Hugging Face

The checkpoint is not stored in this repository. Obtain it from the project owner and place it at `model/best_swin_model.pth`. The app also requires the processor configuration files in `model/processor/`, which are included.

## Run locally
<img width="1872" height="911" alt="Screenshot 2026-10-26 141324" src="https://github.com/user-attachments/assets/9561bac8-c240-43a4-ae35-90be7c38137e" />
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:ORALSCAN_SECRET_KEY = "replace-with-a-random-secret"
python app.py
```

Open the local address printed by Flask. For public deployment, set a strong `ORALSCAN_SECRET_KEY`, disable debug mode, and use a production WSGI server.
## Image
<img width="1892" height="893" alt="Screenshot 2026-10-26 141355" src="https://github.com/user-attachments/assets/091e237c-83d2-4369-8afc-4f70c713c0fc" />

## Repository contents

- `app.py`: Flask routes, SQLite setup, model loading, and inference
- `templates/` and `static/`: web interface
- `model/processor/`: image processor configuration
- `requirements.txt`: Python dependencies

The local SQLite database, uploaded images, and model weight files are excluded from Git. The app creates `oralscan.db` and `user_uploads/` as needed. The fine-tuned checkpoint must be obtained separately.

## Image
<img width="1891" height="897" alt="Screenshot 2026-10-26 141411" src="https://github.com/user-attachments/assets/b39ec619-d65e-4272-933f-bd8b1ff26dc7" />

