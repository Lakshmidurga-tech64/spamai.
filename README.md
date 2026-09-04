# NexGen — AI Voice Security (Version 1)

## Project Overview

NexGen is a simple web app that lets you upload a voice recording and get
a prediction of whether it sounds **Genuine** or **Suspicious/Synthetic**
(possibly AI-generated or cloned). It shows a confidence percentage, a
0–100 risk score, and a risk level (LOW / MEDIUM / HIGH), along with a
plain-language explanation and a security recommendation.

This is **Version 1**: a local, end-to-end demo. There is no login, no
database, no phone-network integration, and no blockchain — just
upload → analyze → result.

## Features (Version 1)

- Drag-and-drop or click-to-upload audio (WAV, MP3, M4A)
- FastAPI backend that extracts audio features and runs an ML model
- Genuine / Suspicious prediction with a confidence percentage
- 0–100 risk score and LOW/MEDIUM/HIGH risk level
- Clear security guidance based on the risk level
- **Demo mode**: works immediately even without a trained model, and
  always tells you clearly when it's in demo mode
- Real ML mode: trains on your own dataset with `training/train_model.py`

## Project Structure

```
voice-security/
│
├── backend/
│   ├── main.py              FastAPI app (the server, endpoints)
│   ├── model.py              Loads the model / runs demo mode
│   ├── audio_processor.py    Turns audio into features
│   ├── risk.py                Turns probability into risk score/level
│   ├── requirements.txt
│   └── models/
│       └── voice_model.pkl   (created after you train a real model)
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── dataset/
│   ├── real/         put real human voice files here
│   └── synthetic/    put synthetic/cloned voice files here
│
├── training/
│   └── train_model.py   run this to train the real ML model
│
└── README.md
```

## Requirements

- Python 3.9 or newer
- pip
- A modern web browser
- (Optional, for training) some real and synthetic voice audio files

## Installation

Open a terminal in the `voice-security` folder.

**1. Create a virtual environment:**

```bash
python -m venv venv
```

**2. Activate it:**

Windows:
```bash
venv\Scripts\activate
```

Linux / Mac:
```bash
source venv/bin/activate
```

**3. Install backend dependencies:**

```bash
pip install -r backend/requirements.txt
```

## Running the Application

**Step 1 — Start the backend (FastAPI + Uvicorn):**

From the `voice-security` folder:

```bash
cd backend
uvicorn main:app --reload --port 8000
```

You should see a message saying Uvicorn is running on
`http://127.0.0.1:8000`. Leave this terminal window open.

**Step 2 — Open the frontend:**

In a file explorer, go to the `frontend` folder and double-click
`index.html` to open it in your browser. (Or right-click → Open With →
your browser.)

That's it — upload an audio file and click **Analyze Voice**.

> The frontend calls `http://127.0.0.1:8000` by default. If you change
> the backend port, update `API_BASE_URL` at the top of
> `frontend/script.js` to match.

## Training the Real ML Model

Right now the app runs in **Demo Mode** (see below) because there's no
trained model yet. To train a real one:

1. Add real human voice recordings to `dataset/real/`
2. Add synthetic/cloned voice recordings to `dataset/synthetic/`
   (`.wav`, `.mp3`, or `.m4a` files — a few dozen of each is a
   reasonable starting point for a hackathon demo)
3. From the `voice-security` folder, run:

```bash
python training/train_model.py
```

This prints accuracy, precision, recall, and F1 score, then saves the
trained model to `backend/models/voice_model.pkl`.

4. Open `backend/model.py` and change:

```python
DEMO_MODE = True
```

to:

```python
DEMO_MODE = False
```

5. Restart the backend server. It will now use your trained model.

## Demo Mode

`DEMO_MODE` lives in `backend/model.py`. When it's `True` — or when no
trained model file exists yet at `backend/models/voice_model.pkl` — the
app returns a randomized but clearly-labeled placeholder prediction.
This lets you test the full upload → analyze → result flow before you
have a real trained model. Every response in demo mode includes
`"demo_mode": true`, and the frontend displays a visible banner:

> "Demo mode — prediction is simulated for interface testing. Not a
> validated AI detection result."

Once you train a real model and set `DEMO_MODE = False`, this banner
disappears and predictions come from the actual trained pipeline.

## Troubleshooting

**"Backend unavailable" error in the browser**
The FastAPI server isn't running, or is running on a different port
than the frontend expects. Make sure `uvicorn main:app --reload --port
8000` is running, and that `API_BASE_URL` in `frontend/script.js`
matches.

**CORS errors in the browser console**
The backend already allows all origins for this MVP. If you still see
CORS errors, double check the backend is actually running and that you
didn't change the CORS settings in `backend/main.py`.

**`ModuleNotFoundError` when starting the backend**
Make sure your virtual environment is activated and you ran `pip
install -r backend/requirements.txt` from inside it.

**Training fails with "You need at least some files in BOTH
dataset/real/ and dataset/synthetic/"**
Add at least a few audio files to both folders before running
`train_model.py`.

**Uploaded file fails with "Unable to analyze this audio"**
The file may be corrupted or in an unsupported format. Try a plain
WAV or MP3 file. Check the backend terminal window for the detailed
error (it's printed there, but never shown to the user).

**`librosa` / audio loading errors on some MP3 files**
Some MP3 encodings can be picky. If a specific file fails, try
re-exporting it as WAV and uploading that instead.
