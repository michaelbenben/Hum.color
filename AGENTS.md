# AGENTS.md

## Run the app

```bash
source .venv/bin/activate && streamlit run app.py
```

App is served at http://localhost:8501. Light mode and headless mode are set in `.streamlit/config.toml` (no CLI flags needed).

## Stack

- Python 3.11.5 (pinned in `.python-version`)
- Streamlit (UI) + OpenCV (`opencv-python-headless`) + PyAV (`av`) + NumPy <2.0.0
- System packages required (see `packages.txt`): `libgl1`, `ffmpeg`. Missing `libgl1` breaks `cv2` import on headless Linux.
- Single-file app: all logic lives in `app.py`. No modules, no tests, no linter config.

## Architecture notes

- `app.py` contains both the image-processing pipeline and the Streamlit UI. Entry point is `main()`.
- Pipeline: upload → detect red (HSV mask + red-dominance filter) → recolor to violet (`VIOLET_HUE = 145`) → optional social canvas (blurred background + contained foreground) → encode.
- Videos are processed frame-by-frame via PyAV in `process_video`. Input is opened once for both metadata and frame decode (do not re-open).
- `EXPORT_PRESETS` defines output dimensions per social platform; the UI mutates `st.session_state.export_key` and reruns.

## Conventions

- UI strings and comments are in French.
- `requirements.txt` pins `numpy<2.0.0` — do not upgrade past 2.x without testing `cv2`/`av` compatibility.
- Uploaded files are written to a temp path; always clean up via `remove_temp_file` in `finally`.
