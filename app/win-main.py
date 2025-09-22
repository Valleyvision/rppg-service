# app/main.py
import os, json, uuid, logging, time, platform, subprocess, shlex
from datetime import datetime, timezone
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional

import numpy as np, cv2, heartpy as hp
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.stats import trim_mean
import httpx

# -----------------------------
# Configuration
# -----------------------------
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FS = float(os.getenv("FS", "30"))
DEFAULT_SEC = float(os.getenv("BUFFER_SEC", "10"))
LOWCUT = float(os.getenv("LOWCUT", "0.7"))
HIGHCUT = float(os.getenv("HIGHCUT", "4.0"))
MIN_BPM = float(os.getenv("MIN_BPM", "40"))
MAX_BPM = float(os.getenv("MAX_BPM", "180"))
SDNN_HIGH = float(os.getenv("SDNN_HIGH", "50"))
SDNN_MOD = float(os.getenv("SDNN_MOD", "30"))
SBP_SLOPE = float(os.getenv("SBP_SLOPE", "0.3"))
SBP_INTERCEPT = float(os.getenv("SBP_INTERCEPT", "100"))
DBP_SLOPE = float(os.getenv("DBP_SLOPE", "0.2"))
DBP_INTERCEPT = float(os.getenv("DBP_INTERCEPT", "60"))
RECORD_DIR = os.getenv("RECORD_DIR", "/app/recordings")
PORT = int(os.getenv("PORT", "8000"))

# LLM (Ollama) settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

os.makedirs(RECORD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("rppg")

app = FastAPI(title="rPPG Robust Service", version="1.4.0")

# -----------------------------
# Models
# -----------------------------
class Reading(BaseModel):
    timestamp: str
    heart_rate: float = Field(..., description="BPM")
    sdnn: float
    rmssd: float
    pnn50: float
    lf: float
    hf: float
    lf_hf: float
    spo2: float
    systolic: float
    diastolic: float
    luminosity: float
    stress_level: str
    perfusion_index: float
    hb_estimate_g_dl: float
    hb_confidence: float
    hb_method: str

class Aggregate(BaseModel):
    heart_rate: float
    sdnn: float
    rmssd: float
    pnn50: float
    lf: float
    hf: float
    lf_hf: float
    spo2: float
    systolic: float
    diastolic: float
    luminosity: float
    perfusion_index: float
    hb_estimate_g_dl: float
    stress_level: str

class MetricsResponse(BaseModel):
    request_id: str
    saved_path: str
    readings: List[Reading]
    aggregate: Aggregate

class LLMQuestionsIn(BaseModel):
    path: str = Field(..., description="Path to metrics JSON (e.g., recordings/metrics_<id>.json)")
    model: Optional[str] = Field(None, description="Override model name")
    temperature: Optional[float] = Field(None, ge=0, le=1)

class LLMQuestionsOut(BaseModel):
    request_id: str
    source_file: str
    used_model: str
    questions: List[str]

# -----------------------------
# Signal helpers
# -----------------------------
_tc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def butter_bp(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return b, a

_b_bp, _a_bp = butter_bp(LOWCUT, HIGHCUT, FS)

def bandpass(sig: np.ndarray) -> np.ndarray:
    if len(sig) < 5:
        return sig
    return filtfilt(_b_bp, _a_bp, sig)

def get_roi(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _tc.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return slice(y, y + max(1, h // 6)), slice(x + w // 4, x + 3 * w // 4)
    h, w = gray.shape
    return slice(h // 4, h // 4 + h // 8), slice(w // 2 - w // 8, w // 2 + w // 8)

def classify_stress(sdnn: float) -> str:
    if sdnn > SDNN_HIGH: return "Low Stress"
    if sdnn >= SDNN_MOD: return "Moderate Stress"
    return "High Stress"

def estimate_bp(hr: float) -> Tuple[float, float]:
    sys = SBP_INTERCEPT + SBP_SLOPE * (hr - 60.0)
    dia = DBP_INTERCEPT + DBP_SLOPE * (hr - 60.0)
    return float(np.clip(sys, 80, 200)), float(np.clip(dia, 40, 120))

def compute_spo2(r: np.ndarray, g: np.ndarray) -> float:
    ac_r, dc_r = r.std(), r.mean()
    ac_g, dc_g = g.std(), g.mean()
    if dc_r <= 0 or dc_g <= 0: return 0.0
    ratio = (ac_r / max(1e-6, dc_r)) / (ac_g / max(1e-6, dc_g))
    return float(np.clip(100 - 5 * ratio, 0, 100))

def perfusion_index(g: np.ndarray) -> float:
    dc = float(np.mean(g)); ac = float(np.std(g))
    if dc <= 0: return 0.0
    return float(np.clip(ac / dc, 0, 1))

def hb_from_ppg_proxy(r: np.ndarray, g: np.ndarray) -> Tuple[float, float, str]:
    pi = perfusion_index(g)
    ac_r, dc_r = r.std(), r.mean()
    ac_g, dc_g = g.std(), g.mean()
    ror = (ac_r / max(1e-6, dc_r)) / (ac_g / max(1e-6, dc_g))
    hb = 14.0 + (ror - 1.0) * 1.5 + (pi - 0.02) * 5.0
    hb = float(np.clip(hb, 8.0, 20.0))
    conf = float(np.clip(pi * 2.0, 0.0, 0.6))
    return hb, conf, "ppg_proxy_linear"

def hrv_features_from_rr(rr_ms: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    if rr_ms.size < 3: return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    sdnn = float(np.std(rr_ms, ddof=1))
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(np.square(diff)))) if diff.size else 0.0
    pnn50 = float(100.0 * np.mean(np.abs(diff) > 50.0)) if diff.size else 0.0
    try:
        _, m = hp.frequency_domain(rr_ms, mode='models')
        lf = float(m.get('lf', 0.0)); hf = float(m.get('hf', 0.0))
        lf_hf = float(m.get('lf/hf', 0.0)) if m.get('hf', 0.0) else 0.0
    except Exception:
        lf = hf = lf_hf = 0.0
    return sdnn, rmssd, pnn50, lf, hf, lf_hf

def hr_estimators(sig: np.ndarray) -> float:
    sig = detrend(sig); sig = bandpass(sig)
    freqs = np.fft.rfftfreq(len(sig), 1 / FS)
    fftm = np.abs(np.fft.rfft(sig))
    mask = (freqs >= LOWCUT) & (freqs <= HIGHCUT)
    bpm_f = 0.0
    if mask.any():
        peak = freqs[mask][np.argmax(fftm[mask])]
        bpm_f = peak * 60.0
    peaks, _ = find_peaks(sig, distance=FS * 0.5)
    bpm_t = 0.0
    if len(peaks) >= 2:
        bpm_t = 60.0 / np.diff(peaks).mean() * FS
    candidates = [b for b in [bpm_f, bpm_t] if MIN_BPM <= b <= MAX_BPM]
    return float(np.median(candidates)) if candidates else 0.0

def rr_from_peaks(sig: np.ndarray) -> np.ndarray:
    sig = detrend(sig); sig = bandpass(sig)
    peaks, _ = find_peaks(sig, distance=FS * 0.5)
    if len(peaks) < 2: return np.array([])
    rr_s = np.diff(peaks) / FS
    return rr_s * 1000.0

# -----------------------------
# Camera handling (Linux/WSL vs native Windows)
# -----------------------------
IS_WINDOWS = (os.name == "nt") or ("windows" in platform.system().lower())

SAFE_SIZES = [(640, 480), (800, 600), (1280, 720)]
PIXFMTS = [("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
           ("YUYV", cv2.VideoWriter_fourcc(*"YUYV"))]

def _try_open(idx: int, fourcc: Optional[int], size: Tuple[int,int], fps: int) -> Optional[cv2.VideoCapture]:
    be = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, be)
    if not cap.isOpened():
        cap.release(); return None
    if fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, _ = cap.read()
    if not ok:
        cap.release(); return None
    return cap

def _open_camera(idx: int, fps: int) -> Optional[cv2.VideoCapture]:
    # prefer MJPG then YUYV; iterate sizes
    for name, fcc in PIXFMTS:
        for w, h in SAFE_SIZES:
            cap = _try_open(idx, fcc, (w, h), int(fps))
            if cap:
                logger.info(f"Using {name} {w}x{h}@{fps} on index {idx}")
                return cap
    # last resort without forcing fourcc
    be = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
    cap = cv2.VideoCapture(idx, be)
    if cap.isOpened():
        return cap
    cap.release()
    return None

def capture_buffers(seconds: float) -> Tuple[np.ndarray, np.ndarray, float]:
    indices = [CAMERA_INDEX, 0, 1, 2]
    cap = None
    for i in indices:
        cap = _open_camera(i, int(FS))
        if cap: break
    if not cap:
        raise RuntimeError(f"Cannot open camera (tried {indices})")

    n = int(FS * seconds)
    buf_g: Deque[float] = deque(maxlen=n)
    buf_r: Deque[float] = deque(maxlen=n)
    buf_l: Deque[float] = deque(maxlen=n)

    for _ in range(8):
        cap.read(); time.sleep(0.01)

    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.005); continue
        frame = cv2.flip(frame, 1)
        rows, cols = get_roi(frame)
        roi = frame[rows, cols]
        if roi.size == 0: continue
        buf_g.append(float(roi[:, :, 1].mean()))
        buf_r.append(float(roi[:, :, 2].mean()))
        buf_l.append(float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()

    if len(buf_g) < FS * 2:
        raise RuntimeError("Insufficient data captured")
    return np.asarray(buf_g, float), np.asarray(buf_r, float), float(np.mean(buf_l))

# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.get("/camera")
def camera_probe():
    tried = []
    for i in [CAMERA_INDEX, 0, 1, 2]:
        cap = _open_camera(i, int(FS))
        ok = bool(cap)
        if cap: cap.release()
        tried.append({"index": i, "opened": ok})
    return {"is_windows": IS_WINDOWS, "tried": tried}

@app.get("/v4l2")
def v4l2_list():
    if IS_WINDOWS:
        return {"devices": "n/a on native Windows"}
    try:
        out = subprocess.check_output(shlex.split("v4l2-ctl --list-devices"), text=True, timeout=2)
    except Exception as e:
        out = f"n/a: {e}"
    return {"devices": out}

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(duration: float = Query(DEFAULT_SEC, gt=2, lt=30), count: int = Query(5, gt=1, lt=50)):
    rid = uuid.uuid4().hex[:12]
    readings: List[Reading] = []
    for _ in range(count):
        g, r, lum = capture_buffers(duration)
        hr = hr_estimators(g)
        rr_ms = rr_from_peaks(g)
        sdnn, rmssd, pnn50, lf, hf, lf_hf = hrv_features_from_rr(rr_ms)
        spo2 = compute_spo2(r, g)
        sys, dia = estimate_bp(hr)
        pi = perfusion_index(g)
        hb, hb_conf, hb_m = hb_from_ppg_proxy(r, g)
        readings.append(Reading(
            timestamp=datetime.now(timezone.utc).isoformat(),
            heart_rate=hr, sdnn=sdnn, rmssd=rmssd, pnn50=pnn50,
            lf=lf, hf=hf, lf_hf=lf_hf, spo2=spo2, systolic=sys, diastolic=dia,
            luminosity=lum, stress_level=classify_stress(sdnn),
            perfusion_index=pi, hb_estimate_g_dl=hb, hb_confidence=hb_conf, hb_method=hb_m
        ))

    def agg(name: str) -> float: return float(trim_mean([getattr(r, name) for r in readings], 0.2))
    stresses = [r.stress_level for r in readings]
    aggregate = Aggregate(
        heart_rate=agg('heart_rate'), sdnn=agg('sdnn'), rmssd=agg('rmssd'), pnn50=agg('pnn50'),
        lf=agg('lf'), hf=agg('hf'), lf_hf=agg('lf_hf'), spo2=agg('spo2'),
        systolic=agg('systolic'), diastolic=agg('diastolic'), luminosity=agg('luminosity'),
        perfusion_index=agg('perfusion_index'), hb_estimate_g_dl=agg('hb_estimate_g_dl'),
        stress_level=max(set(stresses), key=stresses.count),
    )
    payload = {"request_id": rid, "readings": [r.dict() for r in readings], "aggregate": aggregate.dict()}
    out_path = os.path.join(RECORD_DIR, f"metrics_{rid}.json")
    with open(out_path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)
    return MetricsResponse(request_id=rid, saved_path=out_path, readings=readings, aggregate=aggregate)

def _build_prompt(metrics: dict) -> str:
    agg = metrics.get("aggregate", {})
    lines = [
        "You are a clinical-adjacent assistant. Generate 5 concise, non-leading baseline questions for a human subject, based ONLY on the metrics below.",
        "Do not diagnose. No advice. One question per line.",
        "",
        "Key aggregates:",
        f"- heart_rate: {agg.get('heart_rate')}",
        f"- sdnn: {agg.get('sdnn')}",
        f"- rmssd: {agg.get('rmssd')}",
        f"- pnn50: {agg.get('pnn50')}",
        f"- spo2: {agg.get('spo2')}",
        f"- systolic/diastolic: {agg.get('systolic')}/{agg.get('diastolic')}",
        f"- stress_level: {agg.get('stress_level')}",
        f"- perfusion_index: {agg.get('perfusion_index')}",
        f"- hb_estimate_g_dl: {agg.get('hb_estimate_g_dl')}",
        "",
        "Output strict JSON array of 5 strings.",
    ]
    return "\n".join(lines)

async def _ollama_generate(prompt: str, model: str, temperature: float) -> str:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(url, json=payload)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error {r.status_code}: {r.text[:200]}")
        data = r.json()
        return data.get("response", "")

@app.post("/questions", response_model=LLMQuestionsOut)
async def questions_from_metrics(body: LLMQuestionsIn):
    src = body.path if os.path.isabs(body.path) else os.path.join(os.getcwd(), body.path)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail=f"File not found: {src}")
    try:
        with open(src, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    model = body.model or OLLAMA_MODEL
    temp = body.temperature if body.temperature is not None else OLLAMA_TEMPERATURE
    raw = await _ollama_generate(_build_prompt(metrics), model=model, temperature=temp)

    try:
        questions = json.loads(raw)
        if not isinstance(questions, list) or len(questions) != 5 or not all(isinstance(x, str) for x in questions):
            raise ValueError("unexpected shape")
    except Exception:
        questions = [q.strip("- ").strip() for q in raw.splitlines() if q.strip()][:5]

    return LLMQuestionsOut(
        request_id=metrics.get("request_id", uuid.uuid4().hex[:12]),
        source_file=src,
        used_model=model,
        questions=questions,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, log_level="info")
