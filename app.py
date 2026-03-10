import os
import json
import re
import csv
import io
import sqlite3
import concurrent.futures
import time
from pathlib import Path
from functools import wraps
from datetime import datetime
from dotenv import load_dotenv

from flask import (Flask, request, jsonify, session, redirect,
                   url_for, render_template, g, abort)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import google.generativeai as genai
from google.cloud import vision

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR_Q = UPLOAD_DIR / "questions"
UPLOAD_DIR_A = UPLOAD_DIR / "answers"

for d in (UPLOAD_DIR_Q, UPLOAD_DIR_A):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tiff", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-please")

FRONTEND_ORIGINS_RAW = os.environ.get(
    "FRONTEND_ORIGINS",
    "http://localhost:5000,http://127.0.0.1:5000,https://10-mathew.github.io",
)
FRONTEND_ORIGINS = [o.strip() for o in FRONTEND_ORIGINS_RAW.split(",") if o.strip()]

if FRONTEND_ORIGINS:
    CORS(
        app,
        resources={r"/api/*": {"origins": FRONTEND_ORIGINS}},
        supports_credentials=True,
    )

# Google / Gemini keys — set in environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_CLOUD_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OCR_LOW_CONF_THRESHOLD = float(os.environ.get("OCR_LOW_CONF_THRESHOLD", "0.75"))
GEMINI_TIMEOUT_SECONDS = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "30"))
GEMINI_USE_NETWORK = os.environ.get("GEMINI_USE_NETWORK", "false").lower() in {"1", "true", "yes", "on"}

# Optional: JSON credentials payload to avoid mounting a credentials file in deployment.
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
if GOOGLE_CREDENTIALS_JSON and not GOOGLE_CLOUD_CREDENTIALS:
    cred_path = BASE_DIR / "instance" / "gcp_credentials.json"
    cred_path.parent.mkdir(exist_ok=True)
    with open(cred_path, "w", encoding="utf-8") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
    GOOGLE_CLOUD_CREDENTIALS = str(cred_path)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CLOUD_CREDENTIALS

if os.environ.get("SESSION_COOKIE_SAMESITE"):
    app.config["SESSION_COOKIE_SAMESITE"] = os.environ.get("SESSION_COOKIE_SAMESITE")
if os.environ.get("SESSION_COOKIE_SECURE"):
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
if os.environ.get("SESSION_COOKIE_DOMAIN"):
    app.config["SESSION_COOKIE_DOMAIN"] = os.environ.get("SESSION_COOKIE_DOMAIN")

# Local HTTP safety: browsers reject/send cookies differently with SameSite=None+Secure on http://127.0.0.1.
# Keep production env behavior, but auto-adjust for local dev hosts.
_local_hosts = ("localhost", "127.0.0.1")
_origins_raw = FRONTEND_ORIGINS_RAW.lower()
_is_local_context = any(h in _origins_raw for h in _local_hosts)
if _is_local_context and app.config.get("SESSION_COOKIE_SECURE", False):
    app.config["SESSION_COOKIE_SECURE"] = False
    if str(app.config.get("SESSION_COOKIE_SAMESITE", "")).lower() == "none":
        app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Database ───────────────────────────────────────────────────────────────────

DB_PATH = Path(os.environ.get("DB_PATH", str(BASE_DIR / "instance" / "edueval.db")))
DB_PATH.parent.mkdir(exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS teachers (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    username  TEXT UNIQUE NOT NULL,
    password  TEXT NOT NULL,
    created   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS exams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id  INTEGER NOT NULL REFERENCES teachers(id),
    title       TEXT NOT NULL,
    exam_code   TEXT,
    qp_path     TEXT,
    qp_text     TEXT,
    created     TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id     INTEGER NOT NULL REFERENCES exams(id),
    number      TEXT NOT NULL,
    text        TEXT NOT NULL,
    max_marks   REAL NOT NULL DEFAULT 10,
    model_answer TEXT
);

CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id     INTEGER NOT NULL REFERENCES exams(id),
    name        TEXT NOT NULL,
    roll_no     TEXT,
    class_name  TEXT,
    corrected_script TEXT
);

CREATE TABLE IF NOT EXISTS answer_sheets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id),
    file_path   TEXT NOT NULL,
    ocr_text    TEXT,
    confidence  REAL,
    low_conf_words TEXT,
    processed   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id      INTEGER NOT NULL REFERENCES students(id),
    question_id     INTEGER NOT NULL REFERENCES questions(id),
    student_answer  TEXT,
    marks_awarded   REAL,
    reason          TEXT,
    evaluated_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id),
    total_marks REAL,
    max_marks   REAL,
    feedback    TEXT,
    generated   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rubrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id        INTEGER NOT NULL REFERENCES exams(id),
    question_id    INTEGER NOT NULL REFERENCES questions(id),
    criteria_json  TEXT NOT NULL,
    version        INTEGER DEFAULT 1,
    created_by     INTEGER REFERENCES teachers(id),
    created        TEXT DEFAULT (datetime('now')),
    UNIQUE(exam_id, question_id, version)
);

CREATE TABLE IF NOT EXISTS rubric_templates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_id    INTEGER NOT NULL REFERENCES teachers(id),
    name          TEXT NOT NULL,
    subject       TEXT,
    chapter       TEXT,
    template_json TEXT NOT NULL,
    created       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS marking_drafts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id       INTEGER NOT NULL REFERENCES students(id),
    question_id      INTEGER NOT NULL REFERENCES questions(id),
    ai_marks         REAL,
    ai_reason        TEXT,
    ai_confidence    REAL,
    teacher_marks    REAL,
    teacher_reason   TEXT,
    status           TEXT NOT NULL DEFAULT 'pending',
    student_answer   TEXT,
    rubric_snapshot  TEXT,
    generated_at     TEXT DEFAULT (datetime('now')),
    reviewed_at      TEXT
);

CREATE TABLE IF NOT EXISTS result_approvals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id   INTEGER NOT NULL REFERENCES students(id),
    approved_by  TEXT,
    approved_at  TEXT,
    finalized    INTEGER NOT NULL DEFAULT 0,
    finalized_at TEXT
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=30.0)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode = WAL")
        g.db.execute("PRAGMA synchronous = NORMAL")
        g.db.execute("PRAGMA busy_timeout = 30000")
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db:
        db.close()


def init_db():
    with app.app_context():
        db = get_db()
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = NORMAL")
        db.execute("PRAGMA busy_timeout = 30000")
        db.executescript(SCHEMA)
        existing_exam_cols = {r["name"] for r in db.execute("PRAGMA table_info(exams)").fetchall()}
        if "exam_code" not in existing_exam_cols:
            db.execute("ALTER TABLE exams ADD COLUMN exam_code TEXT")
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_exams_teacher_exam_code ON exams(teacher_id, exam_code)"
        )
        # Lightweight forward-only migrations for existing local DBs.
        existing_student_cols = {r["name"] for r in db.execute("PRAGMA table_info(students)").fetchall()}
        if "corrected_script" not in existing_student_cols:
            db.execute("ALTER TABLE students ADD COLUMN corrected_script TEXT")
        if "roll_no" not in existing_student_cols:
            db.execute("ALTER TABLE students ADD COLUMN roll_no TEXT")
        if "class_name" not in existing_student_cols:
            db.execute("ALTER TABLE students ADD COLUMN class_name TEXT")
        existing_sheet_cols = {r["name"] for r in db.execute("PRAGMA table_info(answer_sheets)").fetchall()}
        if "low_conf_words" not in existing_sheet_cols:
            db.execute("ALTER TABLE answer_sheets ADD COLUMN low_conf_words TEXT")
        db.commit()
        # Seed a demo teacher if none exist
        if not db.execute("SELECT 1 FROM teachers LIMIT 1").fetchone():
            db.execute(
                "INSERT INTO teachers (username, password) VALUES (?, ?)",
                ("teacher", generate_password_hash("password")),
            )
            db.commit()


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "teacher_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


def page_login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "teacher_id" not in session:
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return wrapper


def get_exam_or_404(exam_id):
    db = get_db()
    exam = db.execute(
        "SELECT * FROM exams WHERE id=? AND teacher_id=?",
        (exam_id, session["teacher_id"]),
    ).fetchone()
    if not exam:
        abort(404)
    return exam


def get_student_owned(student_id):
    db = get_db()
    return db.execute(
        """SELECT s.*, e.teacher_id
           FROM students s
           JOIN exams e ON e.id = s.exam_id
           WHERE s.id=? AND e.teacher_id=?""",
        (student_id, session["teacher_id"]),
    ).fetchone()


def safe_json_load(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except Exception:
        return fallback


def run_db_write(db, writer_fn, *, max_attempts=3):
    """Run a write transaction with retry on transient sqlite lock errors."""
    for attempt in range(max_attempts):
        try:
            writer_fn()
            db.commit()
            return None
        except sqlite3.OperationalError as exc:
            db.rollback()
            if "locked" in str(exc).lower() and attempt < max_attempts - 1:
                time.sleep(0.2 * (attempt + 1))
                continue
            return str(exc)
    return "Database write failed"


def _pdf_escape(text: str) -> str:
    return str(text or "").replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_simple_results_pdf(exam_id: int, rows: list) -> bytes:
    """Build a minimal valid single-page PDF with tabular text content."""
    lines = [f"EduEval Results - Exam {exam_id}", "Roll No | Name | Total/Max", "-" * 72]
    for r in rows:
        roll = str(r.get("roll_no", "") or "-")
        name = str(r.get("name", "") or "-")
        total = str(r.get("total_marks", "") or "0")
        max_marks = str(r.get("max_marks", "") or "0")
        lines.append(f"{roll} | {name} | {total}/{max_marks}")

    # Keep lines reasonably short for fixed-width rendering.
    clipped = [(ln[:120] + "..." if len(ln) > 123 else ln) for ln in lines]
    font_size = 11
    line_h = 14
    start_y = 800
    text_ops = ["BT", "/F1 11 Tf", f"72 {start_y} Td"]
    first = True
    for ln in clipped[:50]:
        esc = _pdf_escape(ln)
        if first:
            text_ops.append(f"({esc}) Tj")
            first = False
        else:
            text_ops.append(f"0 -{line_h} Td ({esc}) Tj")
    text_ops.append("ET")
    stream_data = "\n".join(text_ops).encode("latin-1", "replace")

    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objs.append(b"<< /Length " + str(len(stream_data)).encode("ascii") + b" >>\nstream\n" + stream_data + b"\nendstream")

    out = bytearray()
    out.extend(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n")
    offsets = [0]  # xref entry 0
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


EXAM_CODE_RE = re.compile(r"^[A-Z0-9-]{3,24}$")


def normalize_exam_code(raw: str) -> str:
    return re.sub(r"\s+", "-", (raw or "").strip().upper())


def validate_exam_code(raw: str):
    code = normalize_exam_code(raw)
    if not code:
        return None, "Exam code is required"
    if not EXAM_CODE_RE.fullmatch(code):
        return None, "Exam code must be 3-24 chars using A-Z, 0-9, or '-'"
    return code, None


def ensure_exam_code_unique(db, teacher_id: int, exam_code: str, ignore_exam_id=None):
    if ignore_exam_id is not None:
        row = db.execute(
            "SELECT id FROM exams WHERE teacher_id=? AND exam_code=? AND id<>?",
            (teacher_id, exam_code, ignore_exam_id),
        ).fetchone()
    else:
        row = db.execute(
            "SELECT id FROM exams WHERE teacher_id=? AND exam_code=?",
            (teacher_id, exam_code),
        ).fetchone()
    return row is None


def exam_code_or_empty(exam_row: dict):
    return (exam_row.get("exam_code") or "").strip()


# ── File helpers ──────────────────────────────────────────────────────────────

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


def save_upload(file, directory):
    filename = secure_filename(file.filename)
    dest = directory / filename
    # Avoid name collision
    counter = 1
    while dest.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        dest = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    file.save(dest)
    return str(dest)


# ── OCR (Google Cloud Vision) ─────────────────────────────────────────────────

def ocr_image(file_path: str) -> dict:
    """Run OCR using Google Vision for images and PDF/TIFF files."""
    try:
        client = vision.ImageAnnotatorClient()
        path = Path(file_path)
        with open(path, "rb") as f:
            content = f.read()

        suffix = path.suffix.lower()
        confidences = []
        low_conf_words = []

        def collect_word_conf(words_iter):
            for word in words_iter:
                txt = "".join(s.text for s in word.symbols).strip()
                conf = float(word.confidence or 0.0)
                if conf < OCR_LOW_CONF_THRESHOLD and txt:
                    low_conf_words.append({"text": txt, "confidence": round(conf, 4)})
                confidences.append(conf)

        # PDF/TIFF must use file-based annotate API, not image API.
        if suffix in {".pdf", ".tif", ".tiff"}:
            mime = "application/pdf" if suffix == ".pdf" else "image/tiff"
            request = vision.AnnotateFileRequest(
                input_config=vision.InputConfig(content=content, mime_type=mime),
                features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
            )
            batch = client.batch_annotate_files(requests=[request])
            file_resp = batch.responses[0] if batch.responses else None
            if not file_resp:
                return {"text": "", "confidence": 0.0, "error": "No OCR response for file"}
            if file_resp.error.message:
                return {"text": "", "confidence": 0.0, "error": file_resp.error.message}

            page_texts = []
            for page_resp in file_resp.responses:
                if page_resp.error.message:
                    continue
                ann = page_resp.full_text_annotation
                if ann and ann.text:
                    page_texts.append(ann.text)
                for page in ann.pages if ann else []:
                    for block in page.blocks:
                        for para in block.paragraphs:
                            collect_word_conf(para.words)

            full_text = "\n".join(page_texts).strip()
        else:
            img = vision.Image(content=content)
            response = client.document_text_detection(image=img)
            if response.error.message:
                return {"text": "", "confidence": 0.0, "error": response.error.message}
            ann = response.full_text_annotation
            full_text = (ann.text or "").strip()
            for page in ann.pages if ann else []:
                for block in page.blocks:
                    for para in block.paragraphs:
                        collect_word_conf(para.words)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": full_text,
            "confidence": round(avg_conf, 4),
            "low_conf_words": low_conf_words[:200],
            "error": None,
        }
    except Exception as e:
        return {"text": "", "confidence": 0.0, "low_conf_words": [], "error": str(e)}


def mock_ocr(file_path: str) -> dict:
    """Fallback mock OCR when Vision API isn't configured."""
    return {
        "text": (
            "Q1 The process of photosynthesis converts sunlight into chemical energy stored in glucose. "
            "Plants use carbon dioxide and water to produce glucose and oxygen.\n\n"
            "Q2 Newton's second law states that force equals mass times acceleration (F=ma). "
            "This means a larger force produces greater acceleration for the same mass.\n\n"
            "Q3 The French Revolution began in 1789 due to financial crisis and social inequality. "
            "The people rose against the monarchy and demanded liberty and equality."
        ),
        "confidence": 0.91,
        "low_conf_words": [],
        "error": None,
    }


def do_ocr(file_path: str) -> dict:
    if GOOGLE_CLOUD_CREDENTIALS and Path(GOOGLE_CLOUD_CREDENTIALS).exists():
        result = ocr_image(file_path)
        # Graceful fallback: if Vision is unreachable (DNS/network/API), keep flow usable.
        if result.get("error"):
            mock = mock_ocr(file_path)
            mock["error"] = f"Vision OCR failed; fallback used: {result.get('error')}"
            return mock
        return result
    return mock_ocr(file_path)


# ── Gemini helpers ────────────────────────────────────────────────────────────

def gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY or not GEMINI_USE_NETWORK:
        return _mock_gemini(prompt)
    model_names = []
    for name in [GEMINI_MODEL, "gemini-2.5-flash"]:
        if name and name not in model_names:
            model_names.append(name)
    last_error = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.generate_content, prompt)
                response = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
            if response and response.text:
                return response.text
        except concurrent.futures.TimeoutError:
            last_error = TimeoutError(
                f"{model_name} timed out after {GEMINI_TIMEOUT_SECONDS:.0f}s"
            )
        except Exception as exc:
            last_error = exc
            continue
    # Fail fast to mock so UI flows remain responsive when upstream AI is slow/unavailable.
    return _mock_gemini(prompt)


def _mock_gemini(prompt: str) -> str:
    lower = prompt.lower()
    if "evaluate" in lower or " mark " in f" {lower} " or "marking" in lower or "rubric" in lower:
        return json.dumps(
            {
                "marks": 7,
                "reason": "Good understanding of key concepts. Student correctly identified main points but missed some detail on the mechanism.",
                "confidence": 0.72,
            }
        )
    if "model answer" in lower:
        return json.dumps([
            {"number": "Q1", "model_answer": "Photosynthesis is the process by which plants use sunlight, water, and CO₂ to produce glucose and oxygen. It occurs in chloroplasts via the light-dependent and Calvin cycle reactions. The overall equation is 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂."},
            {"number": "Q2", "model_answer": "Newton's second law (F=ma) states that the net force on an object equals its mass multiplied by its acceleration. A larger net force produces proportionally greater acceleration; doubling the force doubles the acceleration for a given mass."},
            {"number": "Q3", "model_answer": "The French Revolution (1789–1799) was caused by fiscal crisis, Enlightenment ideas, social inequality under the Estates system, and food shortages. It abolished the monarchy, introduced the Declaration of the Rights of Man, and radically reshaped European politics."},
        ])
    return "Mock Gemini response."


def generate_model_answers(questions: list) -> list:
    """Ask Gemini to produce model answers for a list of questions."""
    q_text = "\n".join(
        f"{q['number']} (max {q['max_marks']} marks): {q['text']}"
        for q in questions
    )
    prompt = f"""You are an expert examiner. For each question below, write a concise model answer (2–5 sentences) suitable for marking. Return ONLY valid JSON: a list of objects with keys "number" and "model_answer".

Questions:
{q_text}
"""
    raw = gemini_generate(prompt)
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def build_fallback_model_answers(questions: list) -> list:
    """Create deterministic model answers when AI response is unavailable."""
    fallback = []
    for q in questions:
        number = str(q.get("number", "")).strip()
        text = re.sub(r"\s+", " ", str(q.get("text", "")).strip())
        max_marks = q.get("max_marks", 10)
        summary = text if len(text) <= 220 else f"{text[:217]}..."
        if not summary:
            summary = "the key concepts asked in the question"
        fallback.append(
            {
                "number": number,
                "model_answer": (
                    f"A full-credit response should clearly address {summary}. "
                    f"Maximum marks: {max_marks}."
                ),
            }
        )
    return fallback


def evaluate_answer(question_text, model_answer, student_answer, max_marks) -> dict:
    prompt = f"""You are an examiner marking a student's answer.

Question: {question_text}
Model Answer: {model_answer}
Student Answer: {student_answer}
Maximum Marks: {max_marks}

Evaluate the student's answer against the model answer. Award marks fairly (partial credit allowed). Return ONLY valid JSON with keys "marks" (a number) and "reason" (a brief 1–2 sentence explanation).
"""
    raw = gemini_generate(prompt)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        data = json.loads(raw)
        marks = min(float(data.get("marks", 0)), float(max_marks))
        return {"marks": marks, "reason": data.get("reason", "")}
    except Exception:
        return {"marks": 0, "reason": "Evaluation failed — please review manually."}


# ── Script segmentation ───────────────────────────────────────────────────────

def canonical_question_number(value) -> str:
    """Normalize question labels like '1', 'Q1', 'q.1' to 'Q1'."""
    txt = str(value or "").strip().upper()
    if not txt:
        return ""
    m = re.search(r"(\d+)", txt)
    if not m:
        return txt
    return f"Q{int(m.group(1))}"


def segment_script(script_text: str, question_numbers: list) -> dict:
    """Split a student script by question markers, return {q_number: answer_text}."""
    text = (script_text or "").replace("\r", "\n")
    if not text.strip():
        return {}
    expected = [canonical_question_number(q) for q in (question_numbers or []) if canonical_question_number(q)]
    expected_set = set(expected)

    # Supports:
    # Q1 / Q.1 / Question 1 / 1) / 1. / (1)
    marker = re.compile(
        r"(?:^|\n)\s*(?:"
        r"q(?:uestion)?\s*\.?\s*(\d+)\s*[:.)-]?\s*"
        r"|"
        r"\(?(\d+)\)?\s*[:.)-]\s+"
        r")",
        re.IGNORECASE,
    )
    matches = list(marker.finditer(text))
    segments = {}
    accepted = []
    seen = set()
    for m in matches:
        raw_no = (m.group(1) or m.group(2) or "").strip()
        if not raw_no:
            continue
        q_num = canonical_question_number(raw_no)
        # If expected questions are known, ignore marker-like bullets not in expected set.
        if expected_set and q_num not in expected_set:
            continue
        # Use first occurrence only to avoid nested numbering in answer text re-mapping questions.
        if q_num in seen:
            continue
        seen.add(q_num)
        accepted.append((m.start(), m.end(), q_num))

    for i, (_, marker_end, q_num) in enumerate(accepted):
        next_start = accepted[i + 1][0] if i + 1 < len(accepted) else len(text)
        ans = text[marker_end:next_start].strip()
        if ans:
            segments[q_num] = ans

    # Fallback: only do full-text-to-first-question for single-question exams.
    if not segments and len(expected) == 1:
        segments[expected[0]] = text.strip()

    return segments


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json or {}
    db = get_db()
    teacher = db.execute(
        "SELECT * FROM teachers WHERE username=?", (data.get("username", ""),)
    ).fetchone()
    if not teacher or not check_password_hash(teacher["password"], data.get("password", "")):
        return jsonify({"error": "Invalid credentials"}), 401
    session["teacher_id"] = teacher["id"]
    session["username"] = teacher["username"]
    return jsonify({"message": "Logged in", "username": teacher["username"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"message": "Logged out"})


@app.route("/api/me", methods=["GET"])
def api_me():
    if "teacher_id" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify(
        {
            "authenticated": True,
            "teacher_id": session["teacher_id"],
            "username": session.get("username", ""),
        }
    )


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    confirm_password = data.get("confirm_password", "")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if confirm_password and password != confirm_password:
        return jsonify({"error": "Passwords do not match"}), 400
    db = get_db()
    try:
        db.execute(
            "INSERT INTO teachers (username, password) VALUES (?, ?)",
            (username, generate_password_hash(password)),
        )
        db.commit()
        return jsonify({"message": "Registered"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already taken"}), 409


def parse_questions_from_text(text: str) -> list:
    """
    Parse question blocks from OCR text.
    Supports markers like Q1, Question 1, 1), 1., etc.
    """
    if not text or not text.strip():
        return []

    normalized = text.replace("\r", "\n")
    lines = [ln.strip() for ln in normalized.split("\n") if ln.strip()]
    if not lines:
        return []

    pattern = re.compile(
        r"^(?:q(?:uestion)?\s*\.?\s*(\d+)|\(?(\d+)\)?[.)]\s+)(.*)$",
        re.IGNORECASE,
    )

    blocks = []
    current = None

    for line in lines:
        m = pattern.match(line)
        if m:
            if current:
                blocks.append(current)
            q_no = m.group(1) or m.group(2) or str(len(blocks) + 1)
            tail = (m.group(3) or "").strip()
            current = {"number": f"Q{q_no}", "text_lines": [tail] if tail else []}
        elif current:
            current["text_lines"].append(line)

    if current:
        blocks.append(current)

    if not blocks:
        # Fallback: treat all text as one prompt.
        joined = " ".join(lines).strip()
        if not joined:
            return []
        blocks = [{"number": "Q1", "text_lines": [joined]}]

    def _extract_max_marks(prompt_lines):
        """
        Extract marks from common layouts:
        1) Inline: "(5 marks)", "[5 marks]", "5 marks"
        2) Right-aligned OCR tail: "... question text 1"
        3) Right-aligned OCR next line: last line is just "1"
        """
        lines = [ln.strip() for ln in (prompt_lines or []) if ln.strip()]
        if not lines:
            return 5.0, "", False

        # Handle trailing numeric-only lines.
        # If exactly one numeric tail exists, treat as local mark.
        # If multiple numeric tails exist, they are usually detached right-column marks:
        # remove from prompt and let global detached-marks mapper assign them.
        tail_marks = None
        numeric_tail = 0
        for ln in reversed(lines):
            if re.fullmatch(r"\d{1,2}(?:\.\d+)?", ln):
                numeric_tail += 1
            else:
                break
        if numeric_tail == 1 and len(lines) >= 2:
            candidate = float(lines[-1])
            if 0 < candidate <= 100:
                tail_marks = candidate
                lines = lines[:-1]
        elif numeric_tail > 1:
            lines = lines[:-numeric_tail]

        prompt = re.sub(r"\s+", " ", " ".join(lines).strip())
        prompt = re.sub(r"^[\s\.\)\]:-]+", "", prompt).strip()
        if not prompt:
            return tail_marks or 5.0, "", bool(tail_marks)

        # Case 1: explicit marks text.
        explicit_patterns = [
            r"\((\d+(?:\.\d+)?)\s*marks?\)",
            r"\[(\d+(?:\.\d+)?)\s*marks?\]",
            r"\b(\d+(?:\.\d+)?)\s*marks?\b",
        ]
        for pat in explicit_patterns:
            m = re.search(pat, prompt, flags=re.IGNORECASE)
            if m:
                marks = float(m.group(1))
                cleaned = re.sub(pat, "", prompt, flags=re.IGNORECASE)
                cleaned = re.sub(r"\s+", " ", cleaned).strip(" .:-")
                return marks, cleaned, True

        # Case 2: trailing number at end of line/prompt.
        m_tail = re.search(r"^(.*?)(?:\s+)(\d{1,2}(?:\.\d+)?)\s*$", prompt)
        if m_tail:
            candidate = float(m_tail.group(2))
            prefix = m_tail.group(1).strip()
            # Heuristic: accept only realistic mark values and non-empty prompt.
            if prefix and 0 < candidate <= 100:
                return candidate, prefix, True

        return tail_marks or 5.0, prompt, bool(tail_marks)

    provisional = []
    for b in blocks:
        max_marks, prompt, has_local_marks = _extract_max_marks(b["text_lines"])
        if not prompt:
            continue
        # a), b), c) ... in one question usually indicate split marking.
        joined_block = " ".join((b["text_lines"] or []))
        subparts = len(re.findall(r"\b[a-d]\)\s*", joined_block, flags=re.IGNORECASE))
        provisional.append(
            {
                "number": b["number"],
                "text": prompt,
                "max_marks": float(max_marks),
                "_has_local_marks": has_local_marks,
                "_subparts": max(1, subparts),
            }
        )

    if not provisional:
        return []

    # Detached right-column marks: lines that contain only a numeric mark.
    detached_marks = []
    for ln in lines:
        if re.fullmatch(r"\d{1,2}(?:\.\d+)?", ln):
            v = float(ln)
            if 0 < v <= 100:
                detached_marks.append(v)

    # Assign detached marks sequentially to questions that still lack marks.
    # For subparts (a/b), consume multiple marks and sum.
    if detached_marks:
        idx = 0
        for q in provisional:
            if q["_has_local_marks"]:
                continue
            need = q["_subparts"]
            chunk = detached_marks[idx:idx + need]
            if not chunk:
                break
            q["max_marks"] = float(sum(chunk))
            q["_has_local_marks"] = True
            idx += need

    return [{"number": q["number"], "text": q["text"], "max_marks": q["max_marks"]} for q in provisional]


# ── Exam routes ───────────────────────────────────────────────────────────────

@app.route("/api/exams", methods=["GET"])
@login_required
def list_exams():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM exams WHERE teacher_id=? ORDER BY id DESC",
        (session["teacher_id"],),
    ).fetchall()
    exams = []
    for r in rows:
        item = dict(r)
        item["exam_code"] = item.get("exam_code") or ""
        exams.append(item)
    return jsonify(exams)


@app.route("/api/exams", methods=["POST"])
@login_required
def create_exam():
    title = (request.form.get("title") or "").strip()
    exam_code_raw = request.form.get("exam_code")
    exam_code, exam_code_err = validate_exam_code(exam_code_raw)
    if not title:
        return jsonify({"error": "Title required"}), 400
    if exam_code_err:
        return jsonify({"error": exam_code_err}), 400
    qp_path = None
    qp_text = None
    if "question_paper" in request.files:
        f = request.files["question_paper"]
        if f.filename:
            if not allowed_file(f.filename):
                return jsonify({"error": "Unsupported question paper file type"}), 400
            qp_path = save_upload(f, UPLOAD_DIR_Q)
    db = get_db()
    if not ensure_exam_code_unique(db, session["teacher_id"], exam_code):
        return jsonify({"error": "Exam code already exists"}), 409
    cur = db.execute(
        "INSERT INTO exams (teacher_id, title, exam_code, qp_path) VALUES (?, ?, ?, ?)",
        (session["teacher_id"], title, exam_code, qp_path),
    )
    db.commit()
    return jsonify({"id": cur.lastrowid, "title": title, "exam_code": exam_code})


@app.route("/api/exams/<int:exam_id>", methods=["GET"])
@login_required
def get_exam(exam_id):
    exam = get_exam_or_404(exam_id)
    db = get_db()
    questions = db.execute(
        "SELECT * FROM questions WHERE exam_id=? ORDER BY rowid", (exam_id,)
    ).fetchall()
    students = db.execute(
        "SELECT * FROM students WHERE exam_id=?", (exam_id,)
    ).fetchall()
    draft_stats = db.execute(
        """SELECT d.status, COUNT(*) as cnt
           FROM marking_drafts d
           JOIN students s ON s.id = d.student_id
           WHERE s.exam_id=?
           GROUP BY d.status""",
        (exam_id,),
    ).fetchall()
    rubric_count = db.execute("SELECT COUNT(*) as c FROM rubrics WHERE exam_id=?", (exam_id,)).fetchone()["c"]
    exam_dict = dict(exam)
    exam_dict["exam_code"] = exam_dict.get("exam_code") or ""
    return jsonify({
        **exam_dict,
        "questions": [dict(q) for q in questions],
        "students": [dict(s) for s in students],
        "rubric_count": rubric_count,
        "draft_stats": {r["status"]: r["cnt"] for r in draft_stats},
    })


@app.route("/api/exams/<int:exam_id>", methods=["DELETE"])
@login_required
def delete_exam(exam_id):
    exam = get_exam_or_404(exam_id)
    db = get_db()

    students = db.execute(
        "SELECT id FROM students WHERE exam_id=?",
        (exam_id,),
    ).fetchall()
    student_ids = [s["id"] for s in students]

    sheets = []
    if student_ids:
        placeholders = ",".join("?" for _ in student_ids)
        sheets = db.execute(
            f"SELECT id, file_path FROM answer_sheets WHERE student_id IN ({placeholders})",
            student_ids,
        ).fetchall()

    try:
        db.execute("BEGIN")

        # Remove evaluation/feedback/sheets for students in this exam.
        for sid in student_ids:
            db.execute("DELETE FROM evaluations WHERE student_id=?", (sid,))
            db.execute("DELETE FROM feedback_logs WHERE student_id=?", (sid,))
            db.execute("DELETE FROM marking_drafts WHERE student_id=?", (sid,))
            db.execute("DELETE FROM result_approvals WHERE student_id=?", (sid,))
            db.execute("DELETE FROM answer_sheets WHERE student_id=?", (sid,))

        db.execute("DELETE FROM students WHERE exam_id=?", (exam_id,))
        db.execute("DELETE FROM rubrics WHERE exam_id=?", (exam_id,))
        db.execute("DELETE FROM questions WHERE exam_id=?", (exam_id,))
        db.execute("DELETE FROM exams WHERE id=? AND teacher_id=?", (exam_id, session["teacher_id"]))
        db.commit()
    except Exception:
        db.rollback()
        raise

    # Best-effort local file cleanup.
    for sheet in sheets:
        path = (sheet["file_path"] or "").strip()
        if path and Path(path).exists():
            try:
                Path(path).unlink()
            except Exception:
                pass

    qp_path = (exam["qp_path"] or "").strip()
    if qp_path and Path(qp_path).exists():
        try:
            Path(qp_path).unlink()
        except Exception:
            pass

    return jsonify({"message": "Exam deleted", "exam_id": exam_id})


# ── Question routes ───────────────────────────────────────────────────────────

@app.route("/api/exams/<int:exam_id>/questions", methods=["POST"])
@login_required
def add_questions(exam_id):
    get_exam_or_404(exam_id)
    data = request.json or {}
    questions = data.get("questions", [])  # [{number, text, max_marks}]
    db = get_db()
    db.execute("DELETE FROM questions WHERE exam_id=?", (exam_id,))
    for q in questions:
        db.execute(
            "INSERT INTO questions (exam_id, number, text, max_marks) VALUES (?,?,?,?)",
            (exam_id, q["number"], q["text"], q.get("max_marks", 10)),
        )
    db.commit()
    return jsonify({"message": f"{len(questions)} questions saved"})


@app.route("/api/exams/<int:exam_id>/generate-model-answers", methods=["POST"])
@login_required
def gen_model_answers(exam_id):
    get_exam_or_404(exam_id)
    db = get_db()
    questions = db.execute(
        "SELECT * FROM questions WHERE exam_id=?", (exam_id,)
    ).fetchall()
    if not questions:
        return jsonify({"error": "No questions found"}), 400
    q_list = [dict(q) for q in questions]
    try:
        answers = generate_model_answers(q_list)
    except Exception:
        answers = []
    if not answers:
        answers = build_fallback_model_answers(q_list)
    ans_map = {a["number"]: a["model_answer"] for a in answers}
    for q in q_list:
        ma = ans_map.get(q["number"])
        if ma:
            db.execute(
                "UPDATE questions SET model_answer=? WHERE id=?", (ma, q["id"])
            )
    db.commit()
    return jsonify({"model_answers": ans_map})


@app.route("/api/exams/<int:exam_id>/auto-prepare", methods=["POST"])
@login_required
def auto_prepare_exam(exam_id):
    exam = get_exam_or_404(exam_id)
    if not exam["qp_path"]:
        return jsonify({"error": "Question paper file is required to auto-prepare"}), 400

    db = get_db()
    qp_text = (exam["qp_text"] or "").strip()
    if not qp_text:
        ocr = do_ocr(exam["qp_path"])
        qp_text = (ocr.get("text") or "").strip()
        if not qp_text:
            ocr_error = ocr.get("error") or "No text detected"
            return jsonify({"error": f"Could not extract text from question paper: {ocr_error}"}), 400
        db.execute("UPDATE exams SET qp_text=? WHERE id=?", (qp_text, exam_id))
        db.commit()

    parsed = parse_questions_from_text(qp_text)
    if not parsed:
        return jsonify({"error": "No questions could be parsed from question paper"}), 400

    db.execute("DELETE FROM questions WHERE exam_id=?", (exam_id,))
    for q in parsed:
        db.execute(
            "INSERT INTO questions (exam_id, number, text, max_marks) VALUES (?,?,?,?)",
            (exam_id, q["number"], q["text"], q["max_marks"]),
        )
    db.commit()

    questions = db.execute(
        "SELECT * FROM questions WHERE exam_id=? ORDER BY rowid", (exam_id,)
    ).fetchall()
    q_list = [dict(q) for q in questions]

    fallback_used = False
    fallback_reason = ""
    try:
        answers = generate_model_answers(q_list)
    except Exception as exc:
        answers = []
        fallback_reason = str(exc)
    if not answers:
        answers = build_fallback_model_answers(q_list)
        fallback_used = True
        if not fallback_reason:
            fallback_reason = "Model returned empty response"

    ans_map = {a["number"]: a["model_answer"] for a in answers if a.get("number") and a.get("model_answer")}
    for q in q_list:
        ma = ans_map.get(q["number"])
        if ma:
            db.execute("UPDATE questions SET model_answer=? WHERE id=?", (ma, q["id"]))
    db.commit()

    fresh = db.execute(
        "SELECT number, text, max_marks, model_answer FROM questions WHERE exam_id=? ORDER BY rowid",
        (exam_id,),
    ).fetchall()

    return jsonify(
        {
            "exam_id": exam_id,
            "questions": [dict(q) for q in fresh],
            "question_count": len(fresh),
            "model_answers_generated": sum(1 for q in fresh if q["model_answer"]),
            "model_answer_fallback_used": fallback_used,
            "model_answer_note": (
                f"Used fallback model answers: {fallback_reason}" if fallback_used else ""
            ),
        }
    )


# ── Student / answer-sheet routes ─────────────────────────────────────────────

@app.route("/api/exams/<int:exam_id>/students", methods=["POST"])
@login_required
def add_student(exam_id):
    get_exam_or_404(exam_id)
    data = request.form
    name = data.get("name", "").strip()
    roll_no = data.get("roll_no", "").strip() or None
    class_name = data.get("class_name", "").strip() or None
    if not name:
        return jsonify({"error": "Student name required"}), 400
    db = get_db()
    if roll_no:
        existing = db.execute(
            "SELECT id FROM students WHERE exam_id=? AND roll_no=?",
            (exam_id, roll_no),
        ).fetchone()
        if existing:
            db.execute(
                "UPDATE students SET name=?, class_name=? WHERE id=?",
                (name, class_name, existing["id"]),
            )
            student_id = existing["id"]
        else:
            cur = db.execute(
                "INSERT INTO students (exam_id, name, roll_no, class_name) VALUES (?, ?, ?, ?)",
                (exam_id, name, roll_no, class_name),
            )
            student_id = cur.lastrowid
    else:
        cur = db.execute(
            "INSERT INTO students (exam_id, name, roll_no, class_name) VALUES (?, ?, ?, ?)",
            (exam_id, name, roll_no, class_name),
        )
        student_id = cur.lastrowid
    # Handle uploaded answer sheets
    files = request.files.getlist("sheets")
    sheet_ids = []
    for f in files:
        if not f.filename:
            continue
        if not allowed_file(f.filename):
            continue
        path = save_upload(f, UPLOAD_DIR_A)
        cur2 = db.execute(
            "INSERT INTO answer_sheets (student_id, file_path) VALUES (?, ?)",
            (student_id, path),
        )
        sheet_ids.append(cur2.lastrowid)
    if not sheet_ids:
        db.execute("DELETE FROM students WHERE id=?", (student_id,))
        db.commit()
        return jsonify({"error": "At least one valid answer sheet file is required"}), 400
    db.commit()
    return jsonify({"student_id": student_id, "sheet_ids": sheet_ids})


@app.route("/api/students/<int:student_id>/ocr", methods=["POST"])
@login_required
def run_ocr(student_id):
    db = get_db()
    owner = db.execute(
        """SELECT s.id
           FROM students s
           JOIN exams e ON e.id = s.exam_id
           WHERE s.id=? AND e.teacher_id=?""",
        (student_id, session["teacher_id"]),
    ).fetchone()
    if not owner:
        return jsonify({"error": "Student not found"}), 404

    sheets = db.execute(
        "SELECT * FROM answer_sheets WHERE student_id=?", (student_id,)
    ).fetchall()
    if not sheets:
        return jsonify({"error": "No answer sheets found"}), 404
    combined_text = []
    total_conf = []
    errors = []
    low_conf_all = []
    for sheet in sheets:
        result = do_ocr(sheet["file_path"])
        if result["error"]:
            errors.append(result["error"])
        combined_text.append(result["text"])
        total_conf.append(result["confidence"])
        low_conf_words = result.get("low_conf_words", [])
        low_conf_all.extend(low_conf_words)
        db.execute(
            "UPDATE answer_sheets SET ocr_text=?, confidence=?, low_conf_words=?, processed=1 WHERE id=?",
            (result["text"], result["confidence"], json.dumps(low_conf_words), sheet["id"]),
        )
    db.commit()
    avg_conf = sum(total_conf) / len(total_conf) if total_conf else 0
    requires_review = avg_conf < OCR_LOW_CONF_THRESHOLD or len(low_conf_all) > 0
    return jsonify({
        "full_text": "\n\n".join(combined_text),
        "avg_confidence": avg_conf,
        "low_conf_words": low_conf_all[:200],
        "requires_review": requires_review,
        "errors": errors,
    })


@app.route("/api/students/<int:student_id>/ocr", methods=["GET"])
@login_required
def get_ocr_for_review(student_id):
    db = get_db()
    student = db.execute(
        """SELECT s.id, s.name, s.corrected_script
           FROM students s
           JOIN exams e ON e.id = s.exam_id
           WHERE s.id=? AND e.teacher_id=?""",
        (student_id, session["teacher_id"]),
    ).fetchone()
    if not student:
        return jsonify({"error": "Student not found"}), 404

    sheets = db.execute(
        "SELECT id, ocr_text, confidence, low_conf_words, processed FROM answer_sheets WHERE student_id=?",
        (student_id,),
    ).fetchall()
    combined = "\n\n".join((s["ocr_text"] or "") for s in sheets if s["ocr_text"])
    avg_conf = 0.0
    confs = [float(s["confidence"]) for s in sheets if s["confidence"] is not None]
    if confs:
        avg_conf = sum(confs) / len(confs)

    low_conf_words = []
    for s in sheets:
        raw = s["low_conf_words"]
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                low_conf_words.extend(parsed)
        except Exception:
            continue

    return jsonify(
        {
            "student_id": student["id"],
            "student_name": student["name"],
            "ocr_text": combined,
            "corrected_script": student["corrected_script"] or "",
            "avg_confidence": avg_conf,
            "low_conf_words": low_conf_words[:200],
            "requires_review": avg_conf < OCR_LOW_CONF_THRESHOLD or len(low_conf_words) > 0,
        }
    )


@app.route("/api/students/<int:student_id>/correct-script", methods=["POST"])
@login_required
def save_corrected_script(student_id):
    payload = request.json or {}
    corrected_script = (payload.get("corrected_script") or "").strip()
    if not corrected_script:
        return jsonify({"error": "corrected_script is required"}), 400

    db = get_db()
    row = db.execute(
        """SELECT s.id
           FROM students s
           JOIN exams e ON e.id = s.exam_id
           WHERE s.id=? AND e.teacher_id=?""",
        (student_id, session["teacher_id"]),
    ).fetchone()
    if not row:
        return jsonify({"error": "Student not found"}), 404

    for attempt in range(3):
        try:
            db.execute("UPDATE students SET corrected_script=? WHERE id=?", (corrected_script, student_id))
            db.commit()
            break
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < 2:
                time.sleep(0.2 * (attempt + 1))
                continue
            return jsonify({"error": "Database busy. Please retry save in a moment."}), 503
    return jsonify({"message": "Corrected script saved", "student_id": student_id})


# ── Rubrics / drafts / approvals ──────────────────────────────────────────────

@app.route("/api/exams/<int:exam_id>/rubrics", methods=["GET"])
@login_required
def get_rubrics(exam_id):
    get_exam_or_404(exam_id)
    db = get_db()
    rows = db.execute(
        """SELECT r.*, q.number as question_number
           FROM rubrics r
           JOIN questions q ON q.id = r.question_id
           WHERE r.exam_id=?
           ORDER BY q.rowid, r.version DESC""",
        (exam_id,),
    ).fetchall()
    templates = db.execute(
        "SELECT id, name, subject, chapter, template_json, created FROM rubric_templates WHERE teacher_id=? ORDER BY id DESC",
        (session["teacher_id"],),
    ).fetchall()
    return jsonify(
        {
            "rubrics": [
                {
                    **dict(r),
                    "criteria": safe_json_load(r["criteria_json"], {}),
                }
                for r in rows
            ],
            "templates": [
                {**dict(t), "template": safe_json_load(t["template_json"], {})}
                for t in templates
            ],
        }
    )


@app.route("/api/exams/<int:exam_id>/rubrics", methods=["POST"])
@login_required
def save_rubrics(exam_id):
    get_exam_or_404(exam_id)
    payload = request.json or {}
    rubrics = payload.get("rubrics", [])
    if not isinstance(rubrics, list) or not rubrics:
        return jsonify({"error": "rubrics array is required"}), 400

    db = get_db()
    questions = db.execute(
        "SELECT id FROM questions WHERE exam_id=?",
        (exam_id,),
    ).fetchall()
    valid_qids = {q["id"] for q in questions}
    if not valid_qids:
        return jsonify({"error": "No questions found for exam"}), 400

    for item in rubrics:
        qid = item.get("question_id")
        criteria = item.get("criteria", {})
        if qid not in valid_qids:
            continue
        prev = db.execute(
            "SELECT COALESCE(MAX(version),0) AS v FROM rubrics WHERE exam_id=? AND question_id=?",
            (exam_id, qid),
        ).fetchone()
        next_version = (prev["v"] or 0) + 1
        db.execute(
            "INSERT INTO rubrics (exam_id, question_id, criteria_json, version, created_by) VALUES (?, ?, ?, ?, ?)",
            (exam_id, qid, json.dumps(criteria), next_version, session["teacher_id"]),
        )

    if payload.get("save_as_template"):
        template_name = (payload.get("template_name") or "").strip()
        if not template_name:
            return jsonify({"error": "template_name is required when save_as_template=true"}), 400
        template_obj = {
            str(item.get("question_id")): item.get("criteria", {})
            for item in rubrics
            if item.get("question_id") in valid_qids
        }
        db.execute(
            """INSERT INTO rubric_templates (teacher_id, name, subject, chapter, template_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                session["teacher_id"],
                template_name,
                (payload.get("subject") or "").strip() or None,
                (payload.get("chapter") or "").strip() or None,
                json.dumps(template_obj),
            ),
        )
    db.commit()
    return jsonify({"message": "Rubrics saved", "count": len(rubrics)})


def _fallback_rubric(question_text: str, max_marks: float):
    text = (question_text or "").strip()
    short = text[:120] + ("..." if len(text) > 120 else "")
    return {
        "key_points": [f"Core concept from: {short}", "Correct method/steps", "Relevant final answer"],
        "penalties": ["Conceptual mistake", "Missing critical step", "Incorrect or no conclusion"],
        "partial_credit": "Award partial marks for correct concept even if final answer has minor errors.",
        "scoring_notes": f"Total marks available: {max_marks}.",
    }


def _generate_rubric_with_gemini(question_text: str, model_answer: str, max_marks: float, subject: str = "", chapter: str = ""):
    prompt = f"""You are an expert teacher preparing a grading rubric.

Subject: {subject or "General"}
Chapter: {chapter or "General"}
Question: {question_text}
Model Answer: {model_answer}
Max Marks: {max_marks}

Return ONLY valid JSON with these keys:
- key_points: array of concise expected answer points (3-6 items)
- penalties: array of common mistakes/penalties (2-5 items)
- partial_credit: short rule for partial marks
- scoring_notes: short practical note on mark distribution
"""
    raw = gemini_generate(prompt)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    obj = safe_json_load(raw, {})
    if not isinstance(obj, dict):
        return _fallback_rubric(question_text, max_marks), False
    rubric = {
        "key_points": obj.get("key_points") if isinstance(obj.get("key_points"), list) else [],
        "penalties": obj.get("penalties") if isinstance(obj.get("penalties"), list) else [],
        "partial_credit": str(obj.get("partial_credit") or "").strip(),
        "scoring_notes": str(obj.get("scoring_notes") or "").strip(),
    }
    # Normalize minimal defaults
    if not rubric["key_points"]:
        rubric["key_points"] = _fallback_rubric(question_text, max_marks)["key_points"]
    if not rubric["penalties"]:
        rubric["penalties"] = _fallback_rubric(question_text, max_marks)["penalties"]
    if not rubric["partial_credit"]:
        rubric["partial_credit"] = _fallback_rubric(question_text, max_marks)["partial_credit"]
    if not rubric["scoring_notes"]:
        rubric["scoring_notes"] = _fallback_rubric(question_text, max_marks)["scoring_notes"]
    return rubric, True


@app.route("/api/exams/<int:exam_id>/rubrics/auto-generate", methods=["POST"])
@login_required
def auto_generate_rubrics(exam_id):
    get_exam_or_404(exam_id)
    payload = request.json or {}
    subject = (payload.get("subject") or "").strip()
    chapter = (payload.get("chapter") or "").strip()
    persist = bool(payload.get("persist", False))
    question_ids = payload.get("question_ids")

    db = get_db()
    if isinstance(question_ids, list) and question_ids:
        placeholders = ",".join("?" for _ in question_ids)
        questions = db.execute(
            f"SELECT * FROM questions WHERE exam_id=? AND id IN ({placeholders}) ORDER BY rowid",
            [exam_id, *question_ids],
        ).fetchall()
    else:
        questions = db.execute(
            "SELECT * FROM questions WHERE exam_id=? ORDER BY rowid",
            (exam_id,),
        ).fetchall()

    if not questions:
        return jsonify({"error": "No questions found for rubric generation"}), 400

    generated = []
    for q in questions:
        try:
            criteria, used_ai = _generate_rubric_with_gemini(
                q["text"],
                q["model_answer"] or "",
                float(q["max_marks"] or 0),
                subject=subject,
                chapter=chapter,
            )
        except Exception:
            criteria, used_ai = _fallback_rubric(q["text"], float(q["max_marks"] or 0)), False
        generated.append(
            {
                "question_id": q["id"],
                "question_number": q["number"],
                "criteria": criteria,
                "source": "gemini" if used_ai else "fallback",
            }
        )

    if persist:
        for item in generated:
            qid = item["question_id"]
            prev = db.execute(
                "SELECT COALESCE(MAX(version),0) AS v FROM rubrics WHERE exam_id=? AND question_id=?",
                (exam_id, qid),
            ).fetchone()
            next_version = (prev["v"] or 0) + 1
            db.execute(
                "INSERT INTO rubrics (exam_id, question_id, criteria_json, version, created_by) VALUES (?, ?, ?, ?, ?)",
                (exam_id, qid, json.dumps(item["criteria"]), next_version, session["teacher_id"]),
            )
        db.commit()

    return jsonify(
        {
            "message": "Rubrics auto-generated",
            "persisted": persist,
            "rubrics": generated,
        }
    )


def _build_student_script(db, student):
    corrected = (student["corrected_script"] or "").strip() if "corrected_script" in student.keys() else ""
    if corrected:
        return corrected, 1.0

    sheets = db.execute(
        "SELECT ocr_text, confidence FROM answer_sheets WHERE student_id=? AND processed=1",
        (student["id"],),
    ).fetchall()
    full_script = "\n\n".join(s["ocr_text"] for s in sheets if s["ocr_text"])
    confs = [float(s["confidence"]) for s in sheets if s["confidence"] is not None]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return full_script, avg_conf


def _evaluate_with_rubric(question, model_answer, student_answer, rubric_obj):
    rubric_txt = json.dumps(rubric_obj, ensure_ascii=False)
    prompt = f"""You are an examiner marking with a rubric.

Question: {question['text']}
Maximum Marks: {question['max_marks']}
Model Answer: {model_answer}
Rubric JSON: {rubric_txt}
Student Answer: {student_answer}

Return ONLY valid JSON with keys:
- marks (number)
- reason (short explanation)
- confidence (0 to 1)
"""
    raw = gemini_generate(prompt)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    data = safe_json_load(raw, {})
    marks = min(float(data.get("marks", 0) or 0), float(question["max_marks"]))
    confidence = float(data.get("confidence", 0.65) or 0.65)
    reason = (data.get("reason") or "").strip() or "Marked using rubric."
    return {"marks": max(0.0, marks), "confidence": max(0.0, min(1.0, confidence)), "reason": reason}


@app.route("/api/exams/<int:exam_id>/marking-drafts/generate", methods=["POST"])
@login_required
def generate_marking_drafts(exam_id):
    get_exam_or_404(exam_id)
    db = get_db()
    questions = db.execute("SELECT * FROM questions WHERE exam_id=? ORDER BY rowid", (exam_id,)).fetchall()
    if not questions:
        return jsonify({"error": "No questions configured"}), 400
    students = db.execute("SELECT * FROM students WHERE exam_id=? ORDER BY id", (exam_id,)).fetchall()
    if not students:
        return jsonify({"error": "No students found"}), 400

    rubric_rows = db.execute(
        """SELECT r.question_id, r.criteria_json
           FROM rubrics r
           JOIN (
             SELECT question_id, MAX(version) AS max_v
             FROM rubrics
             WHERE exam_id=?
             GROUP BY question_id
           ) x ON x.question_id = r.question_id AND x.max_v = r.version
           WHERE r.exam_id=?""",
        (exam_id, exam_id),
    ).fetchall()
    rubric_map = {r["question_id"]: safe_json_load(r["criteria_json"], {}) for r in rubric_rows}
    missing = [q["number"] for q in questions if q["id"] not in rubric_map]
    if missing:
        return jsonify({"error": f"Rubric missing for questions: {', '.join(missing)}"}), 400

    draft_rows = []
    counts = {"pending": 0, "needs_review": 0, "blocked_ocr": 0, "blocked_ai": 0}
    for student in students:
        full_script, ocr_conf = _build_student_script(db, student)
        if not full_script.strip():
            for q in questions:
                draft_rows.append(
                    {
                        "student_id": student["id"],
                        "question_id": q["id"],
                        "status": "blocked_ocr",
                        "ai_reason": "No OCR/corrected script available.",
                        "ai_confidence": 0.0,
                        "student_answer": "",
                        "rubric_snapshot": json.dumps(rubric_map[q["id"]]),
                        "ai_marks": None,
                    }
                )
                counts["blocked_ocr"] += 1
            continue

        q_numbers = [canonical_question_number(q["number"]) for q in questions]
        segments = segment_script(full_script, q_numbers)
        for q in questions:
            q_key = canonical_question_number(q["number"])
            student_ans = (segments.get(q_key, "") or "").strip()
            model_ans = q["model_answer"] or ""
            if not student_ans:
                ai_marks = 0.0
                ai_reason = "No answer detected for this question."
                ai_conf = 0.9
                status = "pending" if ocr_conf >= OCR_LOW_CONF_THRESHOLD else "needs_review"
                counts[status] += 1
            else:
                result = None
                last_err = None
                try:
                    result = _evaluate_with_rubric(q, model_ans, student_ans, rubric_map[q["id"]])
                except Exception as exc:
                    last_err = exc
                if not result:
                    draft_rows.append(
                        {
                            "student_id": student["id"],
                            "question_id": q["id"],
                            "status": "blocked_ai",
                            "ai_reason": f"AI evaluation failed: {last_err}",
                            "ai_confidence": 0.0,
                            "student_answer": student_ans,
                            "rubric_snapshot": json.dumps(rubric_map[q["id"]]),
                            "ai_marks": None,
                        }
                    )
                    counts["blocked_ai"] += 1
                    continue
                ai_marks = result["marks"]
                ai_reason = result["reason"]
                ai_conf = result["confidence"]
                status = "needs_review" if (ocr_conf < OCR_LOW_CONF_THRESHOLD or ai_conf < 0.6) else "pending"
                counts[status] += 1

            draft_rows.append(
                {
                    "student_id": student["id"],
                    "question_id": q["id"],
                    "status": status,
                    "ai_reason": ai_reason,
                    "ai_confidence": ai_conf,
                    "student_answer": student_ans,
                    "rubric_snapshot": json.dumps(rubric_map[q["id"]]),
                    "ai_marks": ai_marks,
                }
            )

    def _writer():
        db.execute(
            """DELETE FROM marking_drafts
               WHERE student_id IN (SELECT id FROM students WHERE exam_id=?)""",
            (exam_id,),
        )
        for row in draft_rows:
            if row["ai_marks"] is None:
                db.execute(
                    """INSERT INTO marking_drafts
                       (student_id, question_id, status, ai_reason, ai_confidence, student_answer, rubric_snapshot)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["student_id"],
                        row["question_id"],
                        row["status"],
                        row["ai_reason"],
                        row["ai_confidence"],
                        row["student_answer"],
                        row["rubric_snapshot"],
                    ),
                )
            else:
                db.execute(
                    """INSERT INTO marking_drafts
                       (student_id, question_id, ai_marks, ai_reason, ai_confidence, status, student_answer, rubric_snapshot)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["student_id"],
                        row["question_id"],
                        row["ai_marks"],
                        row["ai_reason"],
                        row["ai_confidence"],
                        row["status"],
                        row["student_answer"],
                        row["rubric_snapshot"],
                    ),
                )

    err = run_db_write(db, _writer)
    if err:
        return jsonify({"error": "Database busy. Please retry draft generation."}), 503
    return jsonify({"message": "Marking drafts generated", "counts": counts})


@app.route("/api/exams/<int:exam_id>/review-queue", methods=["GET"])
@login_required
def get_review_queue(exam_id):
    get_exam_or_404(exam_id)
    db = get_db()
    rows = db.execute(
        """SELECT d.*, s.name as student_name, s.roll_no, q.number as question_number, q.max_marks
           FROM marking_drafts d
           JOIN students s ON s.id = d.student_id
           JOIN questions q ON q.id = d.question_id
           JOIN exams e ON e.id = s.exam_id
           WHERE e.id=? AND d.status IN ('needs_review','pending','blocked_ocr','blocked_ai')
           ORDER BY s.id, q.rowid""",
        (exam_id,),
    ).fetchall()
    return jsonify({"queue": [dict(r) for r in rows]})


@app.route("/api/marking-drafts/<int:draft_id>/approve", methods=["POST"])
@login_required
def approve_draft(draft_id):
    payload = request.json or {}
    db = get_db()
    draft = db.execute(
        """SELECT d.*, s.exam_id, s.id as sid
           FROM marking_drafts d
           JOIN students s ON s.id = d.student_id
           JOIN exams e ON e.id = s.exam_id
           WHERE d.id=? AND e.teacher_id=?""",
        (draft_id, session["teacher_id"]),
    ).fetchone()
    if not draft:
        return jsonify({"error": "Draft not found"}), 404
    if draft["status"] in {"blocked_ocr", "blocked_ai"}:
        return jsonify({"error": "Blocked drafts cannot be approved"}), 400

    teacher_marks = payload.get("teacher_marks")
    teacher_reason = (payload.get("teacher_reason") or "").strip()
    if teacher_marks is None:
        teacher_marks = draft["ai_marks"] if draft["ai_marks"] is not None else 0.0
    if not teacher_reason:
        teacher_reason = (draft["ai_reason"] or "").strip() or "Approved as suggested."

    def _writer():
        db.execute(
            """UPDATE marking_drafts
               SET teacher_marks=?, teacher_reason=?, status='approved', reviewed_at=datetime('now')
               WHERE id=?""",
            (float(teacher_marks), teacher_reason, draft_id),
        )
        db.execute(
            """INSERT INTO result_approvals (student_id, approved_by, approved_at, finalized)
               VALUES (?, ?, datetime('now'), 0)""",
            (draft["student_id"], session.get("username", "teacher")),
        )

    err = run_db_write(db, _writer)
    if err:
        return jsonify({"error": "Database busy. Please retry approve."}), 503
    return jsonify({"message": "Draft approved", "draft_id": draft_id})


@app.route("/api/exams/<int:exam_id>/approve-batch", methods=["POST"])
@login_required
def approve_batch(exam_id):
    get_exam_or_404(exam_id)
    payload = request.json or {}
    ids = payload.get("draft_ids")
    db = get_db()
    if ids and isinstance(ids, list):
        placeholders = ",".join("?" for _ in ids)
        drafts = db.execute(
            f"""SELECT d.*
                FROM marking_drafts d
                JOIN students s ON s.id = d.student_id
                WHERE s.exam_id=? AND d.id IN ({placeholders})""",
            [exam_id, *ids],
        ).fetchall()
    else:
        drafts = db.execute(
            """SELECT d.*
               FROM marking_drafts d
               JOIN students s ON s.id = d.student_id
               WHERE s.exam_id=? AND d.status IN ('pending','needs_review')""",
            (exam_id,),
        ).fetchall()
    approved = 0
    touched_students = set()

    def _writer():
        nonlocal approved, touched_students
        for d in drafts:
            db.execute(
                """UPDATE marking_drafts
                   SET teacher_marks=COALESCE(teacher_marks, ai_marks, 0),
                       teacher_reason=COALESCE(NULLIF(teacher_reason,''), ai_reason, 'Approved in batch'),
                       status='approved',
                       reviewed_at=datetime('now')
                   WHERE id=?""",
                (d["id"],),
            )
            touched_students.add(d["student_id"])
            approved += 1
        for sid in touched_students:
            db.execute(
                "INSERT INTO result_approvals (student_id, approved_by, approved_at, finalized) VALUES (?, ?, datetime('now'), 0)",
                (sid, session.get("username", "teacher")),
            )

    err = run_db_write(db, _writer)
    if err:
        return jsonify({"error": "Database busy. Please retry batch approve."}), 503
    return jsonify({"message": "Batch approved", "approved_count": approved})


@app.route("/api/exams/<int:exam_id>/finalize", methods=["POST"])
@login_required
def finalize_exam(exam_id):
    get_exam_or_404(exam_id)
    db = get_db()
    questions = db.execute("SELECT * FROM questions WHERE exam_id=? ORDER BY rowid", (exam_id,)).fetchall()
    students = db.execute("SELECT * FROM students WHERE exam_id=?", (exam_id,)).fetchall()
    if not questions or not students:
        return jsonify({"error": "Questions and students are required"}), 400

    finalized = []
    pending = []
    finalize_rows = []
    for student in students:
        drafts = db.execute(
            """SELECT d.*, q.number as q_number, q.max_marks, q.id as qid
               FROM marking_drafts d
               JOIN questions q ON q.id = d.question_id
               WHERE d.student_id=? AND q.exam_id=?
               ORDER BY q.rowid""",
            (student["id"], exam_id),
        ).fetchall()
        if len(drafts) != len(questions):
            pending.append({"student_id": student["id"], "student_name": student["name"], "reason": "Drafts missing"})
            continue
        not_approved = [d for d in drafts if d["status"] != "approved"]
        if not_approved:
            pending.append(
                {
                    "student_id": student["id"],
                    "student_name": student["name"],
                    "reason": f"{len(not_approved)} drafts pending approval",
                }
            )
            continue

        total = 0.0
        max_total = 0.0
        evals = []
        eval_inserts = []
        for d in drafts:
            marks = float(d["teacher_marks"] if d["teacher_marks"] is not None else d["ai_marks"] or 0.0)
            reason = d["teacher_reason"] or d["ai_reason"] or "Finalized"
            eval_inserts.append((student["id"], d["question_id"], d["student_answer"] or "", marks, reason))
            total += marks
            max_total += float(d["max_marks"] or 0.0)
            evals.append({"question": d["q_number"], "marks": marks, "max": d["max_marks"], "reason": reason})

        feedback_prompt = (
            f"Student scored {total}/{max_total}. Write 2-3 sentences constructive teacher feedback."
        )
        feedback = gemini_generate(feedback_prompt)
        finalize_rows.append(
            {
                "student_id": student["id"],
                "eval_inserts": eval_inserts,
                "total": total,
                "max_total": max_total,
                "feedback": feedback,
            }
        )
        finalized.append({"student_id": student["id"], "student_name": student["name"], "total": total, "max_total": max_total, "evaluations": evals})

    def _writer():
        for row in finalize_rows:
            db.execute("DELETE FROM evaluations WHERE student_id=?", (row["student_id"],))
            for ev in row["eval_inserts"]:
                db.execute(
                    """INSERT INTO evaluations (student_id, question_id, student_answer, marks_awarded, reason)
                       VALUES (?, ?, ?, ?, ?)""",
                    ev,
                )
            db.execute("DELETE FROM feedback_logs WHERE student_id=?", (row["student_id"],))
            db.execute(
                "INSERT INTO feedback_logs (student_id, total_marks, max_marks, feedback) VALUES (?, ?, ?, ?)",
                (row["student_id"], row["total"], row["max_total"], row["feedback"]),
            )
            db.execute(
                """INSERT INTO result_approvals (student_id, approved_by, approved_at, finalized, finalized_at)
                   VALUES (?, ?, datetime('now'), 1, datetime('now'))""",
                (row["student_id"], session.get("username", "teacher")),
            )

    err = run_db_write(db, _writer)
    if err:
        return jsonify({"error": "Database busy. Please retry finalize."}), 503
    return jsonify(
        {
            "message": "Finalize run complete",
            "finalized_count": len(finalized),
            "pending_count": len(pending),
            "finalized_students": finalized,
            "pending_students": pending,
        }
    )


@app.route("/api/exams/<int:exam_id>/students/import-csv", methods=["POST"])
@login_required
def import_students_csv(exam_id):
    get_exam_or_404(exam_id)
    csv_file = request.files.get("csv_file")
    if not csv_file or not csv_file.filename:
        return jsonify({"error": "csv_file is required"}), 400
    db = get_db()
    raw = csv_file.read()
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except Exception:
            text = None
    if text is None:
        return jsonify({"error": "Could not decode CSV file"}), 400

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return jsonify({"error": "CSV headers missing"}), 400
    headers = {h.strip().lower(): h for h in reader.fieldnames if h}
    if "name" not in headers:
        return jsonify({"error": "CSV must include 'name' header"}), 400

    created = 0
    updated = 0
    skipped = 0
    conflicts = []
    seen_roll = set()
    for idx, row in enumerate(reader, start=2):
        name = (row.get(headers["name"]) or "").strip()
        roll_no = (row.get(headers.get("roll_no", "")) or "").strip() or None
        class_name = (row.get(headers.get("class", "")) or row.get(headers.get("class_name", "")) or "").strip() or None
        if not name:
            skipped += 1
            conflicts.append({"row": idx, "issue": "Empty name"})
            continue
        if roll_no and roll_no in seen_roll:
            skipped += 1
            conflicts.append({"row": idx, "issue": f"Duplicate roll_no '{roll_no}' in file"})
            continue
        if roll_no:
            seen_roll.add(roll_no)
            existing = db.execute(
                "SELECT id FROM students WHERE exam_id=? AND roll_no=?",
                (exam_id, roll_no),
            ).fetchone()
            if existing:
                db.execute(
                    "UPDATE students SET name=?, class_name=? WHERE id=?",
                    (name, class_name, existing["id"]),
                )
                updated += 1
            else:
                db.execute(
                    "INSERT INTO students (exam_id, name, roll_no, class_name) VALUES (?, ?, ?, ?)",
                    (exam_id, name, roll_no, class_name),
                )
                created += 1
        else:
            db.execute(
                "INSERT INTO students (exam_id, name, roll_no, class_name) VALUES (?, ?, ?, ?)",
                (exam_id, name, roll_no, class_name),
            )
            created += 1
    db.commit()
    return jsonify({"created": created, "updated": updated, "skipped": skipped, "conflicts": conflicts})


@app.route("/api/exams/<int:exam_id>/results/export", methods=["GET"])
@login_required
def export_results(exam_id):
    get_exam_or_404(exam_id)
    export_format = (request.args.get("format") or "csv").lower()
    db = get_db()
    questions = db.execute("SELECT id, number FROM questions WHERE exam_id=? ORDER BY rowid", (exam_id,)).fetchall()
    students = db.execute("SELECT id, name, roll_no, class_name FROM students WHERE exam_id=? ORDER BY id", (exam_id,)).fetchall()
    if export_format not in {"csv", "pdf"}:
        return jsonify({"error": "format must be csv or pdf"}), 400

    rows = []
    for s in students:
        fb = db.execute("SELECT total_marks, max_marks FROM feedback_logs WHERE student_id=? ORDER BY id DESC LIMIT 1", (s["id"],)).fetchone()
        evals = db.execute("SELECT question_id, marks_awarded FROM evaluations WHERE student_id=?", (s["id"],)).fetchall()
        eval_map = {e["question_id"]: e["marks_awarded"] for e in evals}
        row = {
            "roll_no": s["roll_no"] or "",
            "name": s["name"],
            "class_name": s["class_name"] or "",
            "total_marks": fb["total_marks"] if fb else "",
            "max_marks": fb["max_marks"] if fb else "",
        }
        for q in questions:
            row[q["number"]] = eval_map.get(q["id"], "")
        rows.append(row)

    if export_format == "csv":
        output = io.StringIO()
        fieldnames = ["roll_no", "name", "class_name", "total_marks", "max_marks"] + [q["number"] for q in questions]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        filename = f"exam_{exam_id}_results.csv"
        return (
            output.getvalue(),
            200,
            {
                "Content-Type": "text/csv; charset=utf-8",
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    pdf_bytes = build_simple_results_pdf(exam_id, rows)
    filename = f"exam_{exam_id}_results.pdf"
    return (
        pdf_bytes,
        200,
        {
            "Content-Type": "application/pdf",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.route("/api/exams/<int:exam_id>/clone", methods=["POST"])
@login_required
def clone_exam(exam_id):
    exam = get_exam_or_404(exam_id)
    payload = request.json or {}
    new_title = (payload.get("title") or f"{exam['title']} (Copy)").strip()
    requested_code_raw = payload.get("exam_code")
    db = get_db()
    if requested_code_raw:
        new_code, err = validate_exam_code(requested_code_raw)
        if err:
            return jsonify({"error": err}), 400
        if not ensure_exam_code_unique(db, session["teacher_id"], new_code):
            return jsonify({"error": "Exam code already exists"}), 409
    else:
        base = normalize_exam_code(exam["exam_code"] or exam["title"] or f"EXAM-{exam_id}")
        base = re.sub(r"[^A-Z0-9-]+", "-", base).strip("-")
        if len(base) < 3:
            base = f"EXAM-{exam_id}"
        base_code = f"{base}-COPY"
        base_code = base_code[:24]
        if not EXAM_CODE_RE.fullmatch(base_code):
            base_code = "EXAM-COPY"
        new_code = base_code
        suffix = 2
        while not ensure_exam_code_unique(db, session["teacher_id"], new_code):
            suffix_txt = f"-{suffix}"
            stem = base_code[: max(3, 24 - len(suffix_txt))]
            new_code = f"{stem}{suffix_txt}"
            suffix += 1

    cur = db.execute(
        "INSERT INTO exams (teacher_id, title, exam_code, qp_path, qp_text) VALUES (?, ?, ?, ?, ?)",
        (session["teacher_id"], new_title, new_code, exam["qp_path"], exam["qp_text"]),
    )
    new_exam_id = cur.lastrowid
    old_questions = db.execute("SELECT * FROM questions WHERE exam_id=? ORDER BY rowid", (exam_id,)).fetchall()
    qid_map = {}
    for q in old_questions:
        qcur = db.execute(
            "INSERT INTO questions (exam_id, number, text, max_marks, model_answer) VALUES (?, ?, ?, ?, ?)",
            (new_exam_id, q["number"], q["text"], q["max_marks"], q["model_answer"]),
        )
        qid_map[q["id"]] = qcur.lastrowid

    old_rubrics = db.execute(
        """SELECT r.*
           FROM rubrics r
           JOIN (
             SELECT question_id, MAX(version) AS max_v
             FROM rubrics
             WHERE exam_id=?
             GROUP BY question_id
           ) x ON x.question_id = r.question_id AND x.max_v = r.version
           WHERE r.exam_id=?""",
        (exam_id, exam_id),
    ).fetchall()
    for r in old_rubrics:
        new_qid = qid_map.get(r["question_id"])
        if new_qid:
            db.execute(
                "INSERT INTO rubrics (exam_id, question_id, criteria_json, version, created_by) VALUES (?, ?, ?, 1, ?)",
                (new_exam_id, new_qid, r["criteria_json"], session["teacher_id"]),
            )
    db.commit()
    return jsonify({"message": "Exam cloned", "exam_id": new_exam_id, "title": new_title, "exam_code": new_code})


# ── Evaluation routes ─────────────────────────────────────────────────────────

@app.route("/api/exams/<int:exam_id>/evaluate", methods=["POST"])
@login_required
def evaluate_exam(exam_id):
    exam = get_exam_or_404(exam_id)
    db = get_db()
    questions = db.execute(
        "SELECT * FROM questions WHERE exam_id=?", (exam_id,)
    ).fetchall()
    if not questions:
        return jsonify({"error": "No questions configured"}), 400

    students = db.execute(
        "SELECT * FROM students WHERE exam_id=?", (exam_id,)
    ).fetchall()
    results = []

    for student in students:
        # Gather OCR text, or teacher-corrected script if present.
        corrected = (student["corrected_script"] or "").strip() if "corrected_script" in student.keys() else ""
        if corrected:
            full_script = corrected
        else:
            sheets = db.execute(
                "SELECT ocr_text FROM answer_sheets WHERE student_id=? AND processed=1",
                (student["id"],),
            ).fetchall()
            full_script = "\n\n".join(s["ocr_text"] for s in sheets if s["ocr_text"])

        q_numbers = [canonical_question_number(q["number"]) for q in questions]
        segments = segment_script(full_script, q_numbers)

        student_total = 0
        max_total = 0
        evals = []

        db.execute("DELETE FROM evaluations WHERE student_id=?", (student["id"],))

        for q in questions:
            q_num = q["number"]
            q_key = canonical_question_number(q_num)
            student_ans = segments.get(q_key, "")
            model_ans = q["model_answer"] or ""
            if not (student_ans or "").strip():
                marks = 0.0
                reason = "No answer provided for this question."
            else:
                try:
                    result = evaluate_answer(q["text"], model_ans, student_ans, q["max_marks"])
                except Exception as exc:
                    return jsonify({"error": str(exc)}), 502
                marks = result["marks"]
                reason = result["reason"]
            student_total += marks
            max_total += q["max_marks"]
            db.execute(
                "INSERT INTO evaluations (student_id, question_id, student_answer, marks_awarded, reason) VALUES (?,?,?,?,?)",
                (student["id"], q["id"], student_ans, marks, reason),
            )
            evals.append({
                "question": q_num,
                "marks": marks,
                "max": q["max_marks"],
                "reason": reason,
            })

        # Generate overall feedback
        feedback_prompt = (
            f"Student scored {student_total}/{max_total}. "
            f"Write 2-3 sentences of constructive feedback based on these results: "
            + "; ".join(f"{e['question']}: {e['marks']}/{e['max']}" for e in evals)
        )
        feedback = gemini_generate(feedback_prompt)

        db.execute(
            "INSERT OR REPLACE INTO feedback_logs (student_id, total_marks, max_marks, feedback) VALUES (?,?,?,?)",
            (student["id"], student_total, max_total, feedback),
        )
        db.commit()

        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "total_marks": student_total,
            "max_marks": max_total,
            "evaluations": evals,
            "feedback": feedback,
        })

    return jsonify({"results": results})


@app.route("/api/exams/<int:exam_id>/results", methods=["GET"])
@login_required
def get_results(exam_id):
    get_exam_or_404(exam_id)
    db = get_db()
    students = db.execute(
        "SELECT * FROM students WHERE exam_id=?", (exam_id,)
    ).fetchall()
    results = []
    for student in students:
        fb = db.execute(
            "SELECT * FROM feedback_logs WHERE student_id=? ORDER BY id DESC LIMIT 1",
            (student["id"],),
        ).fetchone()
        evals = db.execute(
            """SELECT e.*, q.number, q.text, q.max_marks
               FROM evaluations e JOIN questions q ON e.question_id=q.id
               WHERE e.student_id=? ORDER BY q.rowid""",
            (student["id"],),
        ).fetchall()
        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "total_marks": fb["total_marks"] if fb else None,
            "max_marks": fb["max_marks"] if fb else None,
            "feedback": fb["feedback"] if fb else None,
            "evaluations": [dict(e) for e in evals],
        })
    return jsonify({"results": results})


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return redirect(url_for("app_page"))


@app.route("/login")
def login_page():
    force = request.args.get("force", "").lower() in {"1", "true", "yes"}
    if "teacher_id" in session and not force:
        return redirect(url_for("app_page"))
    return render_template("login.html")


@app.route("/register")
def register_page():
    force = request.args.get("force", "").lower() in {"1", "true", "yes"}
    if "teacher_id" in session and not force:
        return redirect(url_for("app_page"))
    return render_template("register.html")


@app.route("/app")
@page_login_required
def app_page():
    return render_template("index.html", username=session.get("username", "Teacher"))


@app.route("/logout")
def logout_page():
    session.clear()
    return redirect(url_for("login_page"))


# ── Boot ──────────────────────────────────────────────────────────────────────

init_db()

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, debug=debug_mode, port=port)
