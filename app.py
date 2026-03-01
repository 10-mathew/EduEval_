import os
import json
import re
import sqlite3
from pathlib import Path
from functools import wraps

from flask import (Flask, request, jsonify, session, redirect,
                   url_for, render_template, g, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import google.generativeai as genai
from google.cloud import vision

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR_Q = UPLOAD_DIR / "questions"
UPLOAD_DIR_A = UPLOAD_DIR / "answers"

for d in (UPLOAD_DIR_Q, UPLOAD_DIR_A):
    d.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".tiff", ".bmp", ".webp"}

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-production-please")

# Google / Gemini keys — set in environment
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GOOGLE_CLOUD_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
OCR_LOW_CONF_THRESHOLD = float(os.environ.get("OCR_LOW_CONF_THRESHOLD", "0.75"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Database ───────────────────────────────────────────────────────────────────

DB_PATH = BASE_DIR / "instance" / "edueval.db"
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
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
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
        db.executescript(SCHEMA)
        # Lightweight forward-only migrations for existing local DBs.
        existing_student_cols = {r["name"] for r in db.execute("PRAGMA table_info(students)").fetchall()}
        if "corrected_script" not in existing_student_cols:
            db.execute("ALTER TABLE students ADD COLUMN corrected_script TEXT")
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
        return ocr_image(file_path)
    return mock_ocr(file_path)


# ── Gemini helpers ────────────────────────────────────────────────────────────

def gemini_generate(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return _mock_gemini(prompt)
    model_names = [GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_error = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Gemini generate failed: {last_error}")


def _mock_gemini(prompt: str) -> str:
    if "evaluate" in prompt.lower() or "mark" in prompt.lower():
        return json.dumps({"marks": 7, "reason": "Good understanding of key concepts. Student correctly identified main points but missed some detail on the mechanism."})
    if "model answer" in prompt.lower():
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

def segment_script(script_text: str, question_numbers: list) -> dict:
    """Split a student script by question markers, return {q_number: answer_text}."""
    text = (script_text or "").replace("\r", "\n")
    if not text.strip():
        return {}

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

    for i, m in enumerate(matches):
        raw_no = m.group(1) or m.group(2)
        if not raw_no:
            continue
        q_num = f"Q{raw_no.strip()}"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        ans = text[start:end].strip()
        if ans:
            segments[q_num.upper()] = ans

    # Fallback: no markers found. Keep only first question mapped.
    if not segments and question_numbers:
        segments[str(question_numbers[0]).upper()] = text.strip()

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
    return jsonify([dict(r) for r in rows])


@app.route("/api/exams", methods=["POST"])
@login_required
def create_exam():
    title = (request.form.get("title") or "").strip()
    if not title:
        return jsonify({"error": "Title required"}), 400
    qp_path = None
    qp_text = None
    if "question_paper" in request.files:
        f = request.files["question_paper"]
        if f.filename:
            if not allowed_file(f.filename):
                return jsonify({"error": "Unsupported question paper file type"}), 400
            qp_path = save_upload(f, UPLOAD_DIR_Q)
    db = get_db()
    cur = db.execute(
        "INSERT INTO exams (teacher_id, title, qp_path) VALUES (?, ?, ?)",
        (session["teacher_id"], title, qp_path),
    )
    db.commit()
    return jsonify({"id": cur.lastrowid, "title": title})


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
    return jsonify({
        **dict(exam),
        "questions": [dict(q) for q in questions],
        "students": [dict(s) for s in students],
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
            db.execute("DELETE FROM answer_sheets WHERE student_id=?", (sid,))

        db.execute("DELETE FROM students WHERE exam_id=?", (exam_id,))
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
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502
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

    try:
        answers = generate_model_answers(q_list)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

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
        }
    )


# ── Student / answer-sheet routes ─────────────────────────────────────────────

@app.route("/api/exams/<int:exam_id>/students", methods=["POST"])
@login_required
def add_student(exam_id):
    get_exam_or_404(exam_id)
    data = request.form
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Student name required"}), 400
    db = get_db()
    cur = db.execute(
        "INSERT INTO students (exam_id, name) VALUES (?, ?)", (exam_id, name)
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

    db.execute("UPDATE students SET corrected_script=? WHERE id=?", (corrected_script, student_id))
    db.commit()
    return jsonify({"message": "Corrected script saved", "student_id": student_id})


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

        q_numbers = [q["number"] for q in questions]
        segments = segment_script(full_script, q_numbers)

        student_total = 0
        max_total = 0
        evals = []

        db.execute("DELETE FROM evaluations WHERE student_id=?", (student["id"],))

        for q in questions:
            q_num = q["number"]
            student_ans = segments.get(q_num, "")
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

@app.route("/")
def index():
    return redirect(url_for("app_page"))


@app.route("/login")
def login_page():
    if "teacher_id" in session:
        return redirect(url_for("app_page"))
    return render_template("login.html")


@app.route("/register")
def register_page():
    if "teacher_id" in session:
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
    app.run(debug=True, port=5000)
