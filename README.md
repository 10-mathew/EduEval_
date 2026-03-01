# EduEval — Automated Exam Marking System

AI-powered exam marking: upload question papers + handwritten answer sheets → get marks and feedback automatically.

---

## Features

- **Teacher auth** — register / login / session management
- **Exam management** — create exams, upload question papers
- **Question parsing** — manual entry or auto-parse from pasted text
- **Gemini model answers** — AI-generated marking guides per question
- **Google Vision OCR** — handwriting recognition with confidence scoring
- **Gemini evaluation** — per-question marks + reasons vs model answer
- **Results dashboard** — totals, pass/fail, per-question breakdown, AI feedback
- **SQLite persistence** — full database of exams, questions, students, evaluations

---

## Quick Start

### 1. Install dependencies

```bash
cd edueval
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# Required for Gemini AI
export GEMINI_API_KEY="your-gemini-api-key"

# Required for Google Cloud Vision OCR
# Download your service account JSON from GCP Console
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Optional — change for production
export SECRET_KEY="your-flask-secret-key"
```

**Getting API keys:**
- **Gemini**: https://aistudio.google.com/app/apikey
- **Google Cloud Vision**: Enable the Vision API in GCP Console, create a service account, download the JSON key.

> **No API keys?** The app runs in **demo/mock mode** — OCR returns sample text and Gemini returns canned responses, so you can explore the full flow without any credentials.

### 3. Run

```bash
python app.py
```

Open http://localhost:5000/login in your browser.

---

## User Flow

```
1. Login
2. Create Exam  →  upload question paper
3. Configure Questions  →  parse or enter manually with max marks
4. Generate Model Answers  →  Gemini creates marking guides
5. Add Students  →  upload scanned answer sheets per student
6. Run OCR  →  Google Cloud Vision extracts handwritten text
7. Evaluate  →  Gemini compares student answers vs model answers
8. View Results  →  marks, reasons, feedback, pass/fail
```

---

## Project Structure

```
edueval/
├── app.py               # Flask application (all backend logic)
├── requirements.txt
├── README.md
├── instance/
│   └── edueval.db       # SQLite database (auto-created)
├── uploads/
│   ├── questions/       # Uploaded question papers
│   └── answers/         # Uploaded answer sheets
└── templates/
    ├── login.html       # Login page
    └── index.html       # Main app UI (requires session)
```

---

## Database Schema

| Table           | Purpose                                         |
|-----------------|-------------------------------------------------|
| `teachers`      | Auth: username + hashed password                |
| `exams`         | Exam metadata + optional QP file path/text      |
| `questions`     | Questions per exam: number, text, max marks, model answer |
| `students`      | Student name linked to exam                     |
| `answer_sheets` | Uploaded files + OCR text + confidence          |
| `evaluations`   | Per-question marks + reason per student         |
| `feedback_logs` | Overall feedback + total marks per student      |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/login` | Login page |
| GET | `/app` | Main app page (auth required) |
| POST | `/api/login` | Authenticate |
| POST | `/api/logout` | End session |
| GET | `/api/me` | Session status |
| POST | `/api/register` | Create teacher account |
| GET | `/api/exams` | List all exams |
| POST | `/api/exams` | Create exam (multipart: title + question_paper) |
| GET | `/api/exams/:id` | Exam details with questions + students |
| POST | `/api/exams/:id/questions` | Save questions list |
| POST | `/api/exams/:id/generate-model-answers` | Gemini model answers |
| POST | `/api/exams/:id/students` | Add student + upload sheets |
| POST | `/api/students/:id/ocr` | Run Vision OCR on student sheets |
| POST | `/api/exams/:id/evaluate` | Evaluate all students with Gemini |
| GET | `/api/exams/:id/results` | Fetch stored results |

---

## Production Checklist

- [ ] Set `SECRET_KEY` to a long random string
- [ ] Use PostgreSQL/MySQL instead of SQLite for concurrent access
- [ ] Store uploads on S3/GCS instead of local filesystem
- [ ] Add HTTPS (nginx + certbot)
- [ ] Rate-limit API endpoints
- [ ] Add password reset / email verification
- [ ] Add manual review UI for low-confidence OCR regions
