# 🧭 PathPilot AI - AI Career Roadmap & Skill Gap Finder

A clean, beginner-friendly web application designed for B.Tech Computer Science & Artificial Intelligence students and early-career learners. **PathPilot AI** analyzes your current programming skills, identifies missing competencies for your target dream job, and generates a personalized **8-week study plan** with **3 portfolio projects** and **PDF export**.

---

## 📌 Project Overview

- **Purpose:** College Project / Assignment in Python, Streamlit, and Generative AI.
- **Tech Stack:**
  - **Python 3.10+** (Core programming language)
  - **Streamlit** (Interactive frontend web framework)
  - **Google GenAI SDK (`google-genai`)** (Official Gemini API client)
  - **python-dotenv** (Secure environment variable management)
  - **fpdf2** (Lightweight, pure-Python PDF generation)
- **Database:** None (State managed cleanly in-memory via `st.session_state`).
- **Authentication:** None (Direct, single-user student desktop application).

---

## ✨ Features

1. **Intelligent Skill Gap Finder:**
   - Compares your current skills against industry requirements for top tech roles.
   - Categorizes missing skills by priority (**High**, **Medium**, **Low**) with explanations of *why* each skill is necessary.
   - **Guaranteed No-Overlap Rule:** Skills you already possess are never listed as skills you need to learn.

2. **Calibrated 8-Week Learning Roadmap:**
   - Exactly 8 week-by-week learning blocks calibrated for your weekly study hours and experience level (**Beginner**, **Intermediate**, or **Advanced**).
   - Each week provides concrete topics to study and an actionable, hands-on coding task.

3. **Recruiter-Ready Portfolio Projects:**
   - Recommends exactly 3 mini-projects tailored to your target role.
   - Shows project title, description, difficulty level, and recommended tech stack.

4. **One-Click PDF Export:**
   - Generates a formatted PDF report with your complete roadmap, skill comparison, and project ideas.

5. **Regenerate & Reset Controls:**
   - **🔄 Regenerate:** Create an alternate roadmap variation with the same inputs.
   - **✨ Start Again:** Reset all form fields and start fresh.

---

## 🚀 Setup & Installation (Step-by-Step)

Follow these simple steps in your terminal or VS Code to run the project:

### Step 1: Open the Project Directory
```bash
cd PathPilot-AI
```

### Step 2: (Recommended) Create & Activate a Virtual Environment
```bash
# On Windows (PowerShell or Command Prompt):
python -m venv venv
.\venv\Scripts\activate

# On macOS or Linux:
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Your Gemini API Key
1. Get a free API key from **[Google AI Studio](https://aistudio.google.com/app/apikey)**.
2. In the `PathPilot-AI` folder, copy `.env.example` and rename it to `.env`:
   ```bash
   # On Windows PowerShell:
   Copy-Item .env.example .env

   # On macOS / Linux:
   cp .env.example .env
   ```
3. Open `.env` in any text editor and paste your key:
   ```env
   GEMINI_API_KEY=AIzaSyYourActualKeyHere
   ```
   > 🔒 **Security Note:** Never commit your `.env` file or share your API key. The `.gitignore` file automatically protects it from Git commits.

### Step 5: Run the Application
```bash
streamlit run app.py
```
Streamlit will automatically open `http://localhost:8501` in your web browser!

---

## 🔍 How It Works: Data Flow & Architecture

Here is the exact data lifecycle to explain to your college faculty or evaluators:

```
+-------------------------------------------------------------------+
| 1. USER INPUT                                                    |
|    - Current skills: e.g. "C, basic Python, HTML"                 |
|    - Target role: "AI Engineer"                                   |
|    - Experience level: "Beginner"                                 |
|    - Available hours: 10 hrs/week                                 |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 2. PROMPT CONSTRUCTION (`build_prompt()`)                         |
|    - Formulates system role and strict rules.                     |
|    - Enforces JSON-only output with required keys:                |
|      * skills_have, skills_to_learn (High/Med/Low)                |
|      * exactly 8 roadmap weeks                                    |
|      * exactly 3 portfolio projects                               |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 3. GEMINI API CALL (`call_gemini()`)                              |
|    - Uses official `google-genai` SDK with `gemini-3.8-flash`.    |
|    - Configures `response_mime_type="application/json"`.          |
|    - Safely catches network/quota errors with friendly alerts.   |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 4. STRICT VALIDATION (`validate_roadmap_data()`)                  |
|    - Checks that response is a valid dictionary.                  |
|    - Verifies len(roadmap) == 8 and len(projects) == 3.          |
|    - Checks all required fields (topic, task, tech stack, etc.).  |
|    - Validates priorities (High / Medium / Low).                  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
| 5. INTERACTIVE UI & PDF (`show_results()`, `generate_pdf()`)      |
|    - Renders comparison tables with `st.dataframe`.               |
|    - Renders expandable weekly roadmap with `st.expander`.        |
|    - Displays 3 project cards with `st.columns(3)`.               |
|    - Generates downloadable PDF report via `fpdf2`.               |
+-------------------------------------------------------------------+
```

---

## 📂 Code Structure & Function Breakdown

The entire app is contained inside `app.py` in clean, modular functions:

| Function | What It Does | Why It's Needed |
| :--- | :--- | :--- |
| `get_api_key()` | Reads API key from `.env` or `st.secrets`. | Keeps credentials secure and portable without hardcoding. |
| `build_prompt(...)` | Takes the 4 student inputs and builds a prompt. | Guides Gemini to produce accurate, structured analysis. |
| `call_gemini(...)` | Calls `genai.Client().models.generate_content(...)`. | Interacts with the real Gemini model safely with `try/except`. |
| `validate_roadmap_data(...)` | Checks that JSON has 8 weeks, 3 projects, and required keys. | Protects the frontend from malformed or incomplete AI responses. |
| `sanitize_for_pdf(...)` | Replaces smart quotes and unicode dashes with standard characters. | Prevents font encoding crashes in PDF generation. |
| `generate_pdf(...)` | Uses `fpdf2` to construct a multi-page PDF document. | Allows students to download and print their roadmap. |
| `show_results(...)` | Renders the tables, expanders, cards, and buttons. | Creates a clean, modern user experience. |
| `main()` | Sets up page config, manages input forms, and coordinates flow. | Central controller of the Streamlit application. |

---

## 🛠️ Changing the Gemini Model

The model name is defined as a constant at the top of `app.py`:

```python
DEFAULT_MODEL: str = "gemini-3.8-flash"
```

To switch models (e.g. to `gemini-2.5-flash`, `gemini-2.5-pro`, or `gemini-1.5-flash`), simply change this string.

---

## ❓ Frequently Asked Questions (For Viva / Demo)

1. **Why use `response_mime_type="application/json"`?**
   - It instructs Gemini at the decoding layer to only emit syntactically valid JSON, eliminating markdown noise (like ` ```json `) and ensuring the response can be directly parsed with `json.loads()`.

2. **Why use `fpdf2` instead of heavy PDF libraries?**
   - `fpdf2` is written in 100% pure Python with zero C-dependencies, making it fast, cross-platform, and completely reliable on Windows, Mac, and Linux.

3. **How does the app remember my roadmap when I download the PDF?**
   - Streamlit reruns scripts on interaction. We use Streamlit's `st.session_state["roadmap_data"]` so that the generated roadmap is preserved in memory and doesn't disappear when you click buttons.

