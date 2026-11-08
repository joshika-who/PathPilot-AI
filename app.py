"""
PathPilot AI - AI Career Roadmap & Skill Gap Finder
A beginner-friendly Streamlit web application powered by the Google Gemini API.
Designed for B.Tech CSE (AI) students and early-career learners.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
from google import genai
from google.genai import types

# Load environment variables from .env file (if present)
load_dotenv()

# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================
# The Gemini model to use. You can easily switch this to any supported model.
DEFAULT_MODEL: str = "gemini-3.8-flash"

# Target role options for the dropdown
ROLE_OPTIONS = [
    "AI Engineer",
    "Machine Learning Engineer",
    "Data Scientist",
    "Data Analyst",
    "Web Developer",
    "Full Stack Developer",
    "Cybersecurity Analyst",
    "Cloud Engineer",
    "Software Developer",
    "Other",
]

# Experience level options
EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced"]


# ==============================================================================
# 1. API KEY RETRIEVAL
# ==============================================================================
def get_api_key() -> Optional[str]:
    """
    Safely retrieves the Gemini API key from environment variables or Streamlit secrets.
    Never hardcodes keys. Checks:
      1. os.getenv("GEMINI_API_KEY") (from .env file via python-dotenv)
      2. st.secrets["GEMINI_API_KEY"] (for Streamlit Cloud deployment)
    """
    # Check operating system / .env variable first
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key.strip() and api_key.strip() != "your_key_here":
        return api_key.strip()

    # Check Streamlit Cloud secrets as a secondary source
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = st.secrets["GEMINI_API_KEY"]
            if secret_key and secret_key.strip() and secret_key.strip() != "your_key_here":
                return secret_key.strip()
    except Exception:
        pass

    return None


# ==============================================================================
# 2. PROMPT BUILDER
# ==============================================================================
def build_prompt(
    current_skills: str,
    target_role: str,
    experience_level: str,
    hours_per_week: int,
) -> str:
    """
    Constructs a detailed, structured prompt for Gemini.
    Instructs the model to perform a skill-gap analysis, create an 8-week roadmap,
    and suggest 3 practical portfolio projects, outputting strictly valid JSON.
    """
    prompt = f"""
You are an expert career counselor, senior software engineering mentor, and technical curriculum designer.
Analyze the student's profile below and generate a realistic, high-quality, personalized career roadmap.

STUDENT PROFILE:
- Target Job Role: {target_role}
- Experience Level: {experience_level}
- Current Skills: {current_skills}
- Available Study Time: {hours_per_week} hours per week

STRICT INSTRUCTIONS:
1. "skills_have": Analyze the student's current skills and explain their relevance to the target role.
2. "skills_to_learn": Identify missing skills required for the role.
   - CRITICAL RULE: NEVER list any skill the student already possesses as a skill to learn!
   - Every skill to learn MUST have priority strictly set to "High", "Medium", or "Low".
   - Include a concise "why" explanation for each.
3. "roadmap": Create an EXACTLY 8-week week-by-week learning plan (Week 1 through Week 8).
   - The roadmap must be realistic and calibrated for {hours_per_week} hours/week at a {experience_level} level.
   - Each week must include:
     * "week": integer from 1 to 8
     * "topic": concise theme of the week
     * "what_to_learn": concrete concepts and libraries/tools to master
     * "practice_task": a hands-on mini coding exercise or practical task to build competence
4. "projects": Recommend EXACTLY 3 practical portfolio mini-projects that will impress recruiters.
   - Each project must include "title", "description", "technologies", and "difficulty" (Beginner, Intermediate, or Advanced).
5. Output format: You MUST return ONLY a single, valid JSON object matching the schema below. No markdown backticks, no preamble, no explanations outside the JSON.

REQUIRED JSON SCHEMA:
{{
  "skills_have": [
    {{ "skill": "Skill Name", "relevance": "Why this skill helps in the target role" }}
  ],
  "skills_to_learn": [
    {{ "skill": "Skill Name", "priority": "High", "why": "Why this is a must-learn gap" }}
  ],
  "roadmap": [
    {{
      "week": 1,
      "topic": "Topic Title",
      "what_to_learn": "Core concepts and modules to study",
      "practice_task": "Specific actionable exercise for the week"
    }}
  ],
  "projects": [
    {{
      "title": "Project Title",
      "description": "What the project does and its real-world purpose",
      "technologies": "Languages, frameworks, and APIs used",
      "difficulty": "Beginner"
    }}
  ]
}}
"""
    return prompt.strip()


# ==============================================================================
# 3. GEMINI API CALL WITH SAFE ERROR HANDLING
# ==============================================================================
def call_gemini(api_key: str, prompt: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Calls the official Google GenAI SDK to generate content using Gemini.
    Configures response_mime_type='application/json' to guarantee valid JSON.
    Returns: (parsed_json_dict, error_message)
    """
    try:
        # Initialize official Google GenAI client
        client = genai.Client(api_key=api_key)

        # Request generation with structured JSON configuration
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3,  # Low temperature for deterministic, structured output
            ),
        )

        # Check if response text was returned
        if not response.text:
            return None, "Gemini returned an empty response. Please try again."

        # Parse JSON
        raw_text = response.text.strip()
        # Clean potential markdown wrapping if returned despite mime type
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*\n", "", raw_text)
            raw_text = re.sub(r"\n```$", "", raw_text)

        parsed_data = json.loads(raw_text)
        return parsed_data, None

    except json.JSONDecodeError as jde:
        return None, f"Failed to parse AI response as JSON: {str(jde)}"
    except Exception as e:
        error_str = str(e)
        # Give helpful hints for common API issues
        if "API_KEY_INVALID" in error_str or "400" in error_str:
            return None, "Invalid Gemini API Key. Please verify your key in the .env file."
        elif "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
            return None, "Gemini API rate limit or quota exceeded. Please wait a minute and retry."
        elif "UNAVAILABLE" in error_str or "503" in error_str:
            return None, f"The model '{DEFAULT_MODEL}' is currently experiencing high demand at Google. Please wait a few moments and click 'Regenerate'."
        elif "NOT_FOUND" in error_str:
            return None, f"Model '{DEFAULT_MODEL}' was not found. Please verify model name support for your key."
        return None, f"Gemini API Error: {error_str}"


# ==============================================================================
# 4. JSON VALIDATION (REQUIREMENT 2)
# ==============================================================================
def validate_roadmap_data(data: Any) -> Tuple[bool, str]:
    """
    Validates the structure and content of the AI's response:
    - Verifies top-level keys exist (skills_have, skills_to_learn, roadmap, projects).
    - Ensures roadmap contains exactly 8 entries.
    - Ensures projects contains exactly 3 entries.
    - Validates required fields within each item.
    - Ensures priorities are strictly High, Medium, or Low.
    Returns: (is_valid, error_message)
    """
    if not isinstance(data, dict):
        return False, "Response is not a valid JSON dictionary."

    # Check top-level keys
    required_keys = ["skills_have", "skills_to_learn", "roadmap", "projects"]
    for key in required_keys:
        if key not in data:
            return False, f"Missing required top-level section: '{key}'"

    # Validate skills_have
    if not isinstance(data["skills_have"], list):
        return False, "'skills_have' must be a list."
    for idx, item in enumerate(data["skills_have"]):
        if not isinstance(item, dict) or "skill" not in item:
            return False, f"Invalid item in 'skills_have' at index {idx}."

    # Validate skills_to_learn
    valid_priorities = {"HIGH", "MEDIUM", "LOW"}
    if not isinstance(data["skills_to_learn"], list):
        return False, "'skills_to_learn' must be a list."
    for idx, item in enumerate(data["skills_to_learn"]):
        if not isinstance(item, dict) or "skill" not in item:
            return False, f"Invalid item in 'skills_to_learn' at index {idx}."
        priority = str(item.get("priority", "Medium")).strip().upper()
        if priority not in valid_priorities:
            item["priority"] = "Medium"
        else:
            item["priority"] = priority.capitalize()

    # Validate roadmap: EXACTLY 8 weeks
    roadmap = data["roadmap"]
    if not isinstance(roadmap, list):
        return False, "'roadmap' must be a list."
    if len(roadmap) != 8:
        return False, f"The roadmap must contain exactly 8 weeks, but received {len(roadmap)}."

    roadmap_keys = ["week", "topic", "what_to_learn", "practice_task"]
    for idx, week_item in enumerate(roadmap):
        if not isinstance(week_item, dict):
            return False, f"Roadmap entry for week {idx+1} is not an object."
        for rk in roadmap_keys:
            if rk not in week_item or not str(week_item[rk]).strip():
                return False, f"Roadmap week {idx+1} is missing the required field '{rk}'."

    # Validate projects: EXACTLY 3 projects
    projects = data["projects"]
    if not isinstance(projects, list):
        return False, "'projects' must be a list."
    if len(projects) != 3:
        return False, f"Must provide exactly 3 portfolio projects, but received {len(projects)}."

    project_keys = ["title", "description", "technologies", "difficulty"]
    for idx, proj in enumerate(projects):
        if not isinstance(proj, dict):
            return False, f"Project at index {idx+1} is not an object."
        for pk in project_keys:
            if pk not in proj or not str(proj[pk]).strip():
                return False, f"Project {idx+1} is missing the required field '{pk}'."

    return True, ""


# ==============================================================================
# 5. PDF GENERATION (REQUIREMENT 3)
# ==============================================================================
def sanitize_for_pdf(text: str) -> str:
    """
    Cleans up smart quotes, unicode dashes, and characters unsupported by standard
    Latin-1 PDF fonts to prevent encoding errors.
    """
    if not text:
        return ""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2014": " - ",
        "\u2013": " - ",
        "\u2022": "*",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Filter to printable Latin-1 characters
    return text.encode("latin-1", "replace").decode("latin-1")


class PDFReport(FPDF):
    """Custom FPDF class with a header, footer, and styling."""

    def header(self):
        # Primary top banner
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(24, 43, 73)
        self.cell(0, 8, "PathPilot AI - Career Roadmap & Skill Gap Report", new_x="LMARGIN", new_y="NEXT", align="L")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.6)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        page_str = f"Page {self.page_no()}"
        self.cell(0, 8, f"PathPilot AI | Generated for Academic Career Planning | {page_str}", align="C")


def generate_pdf(data: Dict[str, Any], inputs: Dict[str, Any]) -> bytes:
    """
    Creates a clean, downloadable PDF containing:
    1. Student Profile & Target Role
    2. Skills Analysis (Skills I Have vs Skills to Learn)
    3. Complete 8-Week Week-by-Week Roadmap
    4. 3 Portfolio Mini-Projects
    Returns: PDF content as raw bytes.
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Meta information box
    pdf.set_fill_color(240, 245, 255)
    pdf.set_draw_color(200, 220, 250)
    pdf.rect(pdf.get_x(), pdf.get_y(), pdf.w - 2 * pdf.l_margin, 24, "DF")

    pdf.set_xy(pdf.get_x() + 4, pdf.get_y() + 3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(20, 40, 80)
    pdf.cell(0, 6, f"Target Role: {sanitize_for_pdf(inputs['target_role'])}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(pdf.l_margin + 4)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    meta_line = (
        f"Experience: {inputs['experience_level']}  |  "
        f"Study Commitment: {inputs['hours_per_week']} hrs/week  |  "
        f"Generated: {datetime.now().strftime('%d %b %Y')}"
    )
    pdf.cell(0, 5, sanitize_for_pdf(meta_line), new_x="LMARGIN", new_y="NEXT")

    pdf.set_x(pdf.l_margin + 4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, f"Current Skills: {sanitize_for_pdf(inputs['current_skills'])}", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)

    # --------------------------------------------------------------------------
    # SECTION 1: SKILL GAP ANALYSIS
    # --------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "1. Skill Gap Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Skills You Have
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(0, 6, "Skills You Already Have:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for item in data.get("skills_have", []):
        line = f"- {item.get('skill', '')}: {item.get('relevance', '')}"
        pdf.multi_cell(0, 5, sanitize_for_pdf(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.ln(2)

    # Skills to Learn
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(180, 83, 9)
    pdf.cell(0, 6, "Skills to Learn (Ranked by Priority):", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for item in data.get("skills_to_learn", []):
        prio = item.get("priority", "Medium")
        line = f"- [{prio} Priority] {item.get('skill', '')}: {item.get('why', '')}"
        pdf.multi_cell(0, 5, sanitize_for_pdf(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    pdf.ln(4)

    # --------------------------------------------------------------------------
    # SECTION 2: 8-WEEK ROADMAP
    # --------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "2. 8-Week Personalized Learning Roadmap", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for item in data.get("roadmap", []):
        week_num = item.get("week", "")
        topic = item.get("topic", "")
        what = item.get("what_to_learn", "")
        task = item.get("practice_task", "")

        # Week Header box
        pdf.set_fill_color(243, 244, 246)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(17, 24, 39)
        pdf.cell(0, 6, sanitize_for_pdf(f"Week {week_num}: {topic}"), fill=True, new_x="LMARGIN", new_y="NEXT")

        # Content
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(55, 65, 81)
        pdf.multi_cell(0, 5, sanitize_for_pdf(f"What to Learn: {what}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(31, 41, 55)
        pdf.multi_cell(0, 5, sanitize_for_pdf(f"Practice Task: {task}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    pdf.ln(2)

    # --------------------------------------------------------------------------
    # SECTION 3: 3 PORTFOLIO PROJECTS
    # --------------------------------------------------------------------------
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "3. Recommended Portfolio Projects", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    for idx, proj in enumerate(data.get("projects", []), start=1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(29, 78, 216)
        title_line = f"Project {idx}: {proj.get('title', '')} ({proj.get('difficulty', '')})"
        pdf.cell(0, 6, sanitize_for_pdf(title_line), new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, sanitize_for_pdf(f"Description: {proj.get('description', '')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(75, 85, 99)
        pdf.multi_cell(0, 5, sanitize_for_pdf(f"Tech Stack: {proj.get('technologies', '')}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    return bytes(pdf.output())


# ==============================================================================
# 6. RENDER RESULTS ON SCREEN (OUTPUT REQUIREMENTS)
# ==============================================================================
def show_results(data: Dict[str, Any], inputs: Dict[str, Any]):
    """
    Displays the generated skill analysis, 8-week roadmap, and 3 project cards.
    Provides Download, Regenerate, and Start Again buttons.
    """
    st.divider()

    # Success header
    st.success(
        f"🎯 Roadmap Ready for **{inputs['target_role']}** "
        f"({inputs['experience_level']} level | {inputs['hours_per_week']} hrs/week)"
    )

    # --------------------------------------------------------------------------
    # 1. Skills Comparison (I have vs To learn)
    # --------------------------------------------------------------------------
    st.subheader("📊 1. Skill Gap Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Skills You Already Have")
        have_list = data.get("skills_have", [])
        if have_list:
            have_table = [
                {"Skill": item.get("skill", ""), "Role Relevance": item.get("relevance", "")}
                for item in have_list
            ]
            st.dataframe(have_table, use_container_width=True, hide_index=True)
        else:
            st.info("No existing skills mapped yet.")

    with col2:
        st.markdown("#### 🚀 Skills You Need to Learn")
        learn_list = data.get("skills_to_learn", [])
        if learn_list:
            learn_table = [
                {
                    "Skill": item.get("skill", ""),
                    "Priority": item.get("priority", "Medium"),
                    "Why It Matters": item.get("why", ""),
                }
                for item in learn_list
            ]
            st.dataframe(learn_table, use_container_width=True, hide_index=True)
        else:
            st.success("Great job! You already cover the essential skills for this role.")

    st.write("")

    # --------------------------------------------------------------------------
    # 2. 8-Week Step-by-Step Roadmap
    # --------------------------------------------------------------------------
    st.subheader("🗓️ 2. 8-Week Step-by-Step Study Roadmap")
    st.caption("Expand each week to see specific topics, concepts, and your hands-on practice challenge.")

    roadmap_list = data.get("roadmap", [])
    for week_data in roadmap_list:
        w_num = week_data.get("week", 1)
        w_topic = week_data.get("topic", f"Week {w_num}")
        w_learn = week_data.get("what_to_learn", "")
        w_task = week_data.get("practice_task", "")

        with st.expander(f"**Week {w_num}: {w_topic}**", expanded=(w_num == 1)):
            st.markdown(f"**📖 What to Learn & Study:**\n\n{w_learn}")
            st.markdown(f"**🛠️ Hands-on Practice Task:**\n\n{w_task}")

    st.write("")

    # --------------------------------------------------------------------------
    # 3. Three Portfolio Mini-Projects
    # --------------------------------------------------------------------------
    st.subheader("💼 3. Recommended Portfolio Projects")
    st.caption("Build these 3 projects over your journey to demonstrate real skills to recruiters.")

    proj_cols = st.columns(3)
    projects_list = data.get("projects", [])

    for idx, col in enumerate(proj_cols):
        if idx < len(projects_list):
            p = projects_list[idx]
            with col:
                diff = p.get("difficulty", "Beginner")
                # Visual badge for difficulty
                badge = "🟢" if diff.lower() == "beginner" else ("🟡" if diff.lower() == "intermediate" else "🔴")
                st.markdown(f"### {badge} {p.get('title', 'Project')}")
                st.caption(f"**Difficulty:** {diff}")
                st.write(p.get("description", ""))
                st.markdown(f"**🧰 Tech:** `{p.get('technologies', '')}`")

    st.divider()

    # --------------------------------------------------------------------------
    # 4. Action Buttons (Download PDF, Regenerate, Start Again)
    # --------------------------------------------------------------------------
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 1])

    with btn_col1:
        # Generate PDF on the fly
        try:
            pdf_bytes = generate_pdf(data, inputs)
            clean_filename = f"PathPilot_Roadmap_{inputs['target_role'].replace(' ', '_')}.pdf"
            st.download_button(
                label="📥 Download Roadmap (PDF)",
                data=pdf_bytes,
                file_name=clean_filename,
                mime="application/pdf",
                help="Download a clean PDF report with your skills analysis, 8-week plan, and projects.",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.warning(f"Could not prepare PDF: {pdf_err}")

    with btn_col2:
        if st.button("🔄 Regenerate", help="Create a fresh roadmap using the same inputs", use_container_width=True):
            st.session_state["trigger_regenerate"] = True
            st.rerun()

    with btn_col3:
        if st.button("✨ Start Again", help="Clear all inputs and start fresh", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# ==============================================================================
# 7. MAIN APPLICATION CONTROLLER
# ==============================================================================
def main():
    """
    Main Streamlit application entrypoint.
    Handles UI layout, input form, validation, session state, and triggers.
    """
    # Page setup
    st.set_page_config(
        page_title="PathPilot AI - Career Roadmap & Skill Gap Finder",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # App Title and Description
    st.title("🧭 PathPilot AI")
    st.markdown(
        "**AI Career Roadmap & Skill Gap Finder for Computer Science & AI Students**\n\n"
        "Tell PathPilot your current skills and target role. Our AI analyzes your skills, "
        "identifies missing gaps, and creates a realistic **8-week study roadmap** with **3 portfolio projects**."
    )

    # Check for API Key
    api_key = get_api_key()
    if not api_key:
        st.warning(
            "⚠️ **Gemini API Key Not Found**\n\n"
            "To use PathPilot AI, please add your Google Gemini API key:\n"
            "1. Copy `.env.example` to `.env` in the `PathPilot-AI` folder.\n"
            "2. Set `GEMINI_API_KEY=your_actual_key_here` (get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)).\n"
            "3. Restart the Streamlit server."
        )

    # Initialize Session State
    if "roadmap_data" not in st.session_state:
        st.session_state["roadmap_data"] = None
    if "form_inputs" not in st.session_state:
        st.session_state["form_inputs"] = None
    if "trigger_regenerate" not in st.session_state:
        st.session_state["trigger_regenerate"] = False

    # Check if a regeneration was requested
    if st.session_state["trigger_regenerate"]:
        st.session_state["trigger_regenerate"] = False
        saved_inputs = st.session_state["form_inputs"]
        if saved_inputs and api_key:
            with st.spinner("Analyzing your skill profile and generating your personalized 8-week roadmap..."):
                prompt = build_prompt(
                    current_skills=saved_inputs["current_skills"],
                    target_role=saved_inputs["target_role"],
                    experience_level=saved_inputs["experience_level"],
                    hours_per_week=saved_inputs["hours_per_week"],
                )
                data, err = call_gemini(api_key, prompt)
                if err:
                    st.error(f"❌ {err}")
                else:
                    is_valid, val_err = validate_roadmap_data(data)
                    if not is_valid:
                        st.error(f"⚠️ Validation error in AI response: {val_err}. Please click Regenerate.")
                    else:
                        st.session_state["roadmap_data"] = data
            st.rerun()

    # --------------------------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------------------------
    with st.container():
        st.markdown("### 📋 Enter Your Profile")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            # 1. Current skills text input
            current_skills_input = st.text_input(
                "1. Your Current Skills (comma-separated):",
                placeholder="e.g. C, Python basics, HTML/CSS, Git, SQL basics",
                help="List programming languages, tools, frameworks, or CS concepts you already know.",
            )

            # 2. Target job role dropdown
            selected_role = st.selectbox(
                "2. Target Job Role:",
                options=ROLE_OPTIONS,
                index=0,
                help="Select the career path you want to prepare for.",
            )

            # If "Other" is chosen, show a custom text box
            custom_role = ""
            if selected_role == "Other":
                custom_role = st.text_input(
                    "Specify Your Target Role:",
                    placeholder="e.g. DevOps Engineer, Robotics Software Developer",
                )

        with col_right:
            # 3. Experience level dropdown (Requirement 1)
            experience_level = st.selectbox(
                "3. Your Current Experience Level:",
                options=EXPERIENCE_LEVELS,
                index=0,
                help="Beginner = First-year student / just starting. Intermediate = Done coursework & mini-projects. Advanced = Confident programmer.",
            )

            # 4. Hours per week available to study
            hours_per_week = st.slider(
                "4. Available Study Time (Hours / Week):",
                min_value=1,
                max_value=40,
                value=10,
                step=1,
                help="How many hours can you commit each week for learning and practice?",
            )

        # Generate Button
        generate_clicked = st.button(
            "🚀 Generate My Roadmap",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------------------------
    # FORM SUBMISSION & VALIDATION
    # --------------------------------------------------------------------------
    if generate_clicked:
        # Determine final role name
        effective_role = custom_role.strip() if selected_role == "Other" else selected_role.strip()

        # Validate inputs
        if not current_skills_input or not current_skills_input.strip():
            st.warning("⚠️ Please enter at least one current skill (e.g. 'C, basic Python').")
            return

        if not effective_role:
            st.warning("⚠️ Please specify your target job role.")
            return

        if not api_key:
            st.error("❌ Cannot generate roadmap: No Gemini API Key provided. Please add it to your .env file.")
            return

        # Prepare input dictionary
        inputs = {
            "current_skills": current_skills_input.strip(),
            "target_role": effective_role,
            "experience_level": experience_level,
            "hours_per_week": int(hours_per_week),
        }
        st.session_state["form_inputs"] = inputs

        # Call Gemini with spinner
        with st.spinner("Analyzing your skill profile and generating your personalized 8-week roadmap..."):
            prompt = build_prompt(
                current_skills=inputs["current_skills"],
                target_role=inputs["target_role"],
                experience_level=inputs["experience_level"],
                hours_per_week=inputs["hours_per_week"],
            )
            data, err = call_gemini(api_key, prompt)

            if err:
                st.error(f"❌ {err}")
                st.session_state["roadmap_data"] = None
            else:
                # Validate the response schema
                is_valid, val_err = validate_roadmap_data(data)
                if not is_valid:
                    st.error(
                        f"⚠️ The AI response did not match the expected format: {val_err}\n\n"
                        "Please click 'Generate My Roadmap' again to retry."
                    )
                    st.session_state["roadmap_data"] = None
                else:
                    st.session_state["roadmap_data"] = data

    # --------------------------------------------------------------------------
    # SHOW RESULTS IF AVAILABLE
    # --------------------------------------------------------------------------
    if st.session_state["roadmap_data"] is not None and st.session_state["form_inputs"] is not None:
        show_results(st.session_state["roadmap_data"], st.session_state["form_inputs"])


if __name__ == "__main__":
    main()
