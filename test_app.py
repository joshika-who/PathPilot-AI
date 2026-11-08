"""
Unit test suite for PathPilot AI functions.
Tests prompt building, JSON validation, and PDF generation with zero external API calls.
"""

import os
import sys

# Ensure the current directory is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from app import (
    build_prompt,
    validate_roadmap_data,
    generate_pdf,
    sanitize_for_pdf,
    DEFAULT_MODEL,
)


def sample_valid_data():
    return {
        "skills_have": [
            {"skill": "C Language", "relevance": "Strong foundation in memory management and logic"},
            {"skill": "Basic Python", "relevance": "Essential syntax for data manipulation and AI libraries"},
        ],
        "skills_to_learn": [
            {"skill": "NumPy & Pandas", "priority": "High", "why": "Core data science foundations"},
            {"skill": "PyTorch", "priority": "High", "why": "Deep learning and neural network development"},
            {"skill": "Docker", "priority": "Medium", "why": "Deploying machine learning containers"},
            {"skill": "FastAPI", "priority": "Medium", "why": "Serving model inference endpoints"},
            {"skill": "MLOps Basics", "priority": "Low", "why": "Production pipeline management"},
        ],
        "roadmap": [
            {
                "week": 1,
                "topic": "Python for Data Structures & NumPy",
                "what_to_learn": "List comprehensions, generators, NumPy ndarrays, vectorized ops",
                "practice_task": "Implement matrix multiplication and benchmark vectorized vs loops",
            },
            {
                "week": 2,
                "topic": "Data Wrangling with Pandas",
                "what_to_learn": "DataFrames, Series, filtering, missing values, group-by aggregations",
                "practice_task": "Clean and analyze the Titanic dataset with summary statistics",
            },
            {
                "week": 3,
                "topic": "Data Visualization & EDA",
                "what_to_learn": "Matplotlib and Seaborn for correlation heatmaps and distributions",
                "practice_task": "Create a 5-chart exploratory data analysis report on house prices",
            },
            {
                "week": 4,
                "topic": "Classical Machine Learning (Scikit-Learn)",
                "what_to_learn": "Linear regression, logistic regression, train-test splits, metrics",
                "practice_task": "Train a customer churn classifier and evaluate ROC-AUC",
            },
            {
                "week": 5,
                "topic": "Tree Models & Ensembles",
                "what_to_learn": "Decision trees, Random Forest, XGBoost, hyperparameter tuning",
                "practice_task": "Tune a Random Forest on a credit risk dataset with cross-validation",
            },
            {
                "week": 6,
                "topic": "Deep Learning Fundamentals (PyTorch)",
                "what_to_learn": "Tensors, autograd, linear layers, loss functions, optimizers",
                "practice_task": "Build a Multi-Layer Perceptron to classify MNIST handwritten digits",
            },
            {
                "week": 7,
                "topic": "Modern AI & Transformer Basics",
                "what_to_learn": "Self-attention mechanism, Hugging Face Transformers, fine-tuning",
                "practice_task": "Use a pre-trained BERT pipeline for sentiment analysis on reviews",
            },
            {
                "week": 8,
                "topic": "Model Deployment with FastAPI & Streamlit",
                "what_to_learn": "REST endpoints with Pydantic, connecting a Streamlit UI to FastAPI",
                "practice_task": "Deploy your Week 7 sentiment model as a working web app",
            },
        ],
        "projects": [
            {
                "title": "Smart Academic Performance Predictor",
                "description": "An end-to-end ML pipeline predicting student test scores with actionable advice.",
                "technologies": "Python, Pandas, Scikit-Learn, Streamlit",
                "difficulty": "Beginner",
            },
            {
                "title": "Medical Imaging Pneumonia Classifier",
                "description": "Convolutional neural network detecting pneumonia from chest X-ray images.",
                "technologies": "PyTorch, Torchvision, OpenCV, Matplotlib",
                "difficulty": "Intermediate",
            },
            {
                "title": "AI Resume & Job Matcher Assistant",
                "description": "NLP semantic matcher using transformer embeddings to score resumes against job specs.",
                "technologies": "Hugging Face, Sentence-Transformers, FastAPI, Docker",
                "difficulty": "Advanced",
            },
        ],
    }


def test_build_prompt():
    prompt = build_prompt(
        current_skills="C, basic Python",
        target_role="AI Engineer",
        experience_level="Beginner",
        hours_per_week=12,
    )
    assert "AI Engineer" in prompt
    assert "Beginner" in prompt
    assert "12 hours per week" in prompt
    assert "EXACTLY 8-week" in prompt
    assert "EXACTLY 3 practical portfolio" in prompt
    print("test_build_prompt: PASSED")


def test_validation_success():
    data = sample_valid_data()
    is_valid, err = validate_roadmap_data(data)
    assert is_valid is True, f"Validation failed unexpectedly: {err}"
    assert err == ""
    print("test_validation_success: PASSED")


def test_validation_catches_wrong_week_count():
    data = sample_valid_data()
    data["roadmap"].pop()  # Now 7 weeks
    is_valid, err = validate_roadmap_data(data)
    assert is_valid is False
    assert "exactly 8 weeks" in err
    print("test_validation_catches_wrong_week_count: PASSED")


def test_validation_catches_wrong_project_count():
    data = sample_valid_data()
    data["projects"].pop()  # Now 2 projects
    is_valid, err = validate_roadmap_data(data)
    assert is_valid is False
    assert "exactly 3 portfolio projects" in err
    print("test_validation_catches_wrong_project_count: PASSED")


def test_generate_pdf():
    data = sample_valid_data()
    inputs = {
        "current_skills": "C, basic Python, HTML",
        "target_role": "AI Engineer",
        "experience_level": "Beginner",
        "hours_per_week": 10,
    }
    pdf_bytes = generate_pdf(data, inputs)
    assert len(pdf_bytes) > 1000, "PDF bytes should be non-trivial"
    assert pdf_bytes.startswith(b"%PDF"), "Must be a valid PDF file"
    print(f"test_generate_pdf: PASSED (Generated valid PDF: {len(pdf_bytes)} bytes)")


def test_sanitize_text():
    raw = "Here’s a “smart” test — with em-dashes and • bullets."
    cleaned = sanitize_for_pdf(raw)
    assert "’" not in cleaned
    assert "“" not in cleaned
    assert "—" not in cleaned
    print("test_sanitize_text: PASSED")


if __name__ == "__main__":
    print(f"Running PathPilot AI Test Suite on model: {DEFAULT_MODEL}...")
    test_build_prompt()
    test_validation_success()
    test_validation_catches_wrong_week_count()
    test_validation_catches_wrong_project_count()
    test_sanitize_text()
    test_generate_pdf()
    print("\nALL PATHPILOT AI TESTS PASSED!")

