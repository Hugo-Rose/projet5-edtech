"""Tests unitaires — génération de données synthétiques."""
import pandas as pd
import pytest

from src.data.generate_synthetic_data import (
    generate_students,
    generate_modules,
    COHORTS,
    MODULES_PER_COHORT,
)


def test_students_count():
    df = generate_students(100)
    assert len(df) == 100


def test_students_columns():
    df = generate_students(50)
    required = {"student_id", "cohort_id", "email", "status", "entry_grade", "profile"}
    assert required.issubset(df.columns)


def test_students_status_values():
    df = generate_students(200)
    assert set(df["status"].unique()).issubset(
        {"enrolled", "graduated", "dropped_out", "suspended"}
    )


def test_students_no_duplicate_email():
    df = generate_students(500)
    assert df["email"].is_unique


def test_students_age_range():
    df = generate_students(500)
    assert df["age"].between(17, 45).all()


def test_students_entry_grade_range():
    df = generate_students(200)
    assert df["entry_grade"].between(0, 20).all()


def test_modules_count():
    df = generate_modules()
    assert len(df) == len(COHORTS) * MODULES_PER_COHORT


def test_modules_difficulty_range():
    df = generate_modules()
    assert df["difficulty"].between(1, 5).all()


def test_dropout_has_date():
    df = generate_students(500)
    dropouts = df[df["status"] == "dropped_out"]
    assert dropouts["dropout_date"].notna().all()


def test_enrolled_no_dropout_date():
    df = generate_students(500)
    enrolled = df[df["status"] == "enrolled"]
    assert enrolled["dropout_date"].isna().all()
