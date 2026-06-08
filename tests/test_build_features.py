"""Tests unitaires — feature engineering (sans DB)."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    add_rolling_features,
    build_weekly_features,
    merge_student_info,
)


def _make_events(n_weeks: int = 4, student_id: int = 1) -> pd.DataFrame:
    rows = []
    for w in range(n_weeks):
        week = pd.Timestamp("2024-09-02") + pd.Timedelta(weeks=w)
        rows += [
            {"student_id": student_id, "event_type": "login",
             "event_ts": week, "duration_sec": 3600, "score": None, "week_start": week},
            {"student_id": student_id, "event_type": "quiz_attempt",
             "event_ts": week, "duration_sec": 600, "score": 75.0, "week_start": week},
            {"student_id": student_id, "event_type": "quiz_pass",
             "event_ts": week, "duration_sec": None, "score": 75.0, "week_start": week},
            {"student_id": student_id, "event_type": "video_view",
             "event_ts": week, "duration_sec": 900, "score": None, "week_start": week},
            {"student_id": student_id, "event_type": "forum_post",
             "event_ts": week, "duration_sec": None, "score": None, "week_start": week},
            {"student_id": student_id, "event_type": "assignment_submit",
             "event_ts": week, "duration_sec": None, "score": 80.0, "week_start": week},
        ]
    return pd.DataFrame(rows)


def test_build_weekly_features_shape():
    events = _make_events(4)
    wf = build_weekly_features(events)
    assert len(wf) == 4
    assert "login_count" in wf.columns


def test_build_weekly_features_login_count():
    events = _make_events(2)
    wf = build_weekly_features(events)
    assert (wf["login_count"] == 1).all()


def test_build_weekly_features_quiz_pass_rate():
    events = _make_events(1)
    wf = build_weekly_features(events)
    assert wf["quiz_pass_rate"].iloc[0] == pytest.approx(1.0)


def test_build_weekly_features_videos():
    events = _make_events(3)
    wf = build_weekly_features(events)
    assert (wf["videos_watched"] == 1).all()


def test_add_rolling_features_columns():
    events = _make_events(6)
    wf = build_weekly_features(events)
    wf = add_rolling_features(wf)
    assert "login_count_trend4" in wf.columns
    assert "login_count_delta" in wf.columns


def test_merge_student_info_label():
    events = _make_events(4, student_id=1)
    wf = build_weekly_features(events)
    wf = add_rolling_features(wf)

    students = pd.DataFrame([{
        "student_id":    1,
        "cohort_id":     1,
        "status":        "dropped_out",
        "dropout_date":  pd.Timestamp("2024-11-01"),
        "entry_grade":   13.5,
        "has_job":       False,
        "scholarship":   True,
        "distance_km":   15.0,
        "age":           21,
        "gender":        "F",
    }])

    result = merge_student_info(wf, students)
    assert "label_dropout" in result.columns
    # Les semaines avant la date de décrochage ont label=1
    assert result["label_dropout"].dtype == int


def test_merge_no_dropout_label_zero():
    events = _make_events(4, student_id=2)
    wf = build_weekly_features(events)
    wf = add_rolling_features(wf)

    students = pd.DataFrame([{
        "student_id":  2,
        "cohort_id":   1,
        "status":      "enrolled",
        "dropout_date": None,
        "entry_grade": 14.0,
        "has_job":     False,
        "scholarship": False,
        "distance_km": 5.0,
        "age":         22,
        "gender":      "M",
    }])

    result = merge_student_info(wf, students)
    assert (result["label_dropout"] == 0).all()


def test_two_students_independent():
    e1 = _make_events(3, student_id=1)
    e2 = _make_events(5, student_id=2)
    events = pd.concat([e1, e2], ignore_index=True)
    wf = build_weekly_features(events)
    assert len(wf[wf["student_id"] == 1]) == 3
    assert len(wf[wf["student_id"] == 2]) == 5
