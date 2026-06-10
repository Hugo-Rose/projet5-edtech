"""
Tests unitaires — loaders UCI et OULAD (sans réseau, sans téléchargement).

Toutes les fonctions réseau sont mockées ; on teste uniquement
les transformations / mappings.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.load_uci_dropout import transform as uci_transform
from src.data.load_oulad import (
    build_students as oulad_students,
    build_weekly_from_vle,
)
from src.features.build_features import (
    _expand_weekly_to_events,
    SOURCE_CHOICES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _uci_raw(n: int = 50) -> pd.DataFrame:
    """DataFrame imitant le format brut UCI 697."""
    rng = np.random.default_rng(0)
    target_choices = ["Dropout", "Graduate", "Enrolled"]
    return pd.DataFrame({
        "Gender":                               rng.integers(0, 2, n),
        "Age at enrollment":                    rng.integers(17, 45, n),
        "Admission grade":                      rng.uniform(95, 190, n),
        "Scholarship holder":                   rng.integers(0, 2, n),
        "Daytime/evening attendance":           rng.integers(0, 2, n),
        "Course":                               rng.choice([9991, 9119, 9254, 9500], n),
        "Curricular units 1st sem (grade)":     rng.uniform(0, 18, n),
        "Curricular units 1st sem (approved)":  rng.integers(0, 7, n),
        "Curricular units 1st sem (enrolled)":  rng.integers(1, 8, n),
        "Curricular units 1st sem (evaluations)": rng.integers(0, 10, n),
        "Curricular units 2nd sem (grade)":     rng.uniform(0, 18, n),
        "Curricular units 2nd sem (approved)":  rng.integers(0, 7, n),
        "Curricular units 2nd sem (enrolled)":  rng.integers(1, 8, n),
        "Curricular units 2nd sem (evaluations)": rng.integers(0, 10, n),
        "Debtor":                               rng.integers(0, 2, n),
        "Tuition fees up to date":              rng.integers(0, 2, n),
        "International":                        rng.integers(0, 2, n),
        "Target": rng.choice(target_choices, n),
    })


def _oulad_info(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "id_student":           range(1, n + 1),
        "gender":               rng.choice(["M", "F"], n),
        "age_band":             rng.choice(["0-35", "35-55", "55<="], n),
        "highest_education":    rng.choice(["A Level or Equivalent", "HE Qualification"], n),
        "imd_band":             rng.choice(["0-10%", "50-60%", "90-100%"], n),
        "num_of_prev_attempts": rng.integers(0, 3, n),
        "studied_credits":      rng.integers(60, 240, n),
        "disability":           rng.choice(["Y", "N"], n),
        "final_result":         rng.choice(["Pass", "Fail", "Withdrawn", "Distinction"], n),
        "code_module":          rng.choice(["AAA", "BBB", "CCC"], n),
        "code_presentation":    rng.choice(["2013J", "2014B", "2014J"], n),
    })


def _oulad_vle(n_students: int = 30, n_records: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "id_student":         rng.integers(1, n_students + 1, n_records),
        "code_module":        rng.choice(["AAA", "BBB"], n_records),
        "code_presentation":  "2014J",
        "id_site":            rng.integers(100, 200, n_records),
        "date":               rng.integers(0, 250, n_records),
        "sum_click":          rng.integers(1, 50, n_records),
    })


def _weekly_prebuilt(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = [pd.Timestamp("2022-09-05") + pd.Timedelta(weeks=w) for w in range(4)]
    rows = []
    for sid in range(1, n + 1):
        for d in dates:
            rows.append({
                "student_id":          sid,
                "week_start":          d,
                "login_count":         int(rng.integers(0, 10)),
                "total_time_min":      float(rng.uniform(0, 300)),
                "videos_watched":      int(rng.integers(0, 5)),
                "quiz_attempts":       int(rng.integers(0, 4)),
                "quiz_pass_rate":      float(rng.uniform(0, 1)),
                "avg_score":           float(rng.uniform(0, 100)),
                "forum_posts":         0,
                "assignments_on_time": int(rng.integers(0, 2)),
                "assignments_late":    int(rng.integers(0, 2)),
                "label_dropout":       int(rng.random() < 0.2),
            })
    return pd.DataFrame(rows)


# ── Tests UCI transform ───────────────────────────────────────────────────────

class TestUciTransform:

    def test_returns_two_dataframes(self):
        raw = _uci_raw(30)
        students, weekly = uci_transform(raw)
        assert isinstance(students, pd.DataFrame)
        assert isinstance(weekly, pd.DataFrame)

    def test_students_count(self):
        raw = _uci_raw(50)
        students, _ = uci_transform(raw)
        assert len(students) == 50

    def test_weekly_two_snapshots_per_student(self):
        raw = _uci_raw(20)
        _, weekly = uci_transform(raw)
        snaps_per_student = weekly.groupby("student_id").size()
        assert (snaps_per_student == 2).all()

    def test_gender_encoding(self):
        raw = _uci_raw(100)
        students, _ = uci_transform(raw)
        assert set(students["gender"].unique()).issubset({"M", "F", "Other"})

    def test_entry_grade_range(self):
        raw = _uci_raw(100)
        students, _ = uci_transform(raw)
        assert students["entry_grade"].between(0, 20).all()

    def test_label_dropout_binary(self):
        raw = _uci_raw(100)
        students, _ = uci_transform(raw)
        assert set(students["label_dropout"].unique()).issubset({0, 1})

    def test_label_dropout_matches_target(self):
        raw = _uci_raw(100)
        students, _ = uci_transform(raw)
        # Toutes les lignes Target=Dropout → label_dropout=1
        dropout_ids = students[students["status"] == "dropped_out"]["student_id"]
        assert (students.set_index("student_id")
                        .loc[dropout_ids, "label_dropout"] == 1).all()

    def test_status_values(self):
        raw = _uci_raw(100)
        students, _ = uci_transform(raw)
        assert set(students["status"].unique()).issubset(
            {"dropped_out", "graduated", "enrolled"}
        )

    def test_cohort_id_range(self):
        raw = _uci_raw(100)
        students, _ = uci_transform(raw)
        assert students["cohort_id"].between(1, 4).all()

    def test_weekly_score_range(self):
        raw = _uci_raw(50)
        _, weekly = uci_transform(raw)
        assert weekly["avg_score"].between(0, 100).all()

    def test_weekly_pass_rate_range(self):
        raw = _uci_raw(50)
        _, weekly = uci_transform(raw)
        assert weekly["quiz_pass_rate"].between(0, 1).all()

    def test_missing_columns_handled(self):
        raw = _uci_raw(20).drop(columns=[
            "Curricular units 2nd sem (grade)",
            "Debtor",
        ], errors="ignore")
        students, weekly = uci_transform(raw)
        assert len(students) == 20

    def test_no_duplicate_student_ids(self):
        raw = _uci_raw(50)
        students, _ = uci_transform(raw)
        assert students["student_id"].is_unique


# ── Tests OULAD build_students ────────────────────────────────────────────────

class TestOuladStudents:

    def test_returns_dataframe(self):
        info = _oulad_info(40)
        result = oulad_students(info)
        assert isinstance(result, pd.DataFrame)

    def test_student_count(self):
        info = _oulad_info(60)
        result = oulad_students(info)
        assert len(result) == 60

    def test_gender_mapping(self):
        info = _oulad_info(50)
        result = oulad_students(info)
        assert set(result["gender"].unique()).issubset({"M", "F", "Other"})

    def test_age_from_age_band(self):
        info = _oulad_info(50)
        result = oulad_students(info)
        assert result["age"].between(18, 70).all()

    def test_label_dropout_binary(self):
        info = _oulad_info(80)
        result = oulad_students(info)
        assert set(result["label_dropout"].unique()).issubset({0, 1})

    def test_withdrawn_is_dropout(self):
        info = _oulad_info(100)
        result = oulad_students(info)
        raw_withdrawn = info[info["final_result"] == "Withdrawn"]["id_student"].tolist()
        if raw_withdrawn:
            r = result.set_index("student_id")
            for sid in raw_withdrawn[:5]:
                assert r.loc[sid, "label_dropout"] == 1

    def test_pass_is_not_dropout(self):
        info = _oulad_info(100)
        result = oulad_students(info)
        raw_pass = info[info["final_result"] == "Pass"]["id_student"].tolist()
        if raw_pass:
            r = result.set_index("student_id")
            for sid in raw_pass[:5]:
                assert r.loc[sid, "label_dropout"] == 0

    def test_cohort_id_assigned(self):
        info = _oulad_info(50)
        result = oulad_students(info)
        assert result["cohort_id"].notna().all()
        assert result["cohort_id"].between(1, 4).all()

    def test_required_columns(self):
        info = _oulad_info(30)
        result = oulad_students(info)
        required = {"student_id", "cohort_id", "gender", "age",
                    "status", "label_dropout", "email"}
        assert required.issubset(result.columns)


# ── Tests OULAD build_weekly_from_vle ─────────────────────────────────────────

class TestOuladWeekly:

    def test_returns_dataframe(self):
        vle  = _oulad_vle()
        info = _oulad_info()
        result = build_weekly_from_vle(vle, info)
        assert isinstance(result, pd.DataFrame)

    def test_empty_vle_returns_empty(self):
        result = build_weekly_from_vle(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_student_ids_preserved(self):
        vle = _oulad_vle(n_students=10, n_records=200)
        result = build_weekly_from_vle(vle, _oulad_info(10))
        assert set(result["student_id"].unique()).issubset(
            set(range(1, 11))
        )

    def test_weekly_aggregation_per_student(self):
        vle = _oulad_vle(n_students=5, n_records=100)
        result = build_weekly_from_vle(vle, pd.DataFrame())
        assert result.groupby(["student_id", "week_start"]).size().max() == 1

    def test_login_count_non_negative(self):
        vle = _oulad_vle()
        result = build_weekly_from_vle(vle, pd.DataFrame())
        assert (result["login_count"] >= 0).all()

    def test_time_min_non_negative(self):
        vle = _oulad_vle()
        result = build_weekly_from_vle(vle, pd.DataFrame())
        assert (result["total_time_min"] >= 0).all()

    def test_label_dropout_added_from_info(self):
        vle  = _oulad_vle(n_students=20, n_records=400)
        info = _oulad_info(20)
        result = build_weekly_from_vle(vle, info)
        assert "label_dropout" in result.columns
        assert result["label_dropout"].isin([0, 1]).all()

    def test_required_columns_present(self):
        vle  = _oulad_vle()
        result = build_weekly_from_vle(vle, pd.DataFrame())
        required = {"student_id", "week_start", "login_count",
                    "total_time_min", "videos_watched", "label_dropout"}
        assert required.issubset(result.columns)


# ── Tests _expand_weekly_to_events ────────────────────────────────────────────

class TestExpandWeekly:

    def test_returns_dataframe(self):
        weekly = _weekly_prebuilt(10)
        events = _expand_weekly_to_events(weekly)
        assert isinstance(events, pd.DataFrame)

    def test_events_have_student_id(self):
        weekly = _weekly_prebuilt(5)
        events = _expand_weekly_to_events(weekly)
        assert "student_id" in events.columns

    def test_event_types_valid(self):
        weekly = _weekly_prebuilt(5)
        events = _expand_weekly_to_events(weekly)
        valid = {"login", "video_view", "quiz_attempt", "quiz_pass",
                 "quiz_fail", "assignment_submit", "assignment_late"}
        assert set(events["event_type"].unique()).issubset(valid)

    def test_empty_weekly_returns_empty(self):
        empty = pd.DataFrame(columns=["student_id", "week_start", "login_count",
                                       "videos_watched", "quiz_attempts",
                                       "quiz_pass_rate", "avg_score",
                                       "assignments_on_time", "assignments_late"])
        result = _expand_weekly_to_events(empty)
        assert result.empty or len(result) == 0

    def test_login_count_reflected(self):
        weekly = pd.DataFrame([{
            "student_id": 1,
            "week_start": pd.Timestamp("2024-01-01"),
            "login_count": 5,
            "videos_watched": 0, "quiz_attempts": 0,
            "quiz_pass_rate": 0, "avg_score": 70,
            "assignments_on_time": 0, "assignments_late": 0,
        }])
        events = _expand_weekly_to_events(weekly)
        logins = events[events["event_type"] == "login"]
        assert len(logins) == 5

    def test_round_trip_preserves_student_ids(self):
        weekly = _weekly_prebuilt(20)
        events = _expand_weekly_to_events(weekly)
        assert set(events["student_id"].unique()) == set(weekly["student_id"].unique())


# ── Tests SOURCE_CHOICES ─────────────────────────────────────────────────────

def test_source_choices_valid():
    assert set(SOURCE_CHOICES) == {"db", "uci", "oulad", "synthetic"}


# ── Tests fetch avec mock réseau ──────────────────────────────────────────────

class TestUciFetchMocked:

    def test_fetch_called_with_correct_id(self):
        mock_dataset = MagicMock()
        mock_dataset.data.features = _uci_raw(10).drop(columns=["Target"])
        mock_dataset.data.targets = pd.DataFrame(
            {"Target": ["Dropout"] * 5 + ["Graduate"] * 5}
        )
        with patch("ucimlrepo.fetch_ucirepo",
                   return_value=mock_dataset) as mock_fetch:
            from src.data.load_uci_dropout import fetch_raw
            result = fetch_raw()
            mock_fetch.assert_called_once_with(id=697)
            assert len(result) == 10

    def test_missing_ucimlrepo_raises_import_error(self):
        import sys
        with patch.dict(sys.modules, {"ucimlrepo": None}):
            from importlib import reload
            import src.data.load_uci_dropout as mod
            with pytest.raises((ImportError, ModuleNotFoundError)):
                mod.fetch_raw()


class TestOuladFetchMocked:

    def test_load_hf_returns_dataframe(self):
        mock_ds = MagicMock()
        mock_split = MagicMock()
        mock_split.to_pandas.return_value = _oulad_info(20)
        mock_ds.__getitem__ = lambda self, k: mock_split
        mock_ds.keys.return_value = ["train"]

        with patch("datasets.load_dataset", return_value=mock_ds):
            from src.data.load_oulad import _load_hf_table
            result = _load_hf_table("studentInfo")
            assert isinstance(result, pd.DataFrame)

    def test_load_hf_network_error_returns_empty(self):
        with patch("datasets.load_dataset",
                   side_effect=Exception("Network error")):
            from src.data.load_oulad import _load_hf_table
            result = _load_hf_table("studentVle")
            assert result.empty
