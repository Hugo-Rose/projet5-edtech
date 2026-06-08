"""
Génération de données synthétiques pour 10 000 étudiants sur 2 ans.

Usage:
    python -m src.data.generate_synthetic_data [--students 10000] [--output data/raw]
"""
import argparse
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("fr_FR")
rng = np.random.default_rng(42)


# ── Paramètres ────────────────────────────────────────────────────────────────

COHORTS = [
    {"cohort_id": 1, "name": "Promo 2023 - Data Science",
     "start": date(2023, 9, 4),  "end": date(2025, 6, 30),  "capacity": 35},
    {"cohort_id": 2, "name": "Promo 2024 - Développement Web",
     "start": date(2024, 9, 2),  "end": date(2026, 6, 30),  "capacity": 40},
    {"cohort_id": 3, "name": "Promo 2024 - Cybersécurité",
     "start": date(2024, 9, 2),  "end": date(2026, 6, 30),  "capacity": 30},
    {"cohort_id": 4, "name": "Promo 2025 - Data Science",
     "start": date(2025, 9, 1),  "end": date(2027, 6, 30),  "capacity": 35},
]

MODULES_PER_COHORT = 8
EVENT_TYPES = [
    "login", "logout", "video_view", "quiz_attempt", "quiz_pass", "quiz_fail",
    "forum_post", "forum_reply", "assignment_submit", "assignment_late",
    "resource_download", "live_session_join", "live_session_leave",
]

# Probabilités par profil (engaged / average / at_risk)
PROFILE_WEIGHTS = {"engaged": 0.35, "average": 0.45, "at_risk": 0.20}

PROFILE_PARAMS = {
    "engaged": {
        "logins_per_week": (8, 12),
        "videos_per_week": (5, 10),
        "quiz_pass_rate":   0.85,
        "forum_posts_week": (2, 6),
        "dropout_prob":     0.04,
        "late_assign_rate": 0.05,
    },
    "average": {
        "logins_per_week": (3, 7),
        "videos_per_week": (2, 5),
        "quiz_pass_rate":   0.65,
        "forum_posts_week": (0, 2),
        "dropout_prob":     0.15,
        "late_assign_rate": 0.20,
    },
    "at_risk": {
        "logins_per_week": (0, 3),
        "videos_per_week": (0, 2),
        "quiz_pass_rate":   0.40,
        "forum_posts_week": (0, 1),
        "dropout_prob":     0.45,
        "late_assign_rate": 0.50,
    },
}


# ── Génération des étudiants ──────────────────────────────────────────────────

def _pick_profile() -> str:
    return rng.choice(
        list(PROFILE_WEIGHTS.keys()),
        p=list(PROFILE_WEIGHTS.values()),
    )


def generate_students(n: int) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        cohort = random.choice(COHORTS)
        profile = _pick_profile()
        p = PROFILE_PARAMS[profile]

        gender = rng.choice(["M", "F", "Other"], p=[0.48, 0.48, 0.04])
        age = int(rng.normal(23, 4))
        age = max(17, min(45, age))

        entry_grade = float(rng.normal(12.5, 2.5))
        entry_grade = round(max(0, min(20, entry_grade)), 2)

        has_job = bool(rng.random() < 0.30)
        scholarship = bool(rng.random() < 0.25)
        distance_km = round(float(rng.exponential(20)), 1)

        # Statut final
        dropout = rng.random() < p["dropout_prob"]
        if dropout:
            max_days = (cohort["end"] - cohort["start"]).days
            dropout_offset = int(rng.integers(30, max(31, max_days // 2)))
            dropout_date = cohort["start"] + timedelta(days=dropout_offset)
            status = "dropped_out"
        else:
            dropout_date = None
            status = "enrolled" if cohort["end"] > date(2026, 6, 8) else "graduated"

        rows.append({
            "student_id":      i,
            "cohort_id":       cohort["cohort_id"],
            "first_name":      fake.first_name_male() if gender == "M" else fake.first_name_female(),
            "last_name":       fake.last_name(),
            "email":           f"student_{i}@edtech-sim.fr",
            "age":             age,
            "gender":          gender,
            "has_job":         has_job,
            "scholarship":     scholarship,
            "distance_km":     distance_km,
            "entry_grade":     entry_grade,
            "enrollment_date": cohort["start"],
            "status":          status,
            "dropout_date":    dropout_date,
            "profile":         profile,          # colonne interne (non exportée vers SQL)
            "cohort_start":    cohort["start"],
            "cohort_end":      cohort["end"],
        })
    return pd.DataFrame(rows)


# ── Génération des modules ────────────────────────────────────────────────────

SUBJECT_POOL = [
    "Mathématiques", "Algorithmique", "Machine Learning", "Deep Learning",
    "SQL & NoSQL", "Python", "Développement Web", "Cybersécurité",
    "Cloud Computing", "Visualisation", "Statistiques", "Projet fil rouge",
    "Communication professionnelle", "Anglais technique",
]


def generate_modules() -> pd.DataFrame:
    rows = []
    mid = 1
    for cohort in COHORTS:
        subjects = random.sample(SUBJECT_POOL, MODULES_PER_COHORT)
        for rank, subj in enumerate(subjects, 1):
            rows.append({
                "module_id":   mid,
                "cohort_id":   cohort["cohort_id"],
                "name":        f"{subj} — {cohort['name'].split(' - ')[0]}",
                "subject":     subj,
                "credits":     int(rng.choice([3, 4, 6])),
                "difficulty":  int(rng.integers(1, 6)),
            })
            mid += 1
    return pd.DataFrame(rows)


# ── Génération des événements LMS ─────────────────────────────────────────────

def _week_events(student_id: int, module_id: int, week_start: datetime,
                 profile: str) -> list[dict]:
    p = PROFILE_PARAMS[profile]
    events = []

    n_logins = int(rng.integers(*p["logins_per_week"]))
    for _ in range(n_logins):
        ts = week_start + timedelta(
            days=int(rng.integers(0, 7)),
            hours=int(rng.integers(7, 22)),
            minutes=int(rng.integers(0, 60)),
        )
        session_min = int(rng.integers(5, 120))
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    "login",
            "event_ts":      ts,
            "duration_sec":  session_min * 60,
            "score":         None,
            "attempt_number": None,
        })
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    "logout",
            "event_ts":      ts + timedelta(minutes=session_min),
            "duration_sec":  None,
            "score":         None,
            "attempt_number": None,
        })

    n_videos = int(rng.integers(*p["videos_per_week"]))
    for _ in range(n_videos):
        ts = week_start + timedelta(
            days=int(rng.integers(0, 7)),
            hours=int(rng.integers(8, 23)),
        )
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    "video_view",
            "event_ts":      ts,
            "duration_sec":  int(rng.integers(180, 1800)),
            "score":         None,
            "attempt_number": None,
        })

    # Quiz (1-2 par semaine)
    for attempt in range(1, int(rng.integers(1, 3)) + 1):
        ts = week_start + timedelta(days=int(rng.integers(0, 7)), hours=14)
        passed = rng.random() < p["quiz_pass_rate"]
        score = float(rng.normal(75, 15) if passed else rng.normal(40, 15))
        score = round(max(0, min(100, score)), 2)
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    "quiz_attempt",
            "event_ts":      ts,
            "duration_sec":  int(rng.integers(300, 3600)),
            "score":         score,
            "attempt_number": attempt,
        })
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    "quiz_pass" if passed else "quiz_fail",
            "event_ts":      ts + timedelta(seconds=int(rng.integers(1, 10))),
            "duration_sec":  None,
            "score":         score,
            "attempt_number": attempt,
        })

    # Forum
    n_posts = int(rng.integers(*p["forum_posts_week"]))
    for _ in range(n_posts):
        ts = week_start + timedelta(days=int(rng.integers(0, 7)), hours=int(rng.integers(9, 21)))
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    rng.choice(["forum_post", "forum_reply"]),
            "event_ts":      ts,
            "duration_sec":  None,
            "score":         None,
            "attempt_number": None,
        })

    # Assignment (1 par semaine, probabilité)
    if rng.random() < 0.6:
        is_late = rng.random() < p["late_assign_rate"]
        ts = week_start + timedelta(days=6 if is_late else 4, hours=23)
        events.append({
            "student_id":    student_id,
            "module_id":     module_id,
            "event_type":    "assignment_late" if is_late else "assignment_submit",
            "event_ts":      ts,
            "duration_sec":  None,
            "score":         round(float(rng.normal(70, 15)), 2),
            "attempt_number": 1,
        })

    return events


def generate_lms_events(students_df: pd.DataFrame,
                        modules_df: pd.DataFrame) -> pd.DataFrame:
    all_events: list[dict] = []

    module_by_cohort: dict[int, list[int]] = (
        modules_df.groupby("cohort_id")["module_id"]
        .apply(list)
        .to_dict()
    )

    sim_end = date(2026, 6, 8)

    for _, student in students_df.iterrows():
        cohort_modules = module_by_cohort.get(student["cohort_id"], [])
        if not cohort_modules:
            continue

        active_end = student["dropout_date"] if pd.notna(student["dropout_date"]) else sim_end
        active_end = min(active_end, student["cohort_end"], sim_end)

        current = datetime.combine(student["cohort_start"], datetime.min.time())
        end_dt = datetime.combine(active_end, datetime.min.time())

        profile = student["profile"]

        # Dégradation progressive pour les at_risk
        degradation_start = current + (end_dt - current) * 0.4 if profile == "at_risk" else None

        while current < end_dt:
            # Réduire l'engagement progressivement
            eff_profile = profile
            if degradation_start and current > degradation_start:
                eff_profile = "at_risk"

            module_id = int(rng.choice(cohort_modules))
            week_events = _week_events(
                int(student["student_id"]),
                module_id,
                current,
                eff_profile,
            )
            all_events.extend(week_events)
            current += timedelta(weeks=1)

    df = pd.DataFrame(all_events)
    df["event_id"] = range(1, len(df) + 1)
    df["created_at"] = datetime.now()
    return df.sort_values("event_ts").reset_index(drop=True)


# ── Export CSV ────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):>10,} rows  →  {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_students: int = 10_000, output_dir: str = "data/raw") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Génération de données synthétiques ({n_students:,} étudiants) ===\n")

    print("[1/4] Étudiants...")
    students = generate_students(n_students)
    save(students.drop(columns=["profile", "cohort_start", "cohort_end"]), "students", out)

    print("[2/4] Modules...")
    modules = generate_modules()
    save(modules, "modules", out)

    print("[3/4] Cohortes...")
    cohorts_df = pd.DataFrame(COHORTS)
    save(cohorts_df, "cohorts", out)

    print("[4/4] Événements LMS (peut prendre quelques minutes)...")
    events = generate_lms_events(students, modules)
    save(events.drop(columns=["created_at"]), "lms_events", out)

    print("\n=== Résumé ===")
    print(f"  Étudiants  : {len(students):>10,}")
    print(f"  Modules    : {len(modules):>10,}")
    print(f"  Événements : {len(events):>10,}")
    dropout_pct = (students["status"] == "dropped_out").mean() * 100
    print(f"  Taux décrochage : {dropout_pct:.1f}%")

    profile_counts = students["profile"].value_counts()
    for prof, cnt in profile_counts.items():
        print(f"    {prof:<10}: {cnt:,} ({cnt/n_students*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Génère des données LMS synthétiques.")
    parser.add_argument("--students", type=int, default=10_000)
    parser.add_argument("--output",   type=str, default="data/raw")
    args = parser.parse_args()
    main(n_students=args.students, output_dir=args.output)
