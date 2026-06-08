"""Constantes partagées entre training, inférence et API."""

FEATURE_COLS = [
    "login_count", "total_time_min", "videos_watched",
    "quiz_attempts", "quiz_pass_rate", "avg_score",
    "forum_posts", "assignments_on_time", "assignments_late",
    "login_count_trend4", "total_time_min_trend4",
    "quiz_pass_rate_trend4", "avg_score_trend4",
    "login_count_delta", "total_time_min_delta",
    "quiz_pass_rate_delta", "avg_score_delta",
    "entry_grade", "has_job", "scholarship", "distance_km", "age", "gender_enc",
]

TARGET = "label_dropout"

MODELS_DIR_STR = "data/models"
