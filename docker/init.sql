-- ============================================================
-- EdTech Analytics — Schéma PostgreSQL
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ------------------------------------------------------------
-- Cohortes (promotions / filières)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cohorts (
    cohort_id       SERIAL PRIMARY KEY,
    name            VARCHAR(100)  NOT NULL,
    start_date      DATE          NOT NULL,
    end_date        DATE,
    program         VARCHAR(100),
    capacity        INTEGER,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Modules (cours / unités pédagogiques)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS modules (
    module_id       SERIAL PRIMARY KEY,
    cohort_id       INTEGER       REFERENCES cohorts(cohort_id) ON DELETE SET NULL,
    name            VARCHAR(200)  NOT NULL,
    subject         VARCHAR(100),
    credits         SMALLINT      DEFAULT 3,
    difficulty      SMALLINT      CHECK (difficulty BETWEEN 1 AND 5),
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Étudiants
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS students (
    student_id      SERIAL PRIMARY KEY,
    uuid            UUID          DEFAULT uuid_generate_v4() UNIQUE,
    cohort_id       INTEGER       REFERENCES cohorts(cohort_id) ON DELETE SET NULL,
    first_name      VARCHAR(100),
    last_name       VARCHAR(100),
    email           VARCHAR(200)  UNIQUE NOT NULL,
    age             SMALLINT,
    gender          VARCHAR(20),
    -- Socio-démographique
    has_job         BOOLEAN       DEFAULT FALSE,
    scholarship     BOOLEAN       DEFAULT FALSE,
    distance_km     NUMERIC(7,2),
    -- Scolarité
    entry_grade     NUMERIC(4,2),
    enrollment_date DATE,
    -- Statut final (label pour ML)
    status          VARCHAR(30)   DEFAULT 'enrolled'
                    CHECK (status IN ('enrolled','graduated','dropped_out','suspended')),
    dropout_date    DATE,
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_students_cohort  ON students(cohort_id);
CREATE INDEX IF NOT EXISTS idx_students_status  ON students(status);

-- ------------------------------------------------------------
-- Événements LMS
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lms_events (
    event_id        BIGSERIAL PRIMARY KEY,
    student_id      INTEGER       NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    module_id       INTEGER       REFERENCES modules(module_id) ON DELETE SET NULL,
    event_type      VARCHAR(50)   NOT NULL
                    CHECK (event_type IN (
                        'login','logout','video_view','quiz_attempt',
                        'quiz_pass','quiz_fail','forum_post','forum_reply',
                        'assignment_submit','assignment_late','resource_download',
                        'live_session_join','live_session_leave'
                    )),
    event_ts        TIMESTAMPTZ   NOT NULL,
    -- Métriques contextuelle
    duration_sec    INTEGER,
    score           NUMERIC(5,2),   -- pour quiz / assignments (0-100)
    attempt_number  SMALLINT,
    -- Payload JSON libre
    metadata        JSONB,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_lms_student   ON lms_events(student_id);
CREATE INDEX IF NOT EXISTS idx_lms_type      ON lms_events(event_type);
CREATE INDEX IF NOT EXISTS idx_lms_ts        ON lms_events(event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_lms_module    ON lms_events(module_id);

-- ------------------------------------------------------------
-- Features agrégées (snapshot hebdomadaire pour ML)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weekly_features (
    id              BIGSERIAL PRIMARY KEY,
    student_id      INTEGER       NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    week_start      DATE          NOT NULL,
    -- Engagement
    login_count     INTEGER       DEFAULT 0,
    total_time_min  NUMERIC(8,2)  DEFAULT 0,
    videos_watched  INTEGER       DEFAULT 0,
    -- Performance
    quiz_attempts   INTEGER       DEFAULT 0,
    quiz_pass_rate  NUMERIC(5,4),
    avg_score       NUMERIC(5,2),
    -- Collaboration
    forum_posts     INTEGER       DEFAULT 0,
    -- Assiduité
    assignments_on_time  INTEGER  DEFAULT 0,
    assignments_late     INTEGER  DEFAULT 0,
    -- Prédiction
    dropout_risk_score   NUMERIC(5,4),
    UNIQUE (student_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_wf_student    ON weekly_features(student_id);
CREATE INDEX IF NOT EXISTS idx_wf_week       ON weekly_features(week_start);

-- ------------------------------------------------------------
-- Prédictions enregistrées
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   BIGSERIAL PRIMARY KEY,
    student_id      INTEGER       NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    model_version   VARCHAR(50),
    predicted_at    TIMESTAMPTZ   DEFAULT NOW(),
    dropout_prob    NUMERIC(5,4)  NOT NULL,
    risk_label      VARCHAR(20)   CHECK (risk_label IN ('low','medium','high','critical')),
    features_snapshot JSONB
);

CREATE INDEX IF NOT EXISTS idx_pred_student  ON predictions(student_id);
CREATE INDEX IF NOT EXISTS idx_pred_ts       ON predictions(predicted_at DESC);

-- ------------------------------------------------------------
-- Alertes pédagogiques
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        BIGSERIAL PRIMARY KEY,
    student_id      INTEGER       NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    alert_type      VARCHAR(50),
    severity        VARCHAR(20)   CHECK (severity IN ('info','warning','critical')),
    message         TEXT,
    triggered_at    TIMESTAMPTZ   DEFAULT NOW(),
    resolved_at     TIMESTAMPTZ,
    resolved_by     VARCHAR(100)
);

-- ============================================================
-- Données initiales de référence
-- ============================================================
INSERT INTO cohorts (name, start_date, end_date, program, capacity) VALUES
  ('Promo 2023 - Data Science',      '2023-09-04', '2025-06-30', 'Data Science & IA',     35),
  ('Promo 2024 - Développement Web', '2024-09-02', '2026-06-30', 'Développement Web Full-Stack', 40),
  ('Promo 2024 - Cybersécurité',     '2024-09-02', '2026-06-30', 'Cybersécurité',         30),
  ('Promo 2025 - Data Science',      '2025-09-01', '2027-06-30', 'Data Science & IA',     35)
ON CONFLICT DO NOTHING;
