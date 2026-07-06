import sqlite3

def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_name TEXT NOT NULL,
        patient_age INTEGER NOT NULL,
        patient_gender TEXT NOT NULL,
        disease TEXT NOT NULL,
        confidence REAL,
        medicines TEXT,
        advice TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

def insert_record(patient_name, patient_age, patient_gender, disease, confidence, medicines, advice):
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO history (patient_name, patient_age, patient_gender, disease, confidence, medicines, advice)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_name, patient_age, patient_gender, disease, confidence, medicines, advice))

    conn.commit()
    conn.close()

def fetch_all():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data