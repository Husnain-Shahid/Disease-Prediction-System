"""
Database migration script to upgrade the history table with patient details.
Run this script if you have an existing history.db file with the old schema.
"""
import sqlite3
import os

def migrate_database():
    """Migrate the database from old schema to new schema with patient details."""
    db_path = "history.db"
    
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found. A new database will be created with the new schema on next run.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if the new columns already exist
        cursor.execute("PRAGMA table_info(history)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "patient_name" in columns:
            print("Database is already migrated to the new schema.")
            conn.close()
            return
        
        print("Migrating database to new schema...")
        
        # Backup the old data
        cursor.execute("SELECT * FROM history")
        old_data = cursor.fetchall()
        
        # Rename old table
        cursor.execute("ALTER TABLE history RENAME TO history_old")
        
        # Create new table with patient details
        cursor.execute("""
        CREATE TABLE history (
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
        
        # Migrate old data
        # Old schema: id, disease, confidence, medicines, advice
        # New schema: id, patient_name, patient_age, patient_gender, disease, confidence, medicines, advice, created_at
        for row in old_data:
            cursor.execute("""
            INSERT INTO history (id, patient_name, patient_age, patient_gender, disease, confidence, medicines, advice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (row[0], "Unknown", 0, "Unknown", row[1], row[2], row[3], row[4]))
        
        conn.commit()
        print(f"Successfully migrated {len(old_data)} records to new schema!")
        print("Old data backed up in 'history_old' table. You can manually review or delete it after confirming the migration.")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()
