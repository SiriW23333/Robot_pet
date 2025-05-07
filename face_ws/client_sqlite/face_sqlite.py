import sqlite3
import numpy as np

DB_PATH = 'faces.db'

def init_db(db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            face_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            feature BLOB NOT NULL,
            Favorability INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
def add_client(name, feature, db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Convert numpy array to bytes for storage
    feature_bytes = feature.tobytes()

    c.execute('''
        INSERT INTO faces (name, feature)
        VALUES (?, ?)
    ''', (name, feature_bytes))

    conn.commit()
    conn.close()

def delete_client(face_id, db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM faces WHERE face_id=?", (face_id,))
    conn.commit()
    conn.close()

def get_face_by_name(name):
    """根据姓名查询人脸"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT face_id, name, feature, Favorability, created_at FROM faces WHERE name=?', (name,))
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        face_id, name, feature_bytes, Favorability, created_at = row
        feature = np.frombuffer(feature_bytes, dtype=np.float32)
        result.append({
            'face_id': face_id,
            'name': name,
            'feature': feature,
            'Favorability':Favorability,
            'created_at': created_at
        })

    return result

def increment_favorability_by_face_id(face_id, db_path='faces.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''
        UPDATE faces
        SET Favorability = Favorability + 1
        WHERE face_id = ?
    ''', (face_id,))

    conn.commit()
    conn.close()

def client_info():
    """查询所有人脸信息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT face_id, name, feature, Favorability, created_at FROM faces')
    rows = cursor.fetchall()
    conn.close()

    result = []
    for row in rows:
        face_id, name, feature_bytes, Favorability, created_at = row
        feature = np.frombuffer(feature_bytes, dtype=np.float32)
        result.append({
            'face_id': face_id,
            'name': name,
            'feature': feature,
            'Favorability':Favorability,
            'created_at': created_at
        })

    return result

