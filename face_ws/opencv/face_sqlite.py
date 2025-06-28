import sqlite3
import numpy as np
from scipy.spatial.distance import cosine
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

def add_client(feature, db_path='faces.db'):
    name = input("请输入用户姓名: ")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    feature = np.asarray(feature, dtype=np.float32)
    feature_bytes = feature.tobytes()

    c.execute('''
        INSERT INTO faces (name, feature)
        VALUES (?, ?)
    ''', (name, feature_bytes))

    conn.commit()
    user_id = c.lastrowid
    conn.close()
    return user_id

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
    
def set_favorability(face_id, value, db_path='faces.db'):
    """根据face_id设置好感度Favorability"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('UPDATE faces SET Favorability=? WHERE face_id=?', (value, face_id))
    conn.commit()
    conn.close()

    
def find_similar_face(feature, threshold=0.6):
    """
    比较输入的feature与数据库中所有的feature，返回相似度大于阈值的face_id列表。
    相似度采用余弦相似度计算。
    """
    
    db_path='faces.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT face_id, feature FROM faces')
    rows = cursor.fetchall()
    conn.close()

    feature = np.asarray(feature, dtype=np.float32)
    similar_faces = []
    
    for face_id, feature_bytes in rows:
        db_feature = np.frombuffer(feature_bytes, dtype=np.float32)
        
        # 计算两个人脸特征向量的相似度
        if np.linalg.norm(feature) == 0 or np.linalg.norm(db_feature) == 0:
            continue
            
        cosine_dist = cosine(feature, db_feature)
        cosine_similarity = 1 - cosine_dist  # 余弦相似度
        
        if cosine_similarity > threshold:
            similar_faces.append({
                'face_id': face_id,
                'cosine_distance': cosine_dist,
                'cosine_similarity': cosine_similarity
            })
    
    return similar_faces


def get_favorability(face_id, db_path='faces.db'):
    """根据face_id查询好感度Favorability"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT Favorability FROM faces WHERE face_id=?', (face_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    else:
        return None

