import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from deepface import DeepFace
import pickle
from threading import Lock

app = Flask(__name__)

DATASET_DIR = "dataset"
EMBEDDINGS_PATH = "embeddings.pkl"
lock = Lock()

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Load embeddings DB or create empty
if os.path.exists(EMBEDDINGS_PATH):
    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = {"embeddings": [], "names": []}

def save_embeddings_db():
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings_db, f)

def preprocess_base64_img(base64_img):
    # base64_img: data:image/jpeg;base64,/9j/...
    header, encoded = base64_img.split(",", 1)
    data = base64.b64decode(encoded)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def normalize(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    img_b64 = data.get("image")

    if not name or not img_b64:
        return jsonify({"success": False, "message": "Name or image missing."}), 400

    img = preprocess_base64_img(img_b64)
    filename = f"{name}_{len(os.listdir(DATASET_DIR))}.jpg"
    filepath = os.path.join(DATASET_DIR, filename)
    cv2.imwrite(filepath, img)

    try:
        embedding_obj = DeepFace.represent(img_path=filepath, model_name="Facenet", enforce_detection=True)
        embedding = embedding_obj[0]["embedding"] if isinstance(embedding_obj, list) else embedding_obj
        embedding = normalize(embedding)
    except Exception as e:
        return jsonify({"success": False, "message": f"Face not detected or error: {str(e)}"}), 400

    with lock:
        embeddings_db["embeddings"].append(embedding)
        embeddings_db["names"].append(name)
        save_embeddings_db()

    return jsonify({"success": True, "message": f"Face registered for {name}."})

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    img_b64 = data.get("image")
    if not img_b64:
        return jsonify({"success": False, "message": "Image missing."}), 400

    img = preprocess_base64_img(img_b64)

    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        recognized_faces = []

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]

            try:
                rep_obj = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=True)
                rep = rep_obj[0]["embedding"] if isinstance(rep_obj, list) else rep_obj
                rep = normalize(rep)
            except Exception:
                recognized_faces.append({"box": [int(x), int(y), int(w), int(h)], "name": "Unknown"})
                continue

            max_sim = -1
            identity = "Unknown"

            for db_emb, db_name in zip(embeddings_db["embeddings"], embeddings_db["names"]):
                sim = cosine_similarity(db_emb, rep)
                if sim > 0.5 and sim > max_sim:
                    max_sim = sim
                    identity = db_name

            recognized_faces.append({"box": [int(x), int(y), int(w), int(h)], "name": identity})

        return jsonify({"success": True, "faces": recognized_faces})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
