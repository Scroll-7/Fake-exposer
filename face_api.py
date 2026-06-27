from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import os
import shutil
import glob
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

app = FastAPI()

KNOWN_FACES_DIR = "known_faces"
UPLOADS_DIR = "uploads"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

print("Initializing InsightFace model (Buffalo_L)...")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))

# In-memory store of known face embeddings
# dict format: {"player_name": np.array(embedding)}
known_embeddings = {}

def load_known_faces():
    global known_embeddings
    known_embeddings.clear()
    
    images = glob.glob(f"{KNOWN_FACES_DIR}/*.*")
    for img_path in images:
        filename = os.path.basename(img_path)
        name = os.path.splitext(filename)[0].replace("_", " ").lower()
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        faces = face_app.get(img)
        if len(faces) > 0:
            known_embeddings[name] = faces[0].embedding
            print(f"Loaded known face: {name}")

# Load them on startup
load_known_faces()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "known_faces_count": len(known_embeddings),
        "model": "buffalo_l"
    }

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOADS_DIR, f"face_temp_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if not known_embeddings:
            load_known_faces()
            if not known_embeddings:
                return JSONResponse(content={"error": "No known faces", "player": None})

        img = cv2.imread(temp_path)
        faces = face_app.get(img)
        
        if len(faces) > 0:
            target_embedding = faces[0].embedding
            best_match_name = None
            highest_sim = -1
            
            for name, emb in known_embeddings.items():
                sim = cosine_similarity(target_embedding, emb)
                print(f"Similarity with {name}: {sim}")
                if sim > highest_sim:
                    highest_sim = sim
                    best_match_name = name
                    
            # A good threshold for Buffalo_L is typically ~0.3 to 0.4.
            # Let's use 0.35 as our threshold for "this is definitely the same person"
            THRESHOLD = 0.35
            
            if highest_sim > THRESHOLD:
                print(f"[FaceID] Match found: {best_match_name} (Confidence: {highest_sim:.2f})")
                return JSONResponse(content={"player": best_match_name, "confidence": float(highest_sim)})
            else:
                print(f"[FaceID] No match. Highest was {best_match_name} at {highest_sim:.2f}")
                return JSONResponse(content={"player": None, "reason": "Below threshold"})
        
        return JSONResponse(content={"player": None, "reason": "No face detected"})

    except Exception as e:
        return JSONResponse(content={"error": str(e), "player": None}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    print("Running Face API...")
    uvicorn.run(app, host="127.0.0.1", port=8002)
