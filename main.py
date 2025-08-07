# =================================================================
#       main.py (Definitive Version with Robust Image Handling)
# =================================================================

# --- 1. Import Necessary Libraries ---
import face_recognition
import numpy as np
import pickle
import uvicorn
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form
from typing import Dict, Any

# --- 2. Initialize the FastAPI Application ---
app = FastAPI(title="Face Recognition API - Robust Version")

# --- 3. Set up Data Persistence ---
DATA_FILE = "known_faces.dat"

def load_known_faces() -> Dict[str, Any]:
    try:
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {"ids": [], "encodings": []}

def save_known_faces(data: Dict[str, Any]):
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

known_faces_data = load_known_faces()

# --- 4. Define API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Face Recognition API is active and ready."}

@app.post("/register")
async def register_face(user_id: str = Form(...), file: UploadFile = File(...)):
    # --- Robust Image Handling ---
    # 1. Read the file into memory as bytes.
    image_bytes = await file.read()
    
    # 2. Open the image from bytes using Pillow and convert to RGB.
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(pil_image)
    except Exception as e:
        return {"status": "error", "message": f"Invalid image file: {e}"}

    # 3. Now, use the processed numpy array with face_recognition.
    face_encodings = face_recognition.face_encodings(image)
    if not face_encodings:
        return {"status": "error", "message": "No face was found in the uploaded image."}

    known_faces_data["ids"].append(user_id)
    known_faces_data["encodings"].append(face_encodings[0])
    save_known_faces(known_faces_data)

    print(f"Successfully registered face for user: {user_id}")
    return {"status": "success", "user_id": user_id}


@app.post("/identify")
async def identify_face(file: UploadFile = File(...)):
    if not known_faces_data["ids"]:
        return {"status": "error", "message": "There are no registered users in the system."}

    # --- Robust Image Handling ---
    image_bytes = await file.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        unknown_image = np.array(pil_image)
    except Exception as e:
        return {"status": "error", "message": f"Invalid image file: {e}"}

    # --- Location and Recognition Logic ---
    face_locations = face_recognition.face_locations(unknown_image)
    if not face_locations:
        return {"status": "no_face_detected"}

    unknown_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=face_locations)
    unknown_encoding = unknown_encodings[0]

    matches = face_recognition.compare_faces(known_faces_data["encodings"], unknown_encoding, tolerance=0.6)
    
    if True in matches:
        face_distances = face_recognition.face_distance(known_faces_data["encodings"], unknown_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            matched_user_id = known_faces_data["ids"][best_match_index]
            face_location = face_locations[0]
            print(f"Match found: {matched_user_id} at location {face_location}")
            return {"status": "match_found", "user_id": matched_user_id, "location": face_location}
    
    print(f"Unrecognized face detected at location {face_locations[0]}")
    return {"status": "no_match_found", "location": face_locations[0]}

# --- 5. Make the file runnable ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)