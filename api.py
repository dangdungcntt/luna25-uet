import os
import tempfile
import time
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from inference import NoduleProcessor
from typing import Optional

app = FastAPI(title="LUNA25 Lesion Inference API")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
THRESHOLD = float(os.environ.get("THRESHOLD", 0.7))

def run_inference(
    file_path: str,
    series_uid: str,
    patient_id: str,
    study_date: str,
    lesion_id: int,
    coord_x: float,
    coord_y: float,
    coord_z: float,
    age_at_study_date: int,
    gender: str,
):
    """
    Stub function — replace with actual inference from LUNA challenge pipeline.
    Should return: probability(float), label(int)
    """

    input_nodule_locations = {
        "points": [
            {
                "name": "point_1",
                "point": [coord_x, coord_y, coord_z]
            }
        ]
    }
    input_clinical_information = {
        "Age_at_StudyDate": age_at_study_date,
        "Gender": gender
    }

    processor = NoduleProcessor(ct_image_file=file_path,
                                nodule_locations=input_nodule_locations,
                                clinical_information=input_clinical_information,
                                mode="3D",
                                model_name="I3D_UNet-3D-20251208_090637-cls-BCEDice",
                                device=DEVICE)
    image, coords, _ = processor.load_inputs()
    output = processor.predict(image, coords)
    
    probability = float(output[0])
    label = 1 if probability > THRESHOLD else 0
    return probability, label


@app.post("/api/v1/predict/lesion")
async def predict_lesion(
    file: UploadFile = File(...),
    seriesInstanceUID: str = Form(...),
    patientID: str = Form(...),
    studyDate: Optional[str] = Form(None),
    lesionID: int = Form(...),
    coordX: float = Form(...),
    coordY: float = Form(...),
    coordZ: float = Form(...),
    ageAtStudyDate: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
):
    start_time = time.time()

    # Validate file type
    if not file.filename.lower().endswith(".mha"):
        raise HTTPException(status_code=400, detail="File must be a .mha CT scan")

    # Validate studyDate format YYYYMMDD
    if studyDate and (len(studyDate) != 8 or not studyDate.isdigit()):
        raise HTTPException(status_code=400, detail="studyDate must be YYYYMMDD")
    
    if ageAtStudyDate and not ageAtStudyDate.isdigit():
        raise HTTPException(status_code=400, detail="ageAtStudyDate must be a number")

    # Save temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Run inference
    probability, label = run_inference(
        file_path=tmp_path,
        series_uid=seriesInstanceUID,
        patient_id=patientID,
        study_date=studyDate,
        lesion_id=lesionID,
        coord_x=coordX,
        coord_y=coordY,
        coord_z=coordZ,
        age_at_study_date=ageAtStudyDate,
        gender=gender,
    )

    processing_ms = int((time.time() - start_time) * 1000)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "data": {
                "seriesInstanceUID": seriesInstanceUID,
                "lesionID": lesionID,
                "probability": probability,
                "predictionLabel": label,
                "processingTimeMs": processing_ms,
            },
        },
    )


@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=os.environ.get("PORT", 8000), reload=False)
