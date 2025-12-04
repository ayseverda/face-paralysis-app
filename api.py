from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import base64
from typing import Dict, List

from main import analyze_frame

app = FastAPI(title="Yuz Felci Analizi API")

# Geliştirme sırasında mobil cihazdan erişim için CORS'u açıyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # İstersen ileride belirli origin'lere daraltırsın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    pose: str | None = Form(None),
):
    """
    Tek bir yüz görüntüsünü analiz eder ve Mediapipe çizimlerini içeren resmi döner.
    React Native / Expo tarafı, bu endpoint'e multipart/form-data ile
    'file' alanında resmi POST edecek.
    """
    # Dosyayı belleğe oku
    data = await file.read()
    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            {"success": False, "reason": "Görüntü okunamadı"}, status_code=400
        )

    # main.py içindeki analyze_frame fonksiyonu, img üzerinde Mediapipe çizimlerini yapıyor
    score, details = analyze_frame(img)

    # Hata / yüz bulunamadı / kafa çok eğik vb.
    if score is None:
        # details string veya dict olabilir, ikisini de düzgün döndür
        if isinstance(details, dict):
            reason = details.get("reason", "Analiz başarısız")
        else:
            reason = str(details)
        return {"success": False, "reason": reason}

    # Başarılı analiz → sayısal detayları toparla
    numeric_details: Dict[str, float] = {}
    if isinstance(details, dict):
        for k, v in details.items():
            try:
                numeric_details[k] = float(v)
            except (TypeError, ValueError):
                # sadece sayısal olanlar üzerinden yorum üretmek istiyoruz.
                continue

    eye = numeric_details.get("eye_diff", 0.0)
    mouth = numeric_details.get("mouth_diff", 0.0)
    brow = numeric_details.get("brow_diff", 0.0)
    face = numeric_details.get("global_asym", 0.0)

    # Metin açıklama: hangi bölge daha çok etkilenmiş / simetrik
    reasons: List[str] = []

    def describe(val: float, low: float, mid: float, texts: List[str]) -> str:
        """
        texts: [çok düşük, hafif, belirgin]
        """
        if val < low:
            return texts[0]
        if val < mid:
            return texts[1]
        return texts[2]

    mouth_text = describe(
        mouth,
        0.06,
        0.14,
        [
            "Ağız köşeleri oldukça simetrik, belirgin bir kayma yok.",
            "Ağız köşeleri arasında hafif bir seviye farkı var.",
            "Ağız köşeleri arasında belirgin seviye farkı var (dikkat çekici asimetri).",
        ],
    )

    brow_text = describe(
        brow,
        0.06,
        0.14,
        [
            "Kaş seviyeleri büyük oranda dengeli.",
            "Kaşların yüksekliği arasında hafif bir fark tespit edildi.",
            "Kaşların yüksekliği arasında belirgin fark tespit edildi.",
        ],
    )

    eye_text = describe(
        eye,
        0.06,
        0.14,
        [
            "Göz açıklıkları neredeyse eşit, kırpma asimetrisi düşük.",
            "Göz açıklıkları arasında hafif bir fark mevcut.",
            "Göz açıklıkları arasında belirgin fark var (bir göz diğerinden daha kısık).",
        ],
    )

    face_text = describe(
        face,
        0.06,
        0.14,
        [
            "Genel yüz asimetrisi düşük, yüz hatları dengeli.",
            "Genel yüz asimetrisi orta düzeyde.",
            "Genel yüz asimetrisi belirgin, sol ve sağ taraf arasında fark göze çarpıyor.",
        ],
    )

    # Poz tipine göre hangi açıklamaların öne çıkacağı
    pose_key = (pose or "").lower()
    if pose_key in {"smile", "pucker"}:
        reasons.extend([mouth_text, face_text])
    elif pose_key in {"brow_up", "brow_frown"}:
        reasons.extend([brow_text, face_text])
    elif pose_key in {"neutral1", "neutral2"}:
        reasons.append(face_text)
    else:
        # Belirsiz/tek kare analiz: hepsini göster
        reasons.extend([mouth_text, brow_text, eye_text, face_text])

    metrics = {
        "göz": eye,
        "ağız": mouth,
        "kaş": brow,
        "yüz": face,
    }
    main_area = max(metrics, key=metrics.get)
    main_area_msg = f"Bu karede en belirgin asimetri: {main_area} bölgesinde."

    # Çizimleri içeren görüntüyü JPEG'e çevirip base64 string olarak döndür
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        annotated_b64 = None
    else:
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")

    return {
        "success": True,
        "score": float(score),
        "details": numeric_details,
        "reason": " ".join(reasons + [main_area_msg]),
        "annotated_image": annotated_b64,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)