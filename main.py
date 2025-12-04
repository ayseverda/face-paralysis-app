# ... existing code ...
import glob
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# "Sonrakine geç" butonu için alan ve mouse state
NEXT_BTN_RECT = (20, 260, 260, 300)  # (x1, y1, x2, y2)
next_clicked = False


def mouse_callback(event, x, y, flags, param):
    global next_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = NEXT_BTN_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            next_clicked = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
)

LANDMARKS = {
    "left_eye": [33, 133, 160, 159, 158, 153, 144, 145],
    "right_eye": [263, 362, 385, 380, 381, 382, 373, 390],
    # Ağız köşeleri (gülümseme asimetrisi için)
    "mouth_corners": [61, 291],
    # Üst / alt dudak (ağız açıklığı için)
    "mouth_vertical": [13, 14],
    # Kaş için birden fazla nokta (sol / sağ)
    "brow_left": [70, 63],
    "brow_right": [300, 293],
}

# Sol–sağ simetrik noktalar: göz, kaş, ağız, yanak/burun çevresi
LEFT_RIGHT_PAIRS = [
    (33, 263),   # dış göz köşeleri
    (133, 362),  # iç göz köşeleri
    (159, 386),  # üst göz kapağı
    (145, 374),  # alt göz kapağı
    (61, 291),   # ağız köşeleri
    (40, 270),   # nazolabial çizgi çevresi
    (70, 300),   # kaş orta
    (63, 293),   # kaş iç
    (105, 334),  # kaş dış
]


def calc_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def extract_metrics(get_point):
    # GÖZLER: her göz için "eye aspect ratio" (yükseklik / genişlik), sonra iki gözün oran farkı
    left_eye = np.array([get_point(i) for i in LANDMARKS["left_eye"]])
    right_eye = np.array([get_point(i) for i in LANDMARKS["right_eye"]])

    def eye_ear(eye_pts):
        # 0-3 genişlik, (1-5 ve 2-4) çiftleri yükseklik
        width = calc_distance(eye_pts[0], eye_pts[3])
        h1 = calc_distance(eye_pts[1], eye_pts[5])
        h2 = calc_distance(eye_pts[2], eye_pts[4])
        height = (h1 + h2) / 2.0
        return height / (width + 1e-6)

    ear_left = eye_ear(left_eye)
    ear_right = eye_ear(right_eye)
    eye_diff = abs(ear_left - ear_right) / (max(ear_left, ear_right) + 1e-6)

    # AĞIZ: hem gülümseme asimetrisi hem de ağız açıklığı
    mouth_left = get_point(LANDMARKS["mouth_corners"][0])
    mouth_right = get_point(LANDMARKS["mouth_corners"][1])
    upper_lip = get_point(LANDMARKS["mouth_vertical"][0])
    lower_lip = get_point(LANDMARKS["mouth_vertical"][1])

    mouth_width = calc_distance(mouth_left, mouth_right) + 1e-6
    # köşeler arası yükseklik farkı / genişlik -> gülümseme asimetrisi
    smile_asym = abs(mouth_left[1] - mouth_right[1]) / mouth_width
    # üst-alt dudak arası mesafe / genişlik -> ağız açıklığı (gülümseme, dudak büzme vs)
    mouth_open = calc_distance(upper_lip, lower_lip) / mouth_width
    # ikisini birleştiren tek bir ağız metriği
    mouth_diff = 0.5 * smile_asym + 0.5 * mouth_open

    # KAŞLAR: kaşların göz merkezine göre yüksekliği (sol / sağ farkı)
    brow_left_pts = np.array([get_point(i) for i in LANDMARKS["brow_left"]])
    brow_right_pts = np.array([get_point(i) for i in LANDMARKS["brow_right"]])

    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)

    left_brow_y = np.mean(brow_left_pts[:, 1])
    right_brow_y = np.mean(brow_right_pts[:, 1])

    left_eye_width = calc_distance(left_eye[0], left_eye[3]) + 1e-6
    right_eye_width = calc_distance(right_eye[0], right_eye[3]) + 1e-6

    # göz merkezine göre normalize edilmiş kaş yüksekliği
    left_brow_h = (left_eye_center[1] - left_brow_y) / left_eye_width
    right_brow_h = (right_eye_center[1] - right_brow_y) / right_eye_width

    brow_diff = abs(left_brow_h - right_brow_h) / (max(abs(left_brow_h), abs(right_brow_h)) + 1e-6)

    # TÜM YÜZ İÇİN GLOBAL ASİMETRİ:
    # Birçok sol–sağ çiftinin dikey asimetrisini alıp ortalıyoruz.
    pair_asym = []
    for li, ri in LEFT_RIGHT_PAIRS:
        lp = get_point(li)
        rp = get_point(ri)
        base = calc_distance(lp, rp) + 1e-6
        dy = abs(lp[1] - rp[1]) / base  # dikey fark / aralarındaki mesafe
        pair_asym.append(dy)
    global_asym = float(np.mean(pair_asym)) if pair_asym else 0.0

    return eye_diff, mouth_diff, brow_diff, global_asym


def compute_score(eye_diff, mouth_diff, brow_diff, global_asym):
    # Global asimetriyi de skora dahil et
    return (
        0.3 * eye_diff
        + 0.2 * mouth_diff
        + 0.1 * brow_diff
        + 0.4 * global_asym
    )


def analyze_face(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, f"Görüntü okunamadı: {image_path}"

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    if not results.multi_face_landmarks:
        return None, "Yüz bulunamadı"

    h, w, _ = image.shape
    landmarks = results.multi_face_landmarks[0]

    def get_point(idx):
        pt = landmarks.landmark[idx]
        return np.array([pt.x * w, pt.y * h])

    eye_diff, mouth_diff, brow_diff, global_asym = extract_metrics(get_point)
    score = compute_score(eye_diff, mouth_diff, brow_diff, global_asym)
    return score, {
        "eye_diff": eye_diff,
        "mouth_diff": mouth_diff,
        "brow_diff": brow_diff,
        "global_asym": global_asym,
    }


def analyze_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    if not results.multi_face_landmarks:
        return None, "Yüz bulunamadı"

    h, w, _ = frame.shape
    landmarks = results.multi_face_landmarks[0]

    # Başın eğiklik (roll) açısını yaklaşık hesapla (sol–sağ göz merkezlerine bakarak)
    left_eye_pts = np.array(
        [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in LANDMARKS["left_eye"]]
    )
    right_eye_pts = np.array(
        [[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in LANDMARKS["right_eye"]]
    )
    left_center = left_eye_pts.mean(axis=0)
    right_center = right_eye_pts.mean(axis=0)
    dx = right_center[0] - left_center[0]
    dy = right_center[1] - left_center[1]
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))

    # Kafa fazla eğikse bu kareyi skora katma (sadece uyarı yaz)
    if abs(roll_deg) > 8.0:
        warning = f"Bas cok egik (roll={roll_deg:+.1f}°) - lutfen daha dik durun"
        cv2.putText(
            frame,
            warning,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        return None, {"reason": "head_tilt", "roll_deg": roll_deg}

    # Yüz mesh noktalarını ve konturları kareye çiz
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
    )

    # ÖZEL OLARAK KULLANILAN NOKTALARI VURGULA:
    # Gözler (mavi), Ağız (kırmızı), Kaşlar (yeşil)

    # Gözler (mavi)
    for idx in LANDMARKS["left_eye"] + LANDMARKS["right_eye"]:
        pt = landmarks.landmark[idx]
        pt_px = (int(pt.x * w), int(pt.y * h))
        cv2.circle(frame, pt_px, 2, (255, 0, 0), -1)

    # Ağız: köşeler + üst/alt dudak (kırmızı)
    mouth_pts = []
    for idx in LANDMARKS["mouth_corners"] + LANDMARKS["mouth_vertical"]:
        pt = landmarks.landmark[idx]
        pt_px = (int(pt.x * w), int(pt.y * h))
        mouth_pts.append(pt_px)
        cv2.circle(frame, pt_px, 4, (0, 0, 255), -1)
    # köşeler arasında çizgi
    if len(mouth_pts) >= 2:
        cv2.line(frame, mouth_pts[0], mouth_pts[1], (0, 0, 255), 2)

    # Kaş (yeşil)
    for idx in LANDMARKS["brow_left"] + LANDMARKS["brow_right"]:
        pt = landmarks.landmark[idx]
        pt_px = (int(pt.x * w), int(pt.y * h))
        cv2.circle(frame, pt_px, 3, (0, 255, 0), -1)

    def get_point(idx):
        pt = landmarks.landmark[idx]
        return np.array([pt.x * w, pt.y * h])

    eye_diff, mouth_diff, brow_diff, global_asym = extract_metrics(get_point)
    score = compute_score(eye_diff, mouth_diff, brow_diff, global_asym)
    return score, {
        "eye_diff": eye_diff,
        "mouth_diff": mouth_diff,
        "brow_diff": brow_diff,
        "global_asym": global_asym,
    }


def collect_scores():
    felc_scores, saglikli_scores = [], []
    for path in sorted(Path("ornekler/felc").glob("*")):
        score, _ = analyze_face(path)
        if score is not None:
            felc_scores.append(score)
    for path in sorted(Path("ornekler/saglikli").glob("*")):
        score, _ = analyze_face(path)
        if score is not None:
            saglikli_scores.append(score)
    return felc_scores, saglikli_scores


def calibrate_from_dataset():
    """
    ornekler/felc ve ornekler/saglikli klasörlerinden skor istatistikleri toplar,
    otomatik bir global eşik ve kalibrasyon parametreleri döner.
    """
    felc_scores, saglikli_scores = collect_scores()
    if not felc_scores or not saglikli_scores:
        print("Kalibrasyon için yeterli veri yok.")
        return None

    felc_scores = np.array(felc_scores, dtype=float)
    saglikli_scores = np.array(saglikli_scores, dtype=float)

    felc_mean = float(np.mean(felc_scores))
    sag_mean = float(np.mean(saglikli_scores))

    # Basit global eşik: iki ortalamanın ortası
    global_threshold = (felc_mean + sag_mean) / 2.0

    print("\n[Kalibrasyon]")
    print(f"- Felç   ort/min/max: {felc_mean:.3f}  {np.min(felc_scores):.3f}  {np.max(felc_scores):.3f}")
    print(f"- Sağlıklı ort/min/max: {sag_mean:.3f}  {np.min(saglikli_scores):.3f}  {np.max(saglikli_scores):.3f}")
    print(f"- Önerilen global eşik (midpoint): {global_threshold:.3f}")

    return {
        "felc_mean": felc_mean,
        "sag_mean": sag_mean,
        "global_threshold": global_threshold,
    }


def score_to_probability(score, felc_mean, sag_mean):
    """
    Skoru [0,1] arasında 'felç olasılığı'na map eder.
    sag_mean civarı ≈ 0, felc_mean civarı ≈ 1 olacak şekilde lineer ölçekleme.
    """
    denom = (felc_mean - sag_mean)
    if abs(denom) < 1e-6:
        return 0.5  # veri anlamsızsa nötr

    p = (score - sag_mean) / denom
    p = float(np.clip(p, 0.0, 1.0))
    return p


def run_webcam(calibration=None, duration=5.0):
    """
    calibration: None ise eski sabit eşik (0.25) kullanılır.
                 dict ise calibrate_from_dataset() çıktısı beklenir.
    """
    global next_clicked

    if calibration is not None:
        threshold = calibration["global_threshold"]
        felc_mean = calibration["felc_mean"]
        sag_mean = calibration["sag_mean"]
        print(f"\nKamera, kalibre edilmiş global eşik ile çalışıyor: threshold={threshold:.3f}")
    else:
        threshold = 0.25
        felc_mean = None
        sag_mean = None
        print(f"\nKamera, sabit eşik ile çalışıyor: threshold={threshold:.3f}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı")
        return

    window_name = "Yuz Felci Analizi - Egzersiz"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Her faz için: ekrandaki metin, anahtar isim, renk
    # 0. faz: NÖTR – yüzün başlangıç hali
    phases = [
        ("Lütfen NÖTR bakın, yüzünüzü rahat birakın 🙂", "notr", (200, 200, 200)),
        ("Lütfen GÜLÜMSEYİN 😀", "gülümseme", (0, 255, 255)),
        ("Lütfen KAŞLARINIZI KALDIRIN 😯", "kaş_kaldırma", (0, 255, 0)),
        ("Lütfen KAŞLARINIZI ÇATIN 😠", "kaş_çatma", (0, 165, 255)),
        ("Lütfen DUDAKLARINIZI BÜZÜN 😗", "dudak_büzme", (255, 0, 255)),
    ]

    print("Kamera başlatıldı. Sırayla şu hareketleri yapman istenecek:")
    print("0) Nötr bakış, 1) Gülümseme, 2) Kaş kaldırma, 3) Kaş çatma, 4) Dudak büzme")
    print("Her faz, 'Sonrakine geç' butonuna tıklayana veya 'n' tusuna basana kadar sürecek.\n")

    # Toplam skorlar ve fazlara göre skorlar
    all_scores = []
    all_probs = []
    phase_scores = {key: [] for _, key, _ in phases}
    phase_probs = {key: [] for _, key, _ in phases}

    # Faz bazlı bileşen metrikleri (nötre göre fark bakmak için)
    phase_components = {
        key: {"eye": [], "mouth": [], "brow": [], "yuz": []}
        for _, key, _ in phases
    }

    stop_all = False

    for instruction, phase_key, color in phases:
        print(f"Faz: {phase_key} -> {instruction}")
        next_clicked = False

        while True:
            ret, frame = cap.read()
            if not ret:
                stop_all = True
                break

            score, details = analyze_frame(frame)

            # Kafa çok eğikse details dict değil, string/dict uyarı gelebiliyor; o kareyi atla
            if score is not None and isinstance(details, dict):
                all_scores.append(score)
                phase_scores[phase_key].append(score)

                # bileşenleri kaydet (nötre göre fark için)
                phase_components[phase_key]["eye"].append(details["eye_diff"])
                phase_components[phase_key]["mouth"].append(details["mouth_diff"])
                phase_components[phase_key]["brow"].append(details["brow_diff"])
                phase_components[phase_key]["yuz"].append(details["global_asym"])

                # Eğer kalibrasyon varsa olasılık hesapla
                if felc_mean is not None and sag_mean is not None:
                    prob = score_to_probability(score, felc_mean, sag_mean)
                    all_probs.append(prob)
                    phase_probs[phase_key].append(prob)
                    prob_text = f"Felç olasiligi: {prob*100:5.1f}%"
                else:
                    prob = None
                    prob_text = ""

                status = "FELÇ (anlık)" if score > threshold else "Sağlıklı (anlık)"
                info = (
                    f"skor={score:.3f}  "
                    f"göz={details['eye_diff']:.3f}  "
                    f"ağız={details['mouth_diff']:.3f}  "
                    f"kaş={details['brow_diff']:.3f}  "
                    f"yüz={details['global_asym']:.3f}"
                )

                # --- AĞIZ ve KAŞ İÇİN SERT FELÇ SİNYALİ ---
                strong_mouth = details["mouth_diff"] > 0.10
                strong_brow = details["brow_diff"] > 0.10

                if strong_mouth or strong_brow:
                    alert = "GUVENLI FELC BULGUSU: "
                    parts = []
                    if strong_mouth:
                        parts.append("agiz")
                    if strong_brow:
                        parts.append("kas")
                    alert += ", ".join(parts)
                    cv2.putText(
                        frame,
                        alert,
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    # İstersen burada status'u da zorla "FELÇ (kural)" yapabilirsin
                    # status = "FELÇ (kural)"

                # Anlık durum
                cv2.putText(
                    frame,
                    status,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if score <= threshold else (0, 0, 255),
                    2,
                )
                # Metrikler (ham değerler)
                cv2.putText(
                    frame,
                    info,
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                # Olasılık metni (varsa)
                if prob is not None:
                    cv2.putText(
                        frame,
                        prob_text,
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                # HESAP NASIL YAPILIYOR? KISACA AÇIKLAMA
                cv2.putText(
                    frame,
                    "Skor = 0.3*göz + 0.2*ağız + 0.1*kaş + 0.4*yüz",
                    (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 200),
                    2,
                )
                cv2.putText(
                    frame,
                    "göz/ağız/kaş/yüz: sol-sag asimetri (0≈simetrik, yuksek≈asimetrik)",
                    (20, 215),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 255),
                    1,
                )

            # Kullanıcıya ne yapmasını söylediğimiz büyük yazı
            cv2.putText(
                frame,
                instruction,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            # "Sonrakine geç" butonu
            x1, y1, x2, y2 = NEXT_BTN_RECT
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), -1)
            cv2.putText(
                frame,
                "Sonrakine gec",
                (x1 + 10, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC -> tüm süreci bitir
                stop_all = True
                break
            if key == ord("n") or next_clicked:  # 'n' veya buton
                break

        if stop_all or not ret:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not all_scores:
        print("Yüz algılanamadı, sonuç üretilemedi.")
        return

    # Genel ortalama skor
    avg_score = float(np.mean(all_scores))
    result = "Felç bulgusu var" if avg_score > threshold else "Sağlıklı"
    print(f"\nToplam {len(all_scores)} geçerli kare analiz edildi.")
    print(f"GENEL Ortalama skor: {avg_score:.3f}  ->  {result}")

    # Genel ortalama olasılık
    if all_probs:
        avg_prob = float(np.mean(all_probs))
        print(f"GENEL Ortalama felç olasiligi: {avg_prob*100:5.1f}%")

    # NÖTR faz ortalamaları
    neutral = phase_components.get("notr")
    if neutral and neutral["eye"]:
        notr_eye = float(np.mean(neutral["eye"]))
        notr_mouth = float(np.mean(neutral["mouth"]))
        notr_brow = float(np.mean(neutral["brow"]))
        notr_yuz = float(np.mean(neutral["yuz"]))
        print("\nNÖTR faz ortalamalari:")
        print(
            f"  göz={notr_eye:.3f}, ağız={notr_mouth:.3f}, "
            f"kaş={notr_brow:.3f}, yüz={notr_yuz:.3f}"
        )
    else:
        notr_eye = notr_mouth = notr_brow = notr_yuz = None
        print("\nNÖTR fazdan veri alinamadi.")

    # Faz bazlı ortalamalar ve NÖTR'e göre farklar
    print("\nFaz bazlı ortalamalar (ve NÖTR'e göre farklar):")
    for instruction, phase_key, _ in phases:
        scores = phase_scores[phase_key]
        comps = phase_components[phase_key]
        if scores:
            mean_eye = float(np.mean(comps["eye"])) if comps["eye"] else 0.0
            mean_mouth = float(np.mean(comps["mouth"])) if comps["mouth"] else 0.0
            mean_brow = float(np.mean(comps["brow"])) if comps["brow"] else 0.0
            mean_yuz = float(np.mean(comps["yuz"])) if comps["yuz"] else 0.0

            line = (
                f"- {phase_key:12s}: skor={np.mean(scores):.3f} (n={len(scores)})  "
                f"göz={mean_eye:.3f}, ağız={mean_mouth:.3f}, "
                f"kaş={mean_brow:.3f}, yüz={mean_yuz:.3f}"
            )

            # NÖTR'e göre fark
            if notr_eye is not None:
                d_eye = mean_eye - notr_eye
                d_mouth = mean_mouth - notr_mouth
                d_brow = mean_brow - notr_brow
                d_yuz = mean_yuz - notr_yuz
                line += (
                    f"  |  Δgöz={d_eye:+.3f}, Δağız={d_mouth:+.3f}, "
                    f"Δkaş={d_brow:+.3f}, Δyüz={d_yuz:+.3f}"
                )

            # olasılık ortalaması
            probs = phase_probs[phase_key]
            if probs:
                line += f", ort_olasilik={np.mean(probs)*100:5.1f}%"

            print(line)
        else:
            print(f"- {phase_key:12s}: veri yok")


if __name__ == "__main__":
    example = Path("ornekler/felc/felc1.jpg")
    score, details = analyze_face(example)
    if score is None:
        print(details)
    else:
        print(f"Örnek resim skoru: {score:.3f}")
        print(details)

    # Dataset'ten kalibrasyon dene
    calibration = calibrate_from_dataset()

    # Kalibrasyon başarılıysa onunla, değilse sabit eşikle çalış
    if calibration is not None:
        run_webcam(calibration=calibration, duration=5.0)
    else:
        run_webcam(calibration=None, duration=5.0)