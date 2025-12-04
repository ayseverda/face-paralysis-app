// @ts-nocheck
import React, { useEffect, useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Image,
  Alert,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useRouter } from "expo-router";

const API_URL = "http://192.168.50.216:8000/analyze-image";

const POSES = [
  {
    key: "neutral1",
    label: "Nötr 1",
    instruction: "Lütfen nötr ve rahat bir şekilde bakın.",
  },
  {
    key: "smile",
    label: "Gülümse",
    instruction: "Lütfen dişlerinizi göstermeden gülümseyin.",
  },
  {
    key: "brow_up",
    label: "Kaş kaldır",
    instruction: "Lütfen kaşlarınızı olabildiğince yukarı kaldırın.",
  },
  {
    key: "brow_frown",
    label: "Kaş çat",
    instruction: "Lütfen kaşlarınızı çatın.",
  },
  {
    key: "pucker",
    label: "Dudak büz",
    instruction: "Lütfen dudaklarınızı öne doğru büzün.",
  },
  {
    key: "neutral2",
    label: "Nötr 2",
    instruction: "Tekrar nötr ve rahat durun.",
  },
];

export default function TestScreen() {
  const router = useRouter();
  const [permission, requestPermission] = useCameraPermissions();
  const hasPermission = permission?.granted ?? null;
  const [isSending, setIsSending] = useState(false);
  const [annotatedUri, setAnnotatedUri] = useState<string | null>(null);
  const [poseIndex, setPoseIndex] = useState(0);
  const [poseResults, setPoseResults] = useState<any[]>([]);
  const cameraRef = useRef<CameraView | null>(null);

  useEffect(() => {
    if (!permission) {
      requestPermission();
    }
  }, [permission]);

  const takeAndSendPhoto = async () => {
    if (!cameraRef.current) return;
    try {
      setIsSending(true);
      setAnnotatedUri(null);

      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
      });

      const filename = photo.uri.split("/").pop() ?? "face.jpg";
      const match = /\.(\w+)$/.exec(filename);
      const ext = match?.[1]?.toLowerCase() ?? "jpg";
      const mimeType = ext === "png" ? "image/png" : "image/jpeg";

      const formData = new FormData();
      formData.append("file", {
        uri: photo.uri,
        name: filename,
        type: mimeType,
      } as any);
      formData.append("pose", POSES[poseIndex].key);

      const res = await fetch(API_URL, {
        method: "POST",
        headers: { Accept: "application/json" },
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        console.log("API error:", text);
        Alert.alert("Hata", "Sunucu isteği başarısız oldu.");
        return;
      }

      const json = await res.json();

      if (json.success && json.annotated_image) {
        setAnnotatedUri(`data:image/jpeg;base64,${json.annotated_image}`);
      }

      if (!json.success) {
        Alert.alert("Analiz başarısız", json.reason ?? "Bilinmeyen hata");
        return;
      }

      const currentPose = POSES[poseIndex];
      const updated = [
        ...poseResults,
        { pose: currentPose, score: json.score, reason: json.reason },
      ];
      setPoseResults(updated);

      if (poseIndex < POSES.length - 1) {
        setPoseIndex((prev) => prev + 1);
      } else {
        // Tüm pozlar tamamlandı, sonuç ekranına geç
        router.replace({
          pathname: "/result",
          params: { data: JSON.stringify(updated) },
        });
      }
    } catch (e) {
      console.error(e);
      Alert.alert("Hata", "Fotoğraf gönderilirken bir sorun oluştu.");
    } finally {
      setIsSending(false);
    }
  };

  if (hasPermission === null) {
    return (
      <View style={styles.center}>
        <Text>Kamera izni kontrol ediliyor...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.center}>
        <Text>Kameraya erişim izni yok.</Text>
      </View>
    );
  }

  const current = POSES[poseIndex];

  return (
    <View style={styles.container}>
      <Text style={styles.poseText}>
        Adım {poseIndex + 1}/{POSES.length}: {current.instruction}
      </Text>

      <CameraView
        ref={(ref) => (cameraRef.current = ref)}
        style={styles.camera}
        facing="front"
      />

      <View style={styles.bottomPanel}>
        <TouchableOpacity
          style={[styles.smallButton, styles.cancelButton]}
          onPress={() => router.replace("/")}
          disabled={isSending}
        >
          <Text style={styles.smallButtonText}>İptal</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, isSending && styles.buttonDisabled]}
          onPress={takeAndSendPhoto}
          disabled={isSending}
        >
          {isSending ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.buttonText}>
              {poseIndex < POSES.length - 1 ? "Fotoğraf çek" : "Son fotoğraf"}
            </Text>
          )}
        </TouchableOpacity>
      </View>

      {annotatedUri && (
        <View style={styles.previewPanel}>
          <Image source={{ uri: annotatedUri }} style={styles.annotated} />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  poseText: {
    color: "#fff",
    paddingHorizontal: 12,
    paddingTop: 8,
    paddingBottom: 4,
    fontSize: 14,
  },
  camera: { flex: 4, width: "100%" },
  bottomPanel: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: "#111",
  },
  button: {
    flex: 1,
    backgroundColor: "#2e86de",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "600" },
  smallButton: {
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    marginRight: 8,
  },
  cancelButton: {
    backgroundColor: "#555",
  },
  smallButtonText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "500",
  },
  previewPanel: {
    flex: 2,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: "#fff",
  },
  annotated: {
    width: "100%",
    height: "100%",
    resizeMode: "contain",
  },
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
});


