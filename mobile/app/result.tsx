// @ts-nocheck
import React, { useMemo } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";

type PoseResult = {
  pose: { key: string; label: string };
  score: number;
  reason?: string;
};

export default function ResultScreen() {
  const router = useRouter();
  const params = useLocalSearchParams();

  const results: PoseResult[] = useMemo(() => {
    try {
      if (typeof params.data === "string") {
        return JSON.parse(params.data);
      }
      return [];
    } catch {
      return [];
    }
  }, [params.data]);

  const summary = useMemo(() => {
    if (!results.length) return null;
    const avg =
      results.reduce((s, r) => s + (r.score || 0), 0) / results.length;
    const pct = Math.max(0, Math.min(1, avg)) * 100;
    const high = avg > 0.25;
    return {
      avg,
      pct,
      text: high
        ? "Genel olarak felç bulgusu olma olasılığı YÜKSEK."
        : "Genel olarak felç bulgusu olma olasılığı DÜŞÜK.",
    };
  }, [results]);

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Test Sonucu</Text>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.content}>
        {!results.length && (
          <Text style={styles.text}>
            Görüntülenecek veri bulunamadı. Lütfen testi tekrar deneyin.
          </Text>
        )}

        {results.map((item, idx) => (
          <View key={item.pose.key + idx} style={styles.card}>
            <Text style={styles.cardTitle}>
              {idx + 1}. {item.pose.label} – skor: {item.score.toFixed(3)}
            </Text>
            {item.reason && (
              <Text style={styles.cardText}>{item.reason}</Text>
            )}
          </View>
        ))}

        {summary && (
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>
              Ortalama skor: {summary.avg.toFixed(3)} (
              {summary.pct.toFixed(1)}%)
            </Text>
            <Text style={styles.summaryText}>{summary.text}</Text>
          </View>
        )}
      </ScrollView>

      <View style={styles.bottom}>
        <TouchableOpacity
          style={styles.button}
          onPress={() => router.replace("/")}
        >
          <Text style={styles.buttonText}>Başa dön</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  title: {
    color: "#fff",
    fontSize: 20,
    fontWeight: "700",
    textAlign: "center",
    marginTop: 16,
    marginBottom: 8,
  },
  scroll: { flex: 1 },
  content: {
    paddingHorizontal: 12,
    paddingBottom: 24,
  },
  text: { color: "#fff", textAlign: "center", marginTop: 16 },
  card: {
    backgroundColor: "#111",
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
  },
  cardTitle: {
    color: "#fff",
    fontWeight: "600",
    marginBottom: 4,
  },
  cardText: {
    color: "#ccc",
    fontSize: 13,
  },
  summaryCard: {
    backgroundColor: "#2e86de",
    borderRadius: 10,
    padding: 12,
    marginTop: 8,
  },
  summaryTitle: {
    color: "#fff",
    fontWeight: "700",
    marginBottom: 4,
  },
  summaryText: {
    color: "#fff",
  },
  bottom: {
    padding: 12,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: "#333",
    backgroundColor: "#000",
  },
  button: {
    backgroundColor: "#2e86de",
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
});


