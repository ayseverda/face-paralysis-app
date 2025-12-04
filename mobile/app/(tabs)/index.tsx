// @ts-nocheck
import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  Image,
} from "react-native";
import { useRouter } from "expo-router";

const MODULES = [
  {
    id: "ai-test",
    title: "Yapay Zeka Testi",
    description: "Yüz felci asimetrisini yapay zeka ile analiz edin",
    icon: "🤖",
    color: "#17a2b8", // Teal
    gradient: ["#17a2b8", "#138496"],
    route: "/test",
  },
  {
    id: "exercises",
    title: "Yüz Egzersizleri",
    description: "Düzenli egzersizlerle yüz kaslarınızı güçlendirin",
    icon: "💪",
    color: "#6a1b9a", // Mor
    gradient: ["#6a1b9a", "#8e24aa"],
    route: "/exercises",
  },
  {
    id: "medications",
    title: "İlaç Hatırlatmaları",
    description: "İlaçlarınızı zamanında almayı unutmayın",
    icon: "💊",
    color: "#ff7f50", // Koral/Turuncu
    gradient: ["#ff7f50", "#ff6347"],
    route: "/medications",
  },
  {
    id: "progress",
    title: "İlerleme Takibi",
    description: "Tedavi sürecinizi takip edin ve gelişimi görün",
    icon: "📊",
    color: "#90ee90", // Açık Yeşil
    gradient: ["#90ee90", "#7cb342"],
    route: "/progress",
  },
];

export default function Index() {
  const router = useRouter();

  return (
    <ScrollView style={styles.scrollView} contentContainerStyle={styles.scrollContent}>
      <View style={styles.container}>
        {/* Logo ve Başlık Bölümü */}
        <View style={styles.header}>
          <View style={styles.logoContainer}>
            <Text style={styles.logoEmoji}>🎭</Text>
          </View>
          <Text style={styles.appTitle}>Yüz Felci Takip</Text>
          <Text style={styles.appSubtitle}>
            Sağlığınızı takip edin, iyileşmenizi görün
          </Text>
        </View>

        {/* Modül Kartları */}
        <View style={styles.modulesContainer}>
          {MODULES.map((module) => (
            <TouchableOpacity
              key={module.id}
              style={[styles.moduleCard, { borderLeftColor: module.color }]}
              onPress={() => {
                if (module.route === "/test") {
                  router.push(module.route);
                } else {
                  // Diğer modüller henüz hazır değil
                  alert(`${module.title} modülü yakında eklenecek!`);
                }
              }}
              activeOpacity={0.8}
            >
              <View style={styles.moduleIconContainer}>
                <Text style={styles.moduleIcon}>{module.icon}</Text>
              </View>
              <View style={styles.moduleContent}>
                <Text style={styles.moduleTitle}>{module.title}</Text>
                <Text style={styles.moduleDescription}>{module.description}</Text>
              </View>
              <View style={[styles.moduleArrow, { backgroundColor: module.color }]}>
                <Text style={styles.arrowText}>→</Text>
              </View>
            </TouchableOpacity>
          ))}
        </View>

        {/* Alt Bilgi */}
        <View style={styles.footer}>
          <Text style={styles.footerText}>
            Bu uygulama tıbbi teşhisin yerini tutmaz.
          </Text>
          <Text style={styles.footerText}>
            Sağlık sorunlarınız için mutlaka bir doktora danışın.
          </Text>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
    backgroundColor: "#f5f5f5",
  },
  scrollContent: {
    paddingBottom: 40,
  },
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingTop: 60,
  },
  header: {
    alignItems: "center",
    marginBottom: 40,
  },
  logoContainer: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: "#fff",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 20,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 5,
  },
  logoEmoji: {
    fontSize: 50,
  },
  appTitle: {
    fontSize: 28,
    fontWeight: "700",
    color: "#2c3e50",
    marginBottom: 8,
    textAlign: "center",
  },
  appSubtitle: {
    fontSize: 16,
    color: "#7f8c8d",
    textAlign: "center",
    paddingHorizontal: 20,
  },
  modulesContainer: {
    gap: 16,
    marginBottom: 30,
  },
  moduleCard: {
    flexDirection: "row",
    backgroundColor: "#fff",
    borderRadius: 16,
    padding: 20,
    alignItems: "center",
    borderLeftWidth: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  moduleIconContainer: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#f8f9fa",
    justifyContent: "center",
    alignItems: "center",
    marginRight: 16,
  },
  moduleIcon: {
    fontSize: 30,
  },
  moduleContent: {
    flex: 1,
  },
  moduleTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#2c3e50",
    marginBottom: 4,
  },
  moduleDescription: {
    fontSize: 14,
    color: "#7f8c8d",
    lineHeight: 20,
  },
  moduleArrow: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
    marginLeft: 12,
  },
  arrowText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "600",
  },
  footer: {
    marginTop: 20,
    paddingTop: 20,
    borderTopWidth: 1,
    borderTopColor: "#e0e0e0",
    alignItems: "center",
  },
  footerText: {
    fontSize: 12,
    color: "#95a5a6",
    textAlign: "center",
    marginBottom: 4,
    lineHeight: 18,
  },
});
