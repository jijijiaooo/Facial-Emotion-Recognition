#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// -------------------------------
// WIFI CONFIG
// -------------------------------
#define WIFI_SSID "Trisha"
#define WIFI_PASS "trisha22"

// -------------------------------
// OLED CONFIG
// -------------------------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// -------------------------------
// FIREBASE REST API CONFIG
// -------------------------------
String firebaseSecret = "KkUdkmPAy48Vh8NQKXqROs8qieqL2bEmSVDUNN6w";
String firebaseURL = "https://pelioscope-emotion-default-rtdb.asia-southeast1.firebasedatabase.app/meta/last_emotion.json?auth=";

String lastEmotion = "";

// -------------------------------
// Helper: capitalize first letter
// -------------------------------
String capitalizeFirstLetter(String str) {
  if (str.length() == 0) return str;
  str.toLowerCase();
  str[0] = toupper(str[0]);
  return str;
}

// -------------------------------
// OLED helper function (centered)
// -------------------------------
void showEmotion(const String &emotionRaw) {
  String emotion = capitalizeFirstLetter(emotionRaw);
  display.clearDisplay();

  // Map emotion to emoji
  String emoji = "";
  if (emotion == "Happy") emoji = "^_^";
  else if (emotion == "Sad") emoji = "T_T";
  else if (emotion == "Angry") emoji = ">:[";
  else if (emotion == "Disgust") emoji = ">_<";
  else if (emotion == "Fear") emoji = "O_O";
  else if (emotion == "Surprise") emoji = "O.O";
  else if (emotion == "Neutral") emoji = "-_-";
  else emoji = emotion; // fallback

  // Display emotion name at top center
  display.setTextSize(2);
  display.setTextColor(SSD1306_WHITE);
  int16_t x1, y1;
  uint16_t w, h;

  display.getTextBounds(emotion, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, 0);
  display.println(emotion);

  // Display emoji at bottom center
  display.setTextSize(3);
  display.getTextBounds(emoji, 0, 0, &x1, &y1, &w, &h);
  display.setCursor((SCREEN_WIDTH - w) / 2, SCREEN_HEIGHT - h - 5); // 5px padding from bottom
  display.println(emoji);

  display.display();
}

// -------------------------------
// Read from Firebase via REST
// -------------------------------
String getEmotion() {
  HTTPClient http;
  String url = firebaseURL + firebaseSecret;

  http.begin(url);
  int code = http.GET();

  if (code == 200) {
    String result = http.getString();
    http.end();

    result.trim();
    result.replace("\"", ""); // remove quotes
    return result;
  } else {
    http.end();
    return "";
  }
}

// -------------------------------
// Setup
// -------------------------------
void setup() {
  Serial.begin(115200);

  // OLED Init
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed");
    while (true);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 10);
  display.println("Starting...");
  display.display();

  // WIFI
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected!");
  display.clearDisplay();
  display.setCursor(0, 10);
  display.println("WiFi Connected!");
  display.display();
}

// -------------------------------
// Loop
// -------------------------------
void loop() {
  String emotion = getEmotion();

  if (emotion != "" && emotion != lastEmotion) {
    lastEmotion = emotion;
    Serial.println("Emotion: " + emotion);
    showEmotion(emotion);
  }

  delay(500);
}
