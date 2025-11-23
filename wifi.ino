#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// -------------------------------
// WIFI
// -------------------------------
#define WIFI_SSID "Trisha"
#define WIFI_PASS "trisha22"

// -------------------------------
// OLED
// -------------------------------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// -------------------------------
// FIREBASE CONFIG
// -------------------------------
#define FIREBASE_API_KEY "AIzaSyBr5bJ26yvC5ZKl2k3mt97oSqKBjS5uQoM"
#define FIREBASE_DB_URL "https://pelioscope-emotion-default-rtdb.asia-southeast1.firebasedatabase.app"
#define FIREBASE_SECRET "KkUdkmPAy48Vh8NQKXqROs8qieqL2bEmSVDUNN6w"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

String lastEmotion = "";

void setupDisplay(String msg) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 10);
  display.println(msg);
  display.display();
}

void setup() {
  Serial.begin(115200);

  // OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED failed");
    while (true);
  }
  setupDisplay("Starting...");

  // WIFI
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  setupDisplay("Connecting WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  setupDisplay("WiFi Connected!");

  // FIREBASE
  config.api_key = FIREBASE_API_KEY;
  config.database_url = FIREBASE_DB_URL;
  config.signer.tokens.legacy_token = FIREBASE_SECRET;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  setupDisplay("Firebase Ready");
  delay(500);
}

void loop() {
  // Read /emotion_output/latest/value
  if (Firebase.RTDB.getString(&fbdo, "/emotion_output/latest/value")) {
    String emotion = fbdo.stringData();

    if (emotion != lastEmotion) {
      lastEmotion = emotion;

      Serial.println("Emotion: " + emotion);

      // Show on OLED
      display.clearDisplay();
      display.setTextSize(2);
      display.setCursor(0, 20);
      display.print(emotion);
      display.display();
    }
  }

  delay(400); // polling interval
}
