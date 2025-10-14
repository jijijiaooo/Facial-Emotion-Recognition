#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define WIFI_SSID "Trisha"
#define WIFI_PASS "trisha22"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

WebServer server(8080);

// --- Global variable to store the last emotion received ---
String lastEmotion = "";

// --- Handle POST /emotion from Raspberry Pi ---
void handleEmotion() {
  String payload = server.arg("plain");
  lastEmotion = payload;  // Save for Android GET requests

  int spaceIdx = payload.indexOf(' ');
  String emoji = (spaceIdx != -1) ? payload.substring(0, spaceIdx) : payload;
  String emotion = (spaceIdx != -1) ? payload.substring(spaceIdx + 1) : "";

  display.clearDisplay();
  display.setTextSize(2); // Large font for both

  // Center emoji (top)
  int16_t x1, y1;
  uint16_t w, h;
  display.getTextBounds(emoji.c_str(), 0, 0, &x1, &y1, &w, &h);
  int16_t emoji_x = (SCREEN_WIDTH - w) / 2;
  int16_t emoji_y = 4; // Top area
  display.setCursor(emoji_x, emoji_y);
  display.print(emoji.c_str());

  // Center emotion word (below)
  display.getTextBounds(emotion.c_str(), 0, 0, &x1, &y1, &w, &h);
  int16_t word_x = (SCREEN_WIDTH - w) / 2;
  int16_t word_y = emoji_y + 28; // Spacing below emoji
  display.setCursor(word_x, word_y);
  display.print(emotion.c_str());

  display.display();

  server.send(200, "text/plain", "OK");
}

// --- Handle GET /get_emotion from Android app ---
void handleGetEmotion() {
  if (lastEmotion == "") {
    server.send(200, "text/plain", "No emotion received yet");
  } else {
    server.send(200, "text/plain", lastEmotion);
  }
}

void setup() {
  Serial.begin(115200);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("SSD1306 allocation failed");
    for (;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,10);
  display.println("Waiting for WiFi...");
  display.display();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  display.clearDisplay();
  display.setCursor(0,10);
  display.println("WiFi Connected!");
  display.setCursor(0,30);
  display.print(WiFi.localIP());
  display.display();

  // --- Register endpoints ---
  server.on("/emotion", HTTP_POST, handleEmotion);      // RPi sends emotion here
  server.on("/get_emotion", HTTP_GET, handleGetEmotion); // Android fetches here
  server.begin();

  Serial.println("HTTP server started");
  Serial.println("POST to /emotion  (from RPi)");
  Serial.println("GET  /get_emotion (from Android)");
}

void loop() {
  server.handleClient();
}
