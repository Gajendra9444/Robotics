#include <ESP8266WiFi.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <time.h>

// -------- OLED --------
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// -------- WiFi --------
const char* ssid = "TP-LINK_DA28";
const char* password = "59555803";

// -------- Time --------
const char* ntpServer = "pool.ntp.org";
const long  gmtOffset_sec = 5 * 3600 + 45 * 60;   // Nepal Time (GMT +5:45)
const int   daylightOffset_sec = 0;

void setup() {
  Serial.begin(115200);

  // OLED Init (NodeMCU I2C pins)
  Wire.begin(D2, D1);   // SDA, SCL

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found");
    while (true);
  }

  display.clearDisplay();
  display.setTextColor(WHITE);

  // WiFi Connect
  display.setTextSize(1);
  display.setCursor(0, 20);
  display.println("Connecting WiFi...");
  display.display();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }

  // Time Init
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  display.clearDisplay();
  display.setCursor(0, 20);
  display.println("Smart Watch Ready");
  display.display();
  delay(1500);
}

void loop() {
  time_t now = time(nullptr);
  struct tm* timeinfo = localtime(&now);

  display.clearDisplay();

  // ---- TIME ----
  display.setTextSize(2);
  display.setCursor(10, 10);
  display.printf("%02d:%02d:%02d",
                 timeinfo->tm_hour,
                 timeinfo->tm_min,
                 timeinfo->tm_sec);

  // ---- DATE ----
  display.setTextSize(1);
  display.setCursor(25, 40);
  display.printf("%02d-%02d-%04d",
                 timeinfo->tm_mday,
                 timeinfo->tm_mon + 1,
                 timeinfo->tm_year + 1900);

  display.display();
  delay(1000);
}
