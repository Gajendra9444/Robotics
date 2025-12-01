#include <WiFi.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Your WiFi credentials
const char* ssid = "TP-Link_71ED_5G";
const char* password = "73439981";

void setup() {
  Serial.begin(115200);

  Wire.begin(21, 22);   // SDA, SCL

  // Start OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)){
    Serial.println("SSD1306 failed");
    for(;;);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);

  display.setCursor(0, 0);
  display.println("Connecting WiFi...");
  display.display();

  WiFi.begin(ssid, password);

  unsigned long startAttempt = millis();

  // Try for 7 seconds max
  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 7000) {
    delay(500);
    Serial.print(".");
  }

  display.clearDisplay();
  display.setCursor(0, 0);

  if (WiFi.status() == WL_CONNECTED) {
    display.println("WiFi Connected!");
  } else {
    display.println("Failed to connect.");
  }
  
  display.display();
}

void loop() {
  // Add smartwatch tasks like time display here
}
