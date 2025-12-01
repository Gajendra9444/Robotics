#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// Use D1 as SCL and D2 as SDA
#define OLED_SDA D2
#define OLED_SCL D1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

int screenMode = 0;
int buttonPin = D3;  // Button for switching screens
unsigned long lastUpdate = 0;

// Fake sensor values
int steps = 0;
int heartRate = 72;

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);

  // Start I2C on D1(SCL) and D2(SDA)
  Wire.begin(OLED_SDA, OLED_SCL);

  Serial.begin(115200);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println("OLED not found");
    for(;;);
  }

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Smartwatch Booting...");
  display.display();
  delay(1500);
}

void loop() {
  // Handle button press
  if (digitalRead(buttonPin) == LOW) {
    screenMode++;
    if (screenMode > 2) screenMode = 0;
    delay(300);
  }

  // Update fake sensor values every second
  if (millis() - lastUpdate >= 1000) {
    lastUpdate = millis();
    steps += random(1, 4);
    heartRate = random(65, 95);
  }

  display.clearDisplay();

  if (screenMode == 0) showTimeScreen();
  else if (screenMode == 1) showStepsScreen();
  else if (screenMode == 2) showHeartScreen();

  display.display();
}

// --------- Screens ---------

void showTimeScreen() {
  display.setTextSize(2);
  display.setCursor(10, 10);
  display.print("Time");

  display.setTextSize(3);
  display.setCursor(10, 35);
  display.print(getFakeTime());
}

void showStepsScreen() {
  display.setTextSize(2);
  display.setCursor(10, 10);
  display.print("Steps");

  display.setTextSize(3);
  display.setCursor(10, 35);
  display.print(steps);
}

void showHeartScreen() {
  display.setTextSize(2);
  display.setCursor(10, 10);
  display.print("Heart");

  display.setTextSize(3);
  display.setCursor(10, 35);
  display.print(heartRate);
  display.print(" bpm");
}

// ----- Fake time function -----
String getFakeTime() {
  long seconds = millis() / 1000;
  int h = (seconds / 3600) % 24;
  int m = (seconds / 60) % 60;

  char buffer[10];
  sprintf(buffer, "%02d:%02d", h, m);
  return String(buffer);
}
