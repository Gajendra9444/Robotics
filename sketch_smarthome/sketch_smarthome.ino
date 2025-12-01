// Blynk template details
#define BLYNK_TEMPLATE_ID   "TMPL6yycwVkK4"
#define BLYNK_TEMPLATE_NAME " Robotech"      // You can change this to any name
#define BLYNK_AUTH_TOKEN    "hbFxPZyidE2AWexT6J5zvu6CsLFpBauJ"

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>

char auth[] = "hbFxPZyidE2AWexT6J5zvu6CsLFpBauJ";

// WiFi credentials
char ssid[] = "TP-Link_71ED_5G";   // Your WiFi name
char pass[] = "73439981";          // Your WiFi password

// Pin Definitions
int relayPin = 18;
int motorA   = 5;
int motorB   = 17;
int ledPin   = 2;

void setup() {
  Serial.begin(115200);

  // Pin Modes
  pinMode(relayPin, OUTPUT);
  pinMode(motorA, OUTPUT);
  pinMode(motorB, OUTPUT);
  pinMode(ledPin, OUTPUT);

  // Initial OFF States
  digitalWrite(relayPin, HIGH);  // Relay OFF (assuming active LOW)
  digitalWrite(motorA, LOW);
  digitalWrite(motorB, LOW);
  digitalWrite(ledPin, LOW);

  // Start Blynk
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);
}

// ********* BLYNK CONTROL FUNCTIONS *********

// Relay Control (V1)
BLYNK_WRITE(V1) {
  int value = param.asInt();
  // If button ON (1) -> relay LOW (ON if active-low relay)
  digitalWrite(relayPin, value == 1 ? LOW : HIGH);
}

// Motor Control (V2)
BLYNK_WRITE(V2) {
  int value = param.asInt();
  if (value == 1) {
    // Motor ON (one direction)
    digitalWrite(motorA, HIGH);
    digitalWrite(motorB, LOW);
  } else {
    // Motor OFF
    digitalWrite(motorA, LOW);
    digitalWrite(motorB, LOW);
  }
}

// LED Control (V3)
BLYNK_WRITE(V3) {
  int value = param.asInt();
  digitalWrite(ledPin, value);   // 1 = ON, 0 = OFF
}

void loop() {
  Blynk.run();
}