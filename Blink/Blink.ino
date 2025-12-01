// max30100 sensor aafno i2c peripherals bata 1.8v current matra dinxa, tyie vayera
// yo chai 5v logic vako microcontroller sanga kam gardaina. Which means arduino uno, nano, mega vanda pani 
// arduino pro mini ki ta nodemcu use garna parne hunxa yo duitai ko logic 3.3v ko hunxa ani SCL D1 ra SDA D2 nodemcu ko.

#include <Wire.h>
#include "MAX30100_PulseOximeter.h"
#include <U8g2lib.h>

#define REPORTING_PERIOD_MS 1000
#define FILTER_SIZE 5

// Initialize the display (for I2C)
U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE);

PulseOximeter pox;
uint32_t tsLastReport = 0;
float heartRateBuffer[FILTER_SIZE];
float spo2Buffer[FILTER_SIZE];
int filterIndex = 0;
float previousHeartRate = 0;
float previousSpO2 = 0;

#define ECG_BUFFER_SIZE 128
int ecgBuffer[ECG_BUFFER_SIZE];
int ecgIndex = 0;

void onBeatDetected() {
    Serial.println("Beat!");
}

float medianFilter(float *data, int size) {
    float sorted[size];
    memcpy(sorted, data, size * sizeof(float)); // yo chai median filter use garera noise reduce gareko.
    for (int i = 0; i < size - 1; i++) {
        for (int j = i + 1; j < size; j++) {
            if (sorted[i] > sorted[j]) {
                float temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }
    return sorted[size / 2];
}

float adaptiveFilter(float newValue, float oldValue, float alpha) { // ani adaptive filter chai purano ra naya signal lai blend garera smooth signal read gareko.
    return (alpha * newValue) + ((1 - alpha) * oldValue);
}

void setup() {
    Serial.begin(115200);
    u8g2.begin();  
    u8g2.clearBuffer();

    Serial.print("Initializing pulse oximeter...");
    if (!pox.begin()) {
        Serial.println("FAILED");
        for (;;);
    } else {
        Serial.println("SUCCESS");
    }

    pox.setIRLedCurrent(MAX30100_LED_CURR_24MA);  //yo chai current ko vale ko adjust garna parxa 
    pox.setOnBeatDetectedCallback(onBeatDetected);
}

void loop() {
    pox.update();

    float heartRate = pox.getHeartRate();
    float spo2 = pox.getSpO2();

    ecgBuffer[ecgIndex] = (int)(heartRate * sin(2 * 3.14159 * ecgIndex / ECG_BUFFER_SIZE));
    ecgIndex = (ecgIndex + 1) % ECG_BUFFER_SIZE;

    u8g2.clearBuffer();

    for (int i = 0; i < ECG_BUFFER_SIZE - 1; i++) {
        int y1 = map(ecgBuffer[i], 0, 180, 44, 20);
        int y2 = map(ecgBuffer[(i + 1) % ECG_BUFFER_SIZE], 0, 180, 44, 20);
        u8g2.drawLine(i, y1, i + 1, y2); 
    }

    u8g2.setFont(u8g2_font_ncenB08_tr); // Set font for text
    u8g2.setCursor(0, 10);
    u8g2.print("BPM: ");
    u8g2.print(heartRate);
    u8g2.setCursor(0, 20);
    u8g2.print("SpO2: ");
    u8g2.print(spo2);

    u8g2.sendBuffer();  // Refresh the display

    delay(20);  // Adjust for graph speed
}