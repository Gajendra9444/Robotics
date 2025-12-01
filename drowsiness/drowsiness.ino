const int relayPin = 7;

void setup() {
  Serial.begin(9600);
  pinMode(relayPin, OUTPUT);
  digitalWrite(relayPin, LOW); // Start with relay off
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'A') {
      // Activate alert
      digitalWrite(relayPin, HIGH);
      delay(5000); // Vibrate for 5 second
      digitalWrite(relayPin, LOW);
    }
  }
}