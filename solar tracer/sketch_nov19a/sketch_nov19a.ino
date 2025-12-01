#include <Servo.h>
#define LDR1 A0
#define LDR2 A1

#define error 10
int Spoint = 0;

Servo servo;

void setup() {
  servo.attach(11);
  Spoint = constrain(Spoint, 0, 180);
  servo.write(Spoint);
  delay(1000);
}

void loop() {
  int ldr1 = analogRead(LDR1);
  int ldr2 = analogRead(LDR2);
  int servoPosition1 = map(ldr1, 0, 1023, 0, 180);
  int servoPosition2 = map(ldr2, 0, 1023, 0, 180);
  
  if (abs(ldr1 - ldr2) <= error) {
    
  } else {
    
    if (ldr1 > ldr2) {
      Spoint = constrain(Spoint - 3, 0, 180); 
      
    } else {
      Spoint = constrain(Spoint + 3, 0, 180); 
    }
  }

  servo.write(Spoint);
  delay(40); 
}