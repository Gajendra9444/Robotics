#include <SoftwareSerial.h>

// HC-05 on SoftwareSerial (RX, TX)
SoftwareSerial bt(2, 3); // HC-05 TX -> 2, HC-05 RX -> 3 (use voltage divider on Arduino->HC05 TX)

#define in1 5   // Motor A forward (PWM)
#define in2 6   // Motor A backward (PWM)
#define in3 10  // Motor B forward (PWM)
#define in4 11  // Motor B backward (PWM)
#define LED 13

char command = 0;        // store incoming char command
int Speed = 204;         // 0 - 255 default
int Turnradius = 120;    // inner wheel speed for turns (0..254); must be < Speed
int brakeTime = 45;      // ms for electronic brake pulse
char lastCommand = 0;
int brkonoff = 1;        // 1 enable brief electronic braking on 'S', 0 disable

void setup() {
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(LED, OUTPUT);

  digitalWrite(LED, LOW);
  Stop(); // ensure motors are stopped

  Serial.begin(9600); // debug via USB serial
  bt.begin(9600);     // HC-05 default (change if needed)

  Serial.println("BT car ready (SoftwareSerial on 2 RX, 3 TX).");
}

void loop() {
  // prefer bluetooth input
  if (bt.available()) {
    char c = bt.read();
    handleCommand(c);
    Serial.print("BT: ");
    Serial.println(c);
  }

  // allow USB serial for testing
  if (Serial.available()) {
    char c = Serial.read();
    handleCommand(c);
    Serial.print("USB: ");
    Serial.println(c);
  }
}

void handleCommand(char raw) {
  if (raw == '\r' || raw == '\n') return; // ignore newline chars from terminal apps
  command = raw;

  // enforce safe Turnradius bound
  if (Turnradius >= Speed) Turnradius = max(0, Speed - 1);

  switch (command) {
    case 'F': forward();      break;
    case 'B': back();         break;
    case 'L': left();         break;
    case 'R': right();        break;
    case 'G': forwardleft();  break; // forward-left (inner wheel slower)
    case 'I': forwardright(); break; // forward-right
    case 'H': backleft();     break;
    case 'J': backright();    break;
    case 'S': Stop();         break;

    // Speed presets 0..9 and q for max
    case '0': Speed = 100;  break;
    case '1': Speed = 140;  break;
    case '2': Speed = 153;  break;
    case '3': Speed = 165;  break;
    case '4': Speed = 178;  break;
    case '5': Speed = 191;  break;
    case '6': Speed = 204;  break;
    case '7': Speed = 216;  break;
    case '8': Speed = 229;  break;
    case '9': Speed = 242;  break;
    case 'q': Speed = 255;  break;

    // optional: adjust Turnradius on-the-fly with 't' + digit? (not implemented)
    default:
      // unknown command — ignore
      return;
  }

  // braking behavior (brief electronic brake) when enabled:
  if (brkonoff == 1) {
    if (command == 'S' && lastCommand != 'S') {
      // brief electronic brake: set both inputs same side HIGH briefly then stop
      // note: for L298N, IN1=IN2=HIGH causes braking for motor A, similarly for motor B.
      digitalWrite(in1, HIGH);
      digitalWrite(in2, HIGH);
      digitalWrite(in3, HIGH);
      digitalWrite(in4, HIGH);
      delay(brakeTime);
      Stop();
    }
  }

  lastCommand = command;
}

/* Motor helper functions
   Important: always set one input to speed and the opposite to 0,
   never nonzero on both at same time (except for short brake pulse above).
*/

void forward() {
  analogWrite(in1, Speed); // A forward
  analogWrite(in2, 0);

  analogWrite(in3, Speed); // B forward
  analogWrite(in4, 0);

  digitalWrite(LED, HIGH);
}

void back() {
  analogWrite(in1, 0);
  analogWrite(in2, Speed);

  analogWrite(in3, 0);
  analogWrite(in4, Speed);

  digitalWrite(LED, HIGH);
}

void left() {
  // rotate/turn left: left wheel backward, right wheel forward
  analogWrite(in1, 0);
  analogWrite(in2, Speed);   // left wheel backward

  analogWrite(in3, Speed);
  analogWrite(in4, 0);       // right wheel forward

  digitalWrite(LED, HIGH);
}

void right() {
  // rotate/turn right: left wheel forward, right wheel backward
  analogWrite(in1, Speed);
  analogWrite(in2, 0);       // left wheel forward

  analogWrite(in3, 0);
  analogWrite(in4, Speed);   // right wheel backward

  digitalWrite(LED, HIGH);
}

void forwardleft() {
  analogWrite(in1, Turnradius); // left slower forward
  analogWrite(in2, 0);

  analogWrite(in3, Speed);      // right full forward
  analogWrite(in4, 0);

  digitalWrite(LED, HIGH);
}

void forwardright() {
  analogWrite(in1, Speed);
  analogWrite(in2, 0);

  analogWrite(in3, Turnradius);
  analogWrite(in4, 0);

  digitalWrite(LED, HIGH);
}

void backleft() {
  analogWrite(in1, 0);
  analogWrite(in2, Turnradius);

  analogWrite(in3, 0);
  analogWrite(in4, Speed);

  digitalWrite(LED, HIGH);
}

void backright() {
  analogWrite(in1, 0);
  analogWrite(in2, Speed);

  analogWrite(in3, 0);
  analogWrite(in4, Turnradius);

  digitalWrite(LED, HIGH);
}

void Stop() {
  analogWrite(in1, 0);
  analogWrite(in2, 0);
  analogWrite(in3, 0);
  analogWrite(in4, 0);

  digitalWrite(LED, LOW);
}
