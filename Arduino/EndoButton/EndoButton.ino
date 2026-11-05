/*
  EndoButton - hardware push button for the Endoscopy App (Arduino Uno)

  Wiring:
    Push button leg 1  -> digital pin 2
    Push button leg 2  -> GND
    (No resistor needed - the internal pull-up is used. Pin reads LOW when pressed.)
    Built-in LED (pin 13) lights while the button is held, and blinks when a
    long press is registered.

  Serial protocol (9600 baud, each message ends with '\n'):
    "SHORT"     short press   -> app takes a snapshot
    "LONG"      long press    -> app starts / stops recording
    "ENDO_BTN"  identification, sent on boot and as a reply to "?"
                (the app's "Auto-detect" option looks for this)

  LONG is sent as soon as the button has been held for LONG_PRESS_MS, so the
  user gets immediate feedback and does not need to release first.
  SHORT is sent on release if the press was shorter than LONG_PRESS_MS.
*/

const uint8_t BUTTON_PIN = 2;
const uint8_t LED_PIN = LED_BUILTIN;

const unsigned long DEBOUNCE_MS   = 30;   // ignore contact bounce
const unsigned long LONG_PRESS_MS = 800;  // hold time for a long press

bool stablePressed = false;     // debounced state
bool lastRawPressed = false;
unsigned long lastRawChange = 0;
unsigned long pressStart = 0;
bool longSent = false;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(9600);
  Serial.println("ENDO_BTN");
}

void blinkLed() {
  for (int i = 0; i < 2; i++) {
    digitalWrite(LED_PIN, LOW);  delay(60);
    digitalWrite(LED_PIN, HIGH); delay(60);
  }
}

void handleSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '?') Serial.println("ENDO_BTN");
  }
}

void loop() {
  handleSerial();

  unsigned long now = millis();
  bool rawPressed = (digitalRead(BUTTON_PIN) == LOW);

  // --- debounce ---
  if (rawPressed != lastRawPressed) {
    lastRawPressed = rawPressed;
    lastRawChange = now;
  }

  if ((now - lastRawChange) >= DEBOUNCE_MS && rawPressed != stablePressed) {
    stablePressed = rawPressed;

    if (stablePressed) {
      // pressed
      pressStart = now;
      longSent = false;
      digitalWrite(LED_PIN, HIGH);
    } else {
      // released
      digitalWrite(LED_PIN, LOW);
      if (!longSent) {
        Serial.println("SHORT");
      }
    }
  }

  // --- long press: fire once while still held ---
  if (stablePressed && !longSent && (now - pressStart) >= LONG_PRESS_MS) {
    longSent = true;
    Serial.println("LONG");
    blinkLed();
  }
}
