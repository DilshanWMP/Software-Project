/*
  EndoButton - hardware push button + recording LED for the Endoscopy App (Arduino Uno)

  Wiring:
    Push button leg 1  -> digital pin 2
    Push button leg 2  -> GND
    (No resistor needed - the internal pull-up is used. Pin reads LOW when pressed.)

    Green LED (+, long leg)  -> 220 ohm resistor -> digital pin 8
    Green LED (-, short leg) -> GND

    Built-in LED (pin 13) lights while the button is held, and flickers when a
    long press is registered.

  Green LED behaviour (controlled by the app, so it shows the REAL state):
    solid ON         recording is live
    OFF              not recording
    fast blink ~1 s  snapshot saved - then returns to ON (recording) or OFF
                     (works the same while recording or not)

  Serial protocol (9600 baud):
    Arduino -> PC (each line ends with '\n'):
      "SHORT"     short press   -> app takes a snapshot
      "LONG"      long press    -> app starts / stops recording
      "ENDO_BTN"  identification, sent on boot and as a reply to "?"
                  (the app's "Auto-detect" option looks for this)
    PC -> Arduino (single characters, newline optional):
      '1'  recording live    -> green LED ON
      '0'  recording stopped -> green LED OFF
      'p'  snapshot saved    -> green LED fast blink
      '?'  identify          -> replies "ENDO_BTN"
    The app re-sends '1'/'0' every 2 s. If nothing arrives from the PC for
    PC_TIMEOUT_MS (app closed, crashed or cable issue) the green LED turns OFF,
    so it never shows "recording" when the app isn't. (Disabled in debug mode
    so you can test the LED by hand.)

  LONG is sent as soon as the button has been held for LONG_PRESS_MS, so the
  user gets immediate feedback and does not need to release first.
  SHORT is sent on release if the press was shorter than LONG_PRESS_MS.

  ---------------------------------------------------------------------------
  DEBUG / TEST MODE  (test the button and LED WITHOUT the Endoscopy App)
  ---------------------------------------------------------------------------
  1. Close the Endoscopy App (only one program can use the COM port).
  2. Arduino IDE -> Tools -> Serial Monitor, set baud to 9600.
  3. Type a command in the Serial Monitor and press Enter:
       d  -> debug ON / OFF
       s  -> status report (pin state, counters, LED state, settings)
       r  -> reset counters and the simulated recording state
       h  -> help
       1 / 0 / p -> test the green LED (on / off / snapshot blink)
  4. Press the button. With debug ON you will see, for each press:
       # PRESSED
       # held 500 ms...                 (progress while holding)
       # RELEASED after 312 ms, 0 bounce(s) -> SHORT
       # [APP] snapshot #1 saved   (recording OFF)
     and for a long press:
       # LONG PRESS detected at 800 ms
       # [APP] recording START
     "[APP]" lines show what the Endoscopy App would do. In debug mode with
     no app connected, the green LED follows this simulation, so you can test
     the whole thing: long press = LED on, short press = fast blink, etc.

  All debug lines start with '#', and the app ignores them. The normal
  SHORT / LONG lines are always sent, so nothing changes for the app.
  Set DEBUG_DEFAULT to true to start in debug mode after every reset.
*/

const uint8_t BUTTON_PIN    = 2;
const uint8_t GREEN_LED_PIN = 8;
const uint8_t LED_PIN       = LED_BUILTIN;

const unsigned long DEBOUNCE_MS   = 30;    // ignore contact bounce
const unsigned long LONG_PRESS_MS = 800;   // hold time for a long press

const unsigned long SNAP_BLINK_MS     = 1000; // how long the snapshot blink lasts
const unsigned long SNAP_BLINK_PERIOD = 80;   // on/off time of each blink
const unsigned long PC_TIMEOUT_MS     = 5000; // no message from PC -> LED off

const bool DEBUG_DEFAULT = false;          // true = debug output on at power-up
const unsigned long HOLD_REPORT_MS = 500;  // debug: "held ... ms" interval

// ---- button state ----
bool stablePressed = false;     // debounced state
bool lastRawPressed = false;
unsigned long lastRawChange = 0;
unsigned long pressStart = 0;
bool longSent = false;

// ---- green LED state ----
bool recordingOn = false;            // what the green LED should show (from the PC)
unsigned long snapBlinkStart = 0;
bool snapBlinking = false;
unsigned long lastPcMessage = 0;     // millis() of last command from the PC
bool pcSeen = false;                 // has the app sent anything since boot?

// ---- built-in LED long-press flicker (non-blocking) ----
unsigned long builtinFlickerStart = 0;
bool builtinFlickering = false;

// ---- debug state ----
bool debugMode = DEBUG_DEFAULT;
unsigned int bounceCount = 0;        // raw edges seen during the current press
unsigned long lastHoldReport = 0;
unsigned long shortCount = 0;
unsigned long longCount = 0;
bool simRecording = false;           // simulated app recording state

// --------------------------------------------------------------------------

void debugPrint(const String &msg) {
  if (!debugMode) return;
  Serial.print(F("# "));
  Serial.println(msg);
}

void printHelp() {
  Serial.println(F("# ---- EndoButton debug commands ----"));
  Serial.println(F("#  d = debug ON/OFF   s = status   r = reset counters   h = help"));
  Serial.println(F("#  1 = green LED on   0 = green LED off   p = snapshot blink"));
  Serial.println("#  Short press < " + String(LONG_PRESS_MS) + " ms = SNAPSHOT (LED fast blink)");
  Serial.println("#  Long press >= " + String(LONG_PRESS_MS) + " ms = RECORD START/STOP (LED on/off)");
}

void printStatus() {
  bool raw = (digitalRead(BUTTON_PIN) == LOW);
  Serial.println(F("# ---- EndoButton status ----"));
  Serial.println("#  debug mode      : " + String(debugMode ? "ON" : "OFF"));
  Serial.println("#  pin " + String(BUTTON_PIN) + " now       : " + String(raw ? "LOW (pressed)" : "HIGH (released)"));
  Serial.println("#  debounce / long : " + String(DEBOUNCE_MS) + " ms / " + String(LONG_PRESS_MS) + " ms");
  Serial.println("#  short presses   : " + String(shortCount));
  Serial.println("#  long presses    : " + String(longCount));
  Serial.println("#  green LED (pin " + String(GREEN_LED_PIN) + "): " + String(recordingOn ? "ON (recording)" : "OFF"));
  Serial.println("#  app connected   : " + String(pcSeen ? "yes, last message " + String((millis() - lastPcMessage) / 1000) + " s ago" : "no message from app yet"));
  Serial.println("#  [APP] recording : " + String(simRecording ? "ON" : "OFF"));
  Serial.println("#  uptime          : " + String(millis() / 1000) + " s");
}

// ---- green LED ------------------------------------------------------------

void setRecordingLed(bool on, const char *source) {
  if (on != recordingOn) {
    recordingOn = on;
    debugPrint(String(source) + " green LED " + (on ? "ON  (recording live)" : "OFF (not recording)"));
  }
}

void startSnapshotBlink(const char *source) {
  snapBlinkStart = millis();      // a new snapshot restarts the blink
  snapBlinking = true;
  debugPrint(String(source) + " green LED fast blink (snapshot)");
}

void updateGreenLed(unsigned long now) {
  if (snapBlinking && (now - snapBlinkStart) >= SNAP_BLINK_MS) {
    snapBlinking = false;
  }

  bool ledOn;
  if (snapBlinking) {
    // fast blink; then fall back to the recording state
    ledOn = ((now - snapBlinkStart) / SNAP_BLINK_PERIOD) % 2 == 0;
  } else {
    ledOn = recordingOn;
  }
  digitalWrite(GREEN_LED_PIN, ledOn ? HIGH : LOW);

  // Safety: app stopped talking -> don't keep showing "recording".
  if (!debugMode && recordingOn && pcSeen && (now - lastPcMessage) >= PC_TIMEOUT_MS) {
    recordingOn = false;
  }
}

// ---- built-in LED -----------------------------------------------------------

void updateBuiltinLed(unsigned long now) {
  if (builtinFlickering) {
    unsigned long t = now - builtinFlickerStart;
    if (t >= 240) {
      builtinFlickering = false;
    } else {
      digitalWrite(LED_PIN, (t / 60) % 2 == 0 ? LOW : HIGH);
      return;
    }
  }
  digitalWrite(LED_PIN, stablePressed ? HIGH : LOW);
}

// ---- serial commands ----------------------------------------------------------

void handleSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    switch (c) {
      case '1':                       // app: recording live
        lastPcMessage = millis(); pcSeen = true;
        simRecording = true;
        setRecordingLed(true, "[PC]");
        break;
      case '0':                       // app: recording stopped
        lastPcMessage = millis(); pcSeen = true;
        simRecording = false;
        setRecordingLed(false, "[PC]");
        break;
      case 'p': case 'P':             // app: snapshot saved
        lastPcMessage = millis(); pcSeen = true;
        startSnapshotBlink("[PC]");
        break;
      case '?':                       // used by the app's auto-detect
        lastPcMessage = millis();
        Serial.println(F("ENDO_BTN"));
        break;
      case 'd': case 'D':
        debugMode = !debugMode;
        Serial.println(debugMode ? F("# debug ON  (type h for help)") : F("# debug OFF"));
        break;
      case 's': case 'S':
        printStatus();
        break;
      case 'r': case 'R':
        shortCount = 0; longCount = 0; simRecording = false;
        setRecordingLed(false, "[TEST]");
        Serial.println(F("# counters reset, [APP] recording OFF"));
        break;
      case 'h': case 'H':
        printHelp();
        break;
      default:
        break;                        // ignore newline / carriage return / others
    }
  }
}

// --------------------------------------------------------------------------

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(GREEN_LED_PIN, LOW);

  Serial.begin(9600);
  Serial.println(F("ENDO_BTN"));

  // Wiring check: with INPUT_PULLUP the pin must read HIGH when not pressed.
  if (digitalRead(BUTTON_PIN) == LOW) {
    Serial.println("# WARNING: pin " + String(BUTTON_PIN) +
                   " is LOW at start-up - button held, stuck, or wired to the wrong pins");
  }
  if (debugMode) {
    Serial.println(F("# debug ON"));
    printHelp();
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
    bounceCount++;
  }

  if ((now - lastRawChange) >= DEBOUNCE_MS && rawPressed != stablePressed) {
    stablePressed = rawPressed;

    if (stablePressed) {
      // pressed
      pressStart = now;
      longSent = false;
      lastHoldReport = now;
      debugPrint(F("PRESSED"));
      bounceCount = 0;
    } else {
      // released
      unsigned long held = now - pressStart;
      // the release itself is one edge; anything more is contact bounce
      unsigned int bounces = bounceCount > 0 ? bounceCount - 1 : 0;

      if (!longSent) {
        Serial.println(F("SHORT"));
        shortCount++;
        debugPrint("RELEASED after " + String(held) + " ms, " + String(bounces) +
                   " bounce(s) -> SHORT");
        debugPrint("[APP] snapshot #" + String(shortCount) + " saved   (recording " +
                   String(simRecording ? "ON" : "OFF") + ")");
        // Debug test without the app: blink locally. (With the app connected,
        // the app sends 'p' only after the snapshot is really saved.)
        if (debugMode && !pcSeen) startSnapshotBlink("[TEST]");
      } else {
        debugPrint("RELEASED after " + String(held) + " ms, " + String(bounces) +
                   " bounce(s)  (long press already sent)");
      }
      bounceCount = 0;
    }
  }

  // --- debug: progress while holding ---
  if (debugMode && stablePressed && (now - lastHoldReport) >= HOLD_REPORT_MS) {
    lastHoldReport = now;
    debugPrint("held " + String(now - pressStart) + " ms...");
  }

  // --- long press: fire once while still held ---
  if (stablePressed && !longSent && (now - pressStart) >= LONG_PRESS_MS) {
    longSent = true;
    Serial.println(F("LONG"));
    longCount++;
    simRecording = !simRecording;
    debugPrint("LONG PRESS detected at " + String(now - pressStart) + " ms");
    debugPrint(simRecording ? F("[APP] recording START") : F("[APP] recording STOP (saved)"));
    // Debug test without the app: follow the simulated recording state.
    if (debugMode && !pcSeen) setRecordingLed(simRecording, "[TEST]");
    builtinFlickerStart = now;
    builtinFlickering = true;
  }

  updateGreenLed(now);
  updateBuiltinLed(now);
}
