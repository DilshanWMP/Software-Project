# EndoButton – Arduino hardware push button + recording LED

| Action | Result in the app (Record screen) |
|---|---|
| Short press (< 0.8 s) | Snapshot – works while recording or not |
| Long press (≥ 0.8 s) | Start recording / stop recording |

| Green LED | Meaning |
|---|---|
| Solid ON | Recording is live |
| OFF | Not recording |
| Fast blink (~1 s) | Snapshot saved. Then it goes back to ON or OFF, so it works the same during a recording. |

The app controls the LED, so it shows the **real** state. For example, a long press with no patient selected won't turn it on. If the app closes or crashes, the LED turns off within 5 s.

## Wiring (Arduino Uno)

```
Push button ── pin 2
Push button ── GND

pin 8 ── 220 Ω resistor ── green LED long leg (+)
                           green LED short leg (−) ── GND
```
The button needs no resistor (the internal pull-up is used). The LED **does** need the 220 Ω resistor (anything from 150 to 330 Ω is fine). The on-board LED (pin 13) lights while the button is held and flickers when a long press is registered.

## Setup

1. Upload `EndoButton.ino` with the Arduino IDE.
2. **Close the Arduino IDE Serial Monitor/Plotter.** Only one program can use a COM port at a time.
3. In the app go to **Settings → Hardware Push Button**, choose **Auto-detect** or the Uno's COM port, click **Connect / Test**, press the button, then **Save Changes**.

## Testing the button without the app (debug mode)

1. **Close the Endoscopy App.** It holds the COM port while it's running.
2. Arduino IDE → **Tools → Serial Monitor**, set the baud rate to **9600**.
3. Type `d` and press Enter to turn debug on. Other commands: `s` = status, `r` = reset counters, `h` = help, `1` / `0` / `p` = green LED on / off / snapshot blink.
   In debug mode without the app, the LED follows the button: long press turns it on or off, short press makes it blink.
4. Press the button:

```
# PRESSED
# RELEASED after 214 ms, 0 bounce(s) -> SHORT
# [APP] snapshot #1 saved   (recording OFF)
# PRESSED
# LONG PRESS detected at 800 ms
# [APP] recording START
# RELEASED after 1130 ms, 0 bounce(s)  (long press already sent)
# PRESSED
# RELEASED after 180 ms, 0 bounce(s) -> SHORT
# [APP] snapshot #2 saved   (recording ON)
```

`[APP]` lines show what the app would do. Lots of bounces, or one press counted as two, means the button is noisy: raise `DEBOUNCE_MS`. If you see `WARNING: pin 2 is LOW at start-up`, the button is wired to the wrong pins or stuck. Debug lines start with `#` and the app ignores them.

## Using your own sketch

The app only needs these lines at **9600 baud**, each ending with a newline (`Serial.println`):

| Message | Meaning |
|---|---|
| `SHORT` | short press |
| `LONG` | long press |
| `ENDO_BTN` | identification. Send it at boot and reply with it when the PC sends `?`. Only needed for **Auto-detect**. If you pick the COM port by hand, `SHORT`/`LONG` alone is enough. |

The app sends single characters back: `1` = recording live, `0` = recording stopped (both repeated every 2 s as a heartbeat), and `p` = snapshot saved.

Do the debouncing and short/long timing on the Arduino.

## Troubleshooting

| Symptom (status line in Settings) | Fix |
|---|---|
| `access denied … used by another program` | Close the Arduino IDE Serial Monitor (or any other serial tool). |
| `COMx not found` | Cable unplugged, or the port number changed. Plug it in, click **Refresh**, pick the port again. Check Device Manager → *Ports (COM & LPT)*. Clone boards with a CH340 chip need the CH340 driver. |
| `No button device found` (Auto-detect) | The sketch doesn't send `ENDO_BTN`. Select the COM port manually instead. |
| `open - waiting for the button device…` stays | Wrong port (e.g. a Bluetooth port), wrong baud rate, or the sketch isn't running. Pressing the button should switch it to *Connected*. |
| Status is *Connected*, *Last message* updates, but nothing happens | Presses only act on the **Record** screen, and only once a patient is selected and the camera is running. A message appears at the bottom of the video if something is missing. |
| Green LED never lights | Check the LED direction (long leg to the resistor/pin 8) and the resistor. Test it with `1` in the Serial Monitor. |
| Green LED goes off by itself during recording | The app stopped sending heartbeats: the app was closed, or the USB connection dropped. Check the status in Settings. |
| Short press triggers twice / long press triggers short too | Increase `DEBOUNCE_MS`, or adjust `LONG_PRESS_MS` in the sketch. |

If the cable is unplugged, the app reconnects on its own when it's plugged back in.
