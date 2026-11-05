# EndoButton – Arduino hardware push button

| Action | Result in the app (Record screen) |
|---|---|
| Short press (< 0.8 s) | Snapshot – works while recording or not |
| Long press (≥ 0.8 s) | Start recording / stop recording |

## Wiring (Arduino Uno)

```
Push button ── pin 2
Push button ── GND
```
No resistor is needed (the internal pull-up is used). The on-board LED (pin 13) lights while the button is held and blinks twice when a long press is registered.

## Setup

1. Upload `EndoButton.ino` with the Arduino IDE.
2. **Close the Arduino IDE Serial Monitor/Plotter.** Only one program can use a COM port at a time.
3. In the app go to **Settings → Hardware Push Button**, choose **Auto-detect** or the Uno's COM port, click **Connect / Test**, press the button, then **Save Changes**.

## Using your own sketch

The app only needs these lines at **9600 baud**, each ending with a newline (`Serial.println`):

| Message | Meaning |
|---|---|
| `SHORT` | short press |
| `LONG` | long press |
| `ENDO_BTN` | identification. Send it at boot and reply with it when the PC sends `?`. Only needed for **Auto-detect**. If you pick the COM port by hand, `SHORT`/`LONG` alone is enough. |

Do the debouncing and short/long timing on the Arduino.

## Troubleshooting

| Symptom (status line in Settings) | Fix |
|---|---|
| `access denied … used by another program` | Close the Arduino IDE Serial Monitor (or any other serial tool). |
| `COMx not found` | Cable unplugged, or the port number changed. Plug it in, click **Refresh**, pick the port again. Check Device Manager → *Ports (COM & LPT)*. Clone boards with a CH340 chip need the CH340 driver. |
| `No button device found` (Auto-detect) | The sketch doesn't send `ENDO_BTN`. Select the COM port manually instead. |
| `open - waiting for the button device…` stays | Wrong port (e.g. a Bluetooth port), wrong baud rate, or the sketch isn't running. Pressing the button should switch it to *Connected*. |
| Status is *Connected*, *Last message* updates, but nothing happens | Presses only act on the **Record** screen, and only once a patient is selected and the camera is running. A message appears at the bottom of the video if something is missing. |
| Short press triggers twice / long press triggers short too | Increase `DEBOUNCE_MS`, or adjust `LONG_PRESS_MS` in the sketch. |

If the cable is unplugged, the app reconnects on its own when it's plugged back in.
