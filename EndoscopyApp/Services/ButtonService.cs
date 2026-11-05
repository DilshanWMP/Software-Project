using System;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace EndoscopyApp.Services
{
    public enum ButtonConnectionState
    {
        Disabled,
        Searching,
        Connected,
        Error
    }

    /// <summary>
    /// Talks to the Arduino hardware push button over a USB serial (COM) port.
    ///
    /// Protocol (9600 baud, one message per line, '\n' terminated):
    ///   Arduino -> PC : "SHORT"     short press  -> take snapshot
    ///   Arduino -> PC : "LONG"      long press   -> start / stop recording
    ///   Arduino -> PC : "ENDO_BTN"  identification (sent at boot and as a reply to "?")
    ///   PC -> Arduino : "?"         "who are you?" (only used by auto-detect)
    ///   PC -> Arduino : "1" / "0"   recording is live / stopped -> green LED on / off.
    ///                               Re-sent every 2 s as a heartbeat; the Arduino turns
    ///                               the LED off if it hears nothing for 5 s (app closed/crashed).
    ///   PC -> Arduino : "p"         snapshot saved -> green LED blinks fast
    ///
    /// Short / long press timing and debouncing are done on the Arduino, so the
    /// app only reacts to finished events.
    ///
    /// This is a single app-wide instance so that the port stays open while the
    /// user navigates between screens.
    /// </summary>
    public sealed class ButtonService : IDisposable
    {
        public const string DisabledSetting = "";
        public const string AutoDetectSetting = "Auto";
        public const int BaudRate = 9600;
        private const string IdentMessage = "ENDO_BTN";

        public static ButtonService Instance { get; } = new ButtonService();

        /// <summary>Raised on the UI thread.</summary>
        public event Action? ShortPressed;
        /// <summary>Raised on the UI thread.</summary>
        public event Action? LongPressed;
        /// <summary>Raised on the UI thread whenever State / StatusMessage / LastMessage change.</summary>
        public event Action? StatusChanged;

        public ButtonConnectionState State { get; private set; } = ButtonConnectionState.Disabled;
        public string StatusMessage { get; private set; } = "Hardware button disabled";
        public string? ConnectedPort { get; private set; }
        /// <summary>Last line received from the device (for troubleshooting in Settings).</summary>
        public string LastMessage { get; private set; } = "";
        /// <summary>The setting passed to the last Configure() call.</summary>
        public string CurrentSetting { get; private set; } = DisabledSetting;

        private readonly object _sync = new();
        private CancellationTokenSource? _cts;
        private Task? _worker;
        private volatile SerialPort? _port;
        // Incremented on every Configure(); a worker from an older configuration
        // is not allowed to publish status any more.
        private int _generation;

        // Recording-LED state the Arduino should show, and a lock for writes from any thread.
        private volatile bool _recordingLed;
        private readonly object _writeLock = new();
        private static readonly TimeSpan HeartbeatInterval = TimeSpan.FromSeconds(2);

        private ButtonService() { }

        /// <summary>Tell the Arduino whether recording is live (green LED on/off).</summary>
        public void SetRecording(bool isRecording)
        {
            _recordingLed = isRecording;
            SendInBackground(RecordingCommand);
        }

        /// <summary>Tell the Arduino a snapshot was saved (green LED fast blink).</summary>
        public void NotifySnapshot() => SendInBackground("p");

        private string RecordingCommand => _recordingLed ? "1" : "0";

        private void SendInBackground(string message)
        {
            var port = _port;
            if (port == null || State != ButtonConnectionState.Connected) return;
            // Don't block the UI thread on a slow / stuck port.
            Task.Run(() => TryWrite(port, message));
        }

        private bool TryWrite(SerialPort port, string message)
        {
            lock (_writeLock)
            {
                try
                {
                    if (!port.IsOpen) return false;
                    port.Write(message + "\n");
                    return true;
                }
                catch
                {
                    return false; // read loop will notice a dead port and reconnect
                }
            }
        }

        public static string[] GetAvailablePorts()
        {
            try
            {
                return SerialPort.GetPortNames()
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(p => p.Length).ThenBy(p => p, StringComparer.OrdinalIgnoreCase)
                    .ToArray();
            }
            catch
            {
                return Array.Empty<string>();
            }
        }

        /// <summary>
        /// Returns true if the port is a Bluetooth serial link (e.g. "Standard Serial over
        /// Bluetooth link"). Opening those is slow and may try to connect to a paired phone,
        /// so auto-detect skips them. Uses HKLM\HARDWARE\DEVICEMAP\SERIALCOMM.
        /// </summary>
        public static bool IsBluetoothPort(string portName)
        {
            try
            {
                using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DEVICEMAP\SERIALCOMM");
                if (key == null) return false;
                foreach (var valueName in key.GetValueNames())
                {
                    if (string.Equals(key.GetValue(valueName) as string, portName, StringComparison.OrdinalIgnoreCase))
                        return valueName.Contains("Bth", StringComparison.OrdinalIgnoreCase);
                }
            }
            catch { /* ignore */ }
            return false;
        }

        /// <summary>
        /// (Re)starts the connection using the given setting:
        /// "" = disabled, "Auto" = auto-detect, otherwise a port name such as "COM3".
        /// </summary>
        public void Configure(string? portSetting)
        {
            portSetting = (portSetting ?? DisabledSetting).Trim();

            lock (_sync)
            {
                StopWorker();
                int gen = Interlocked.Increment(ref _generation);
                CurrentSetting = portSetting;

                if (string.IsNullOrEmpty(portSetting))
                {
                    SetStatus(gen, ButtonConnectionState.Disabled, "Hardware button disabled", null);
                    return;
                }

                _cts = new CancellationTokenSource();
                var token = _cts.Token;
                _worker = Task.Factory.StartNew(() => Run(portSetting, gen, token), token,
                    TaskCreationOptions.LongRunning, TaskScheduler.Default);
            }
        }

        // ------------------------------------------------------------------
        // Worker thread
        // ------------------------------------------------------------------

        private void Run(string portSetting, int gen, CancellationToken token)
        {
            bool auto = string.Equals(portSetting, AutoDetectSetting, StringComparison.OrdinalIgnoreCase);

            while (!token.IsCancellationRequested)
            {
                SerialPort? port = null;
                try
                {
                    if (auto)
                    {
                        SetStatus(gen, ButtonConnectionState.Searching, "Searching for button device...", null);
                        port = DetectDevice(token);
                        if (port == null)
                        {
                            SetStatus(gen, ButtonConnectionState.Searching,
                                "No button device found. Check the USB cable, or select the COM port manually.", null);
                            Wait(token, 3000);
                            continue;
                        }
                    }
                    else
                    {
                        if (!GetAvailablePorts().Contains(portSetting, StringComparer.OrdinalIgnoreCase))
                        {
                            SetStatus(gen, ButtonConnectionState.Searching,
                                $"{portSetting} not found. Waiting for the device to be plugged in...", null);
                            Wait(token, 2000);
                            continue;
                        }

                        SetStatus(gen, ButtonConnectionState.Searching, $"Opening {portSetting}...", null);
                        port = OpenPort(portSetting);
                    }

                    _port = port;
                    if (auto)
                    {
                        SetStatus(gen, ButtonConnectionState.Connected, $"Connected to button on {port.PortName}", port.PortName);
                        TryWrite(port, RecordingCommand);
                    }
                    else
                        SetStatus(gen, ButtonConnectionState.Searching,
                            $"{port.PortName} open - waiting for the button device to respond (press the button to test)...", null);

                    ReadLoop(port, gen, verified: auto, token);
                }
                catch (UnauthorizedAccessException)
                {
                    SetStatus(gen, ButtonConnectionState.Error,
                        $"{portSetting}: access denied. The port is used by another program " +
                        "(close the Arduino IDE Serial Monitor / Plotter).", null);
                }
                catch (Exception ex) when (!token.IsCancellationRequested)
                {
                    SetStatus(gen, ButtonConnectionState.Error,
                        $"Connection lost ({ex.Message}). Reconnecting...", null);
                }
                catch
                {
                    // cancelled - fall through
                }
                finally
                {
                    if (ReferenceEquals(_port, port)) _port = null;
                    SafeClose(port);
                }

                Wait(token, 2000);
            }
        }

        private void ReadLoop(SerialPort port, int gen, bool verified, CancellationToken token)
        {
            string portName = port.PortName;
            var opened = DateTime.UtcNow;
            var lastPresenceCheck = DateTime.UtcNow;
            bool asked = false;
            var lastHeartbeat = DateTime.UtcNow;

            while (!token.IsCancellationRequested)
            {
                // Heartbeat: keep the Arduino's recording LED in sync (and alive).
                if (verified && DateTime.UtcNow - lastHeartbeat >= HeartbeatInterval)
                {
                    lastHeartbeat = DateTime.UtcNow;
                    TryWrite(port, RecordingCommand);
                }

                string? line = null;
                try
                {
                    line = port.ReadLine();
                }
                catch (TimeoutException)
                {
                    // An Uno resets when the port opens and prints ENDO_BTN on boot. If we
                    // missed it (or the board doesn't reset), ask once after boot time.
                    if (!verified && !asked && (DateTime.UtcNow - opened).TotalMilliseconds > 2500)
                    {
                        asked = true;
                        TryWrite(port, "?");
                    }

                    // No data - make sure the device is still plugged in.
                    // (Some USB-serial drivers never raise an error when unplugged.)
                    if ((DateTime.UtcNow - lastPresenceCheck).TotalSeconds >= 2)
                    {
                        lastPresenceCheck = DateTime.UtcNow;
                        if (!GetAvailablePorts().Contains(portName, StringComparer.OrdinalIgnoreCase))
                            throw new IOException($"{portName} was unplugged");
                    }
                    continue;
                }

                if (!verified && IsDeviceMessage(line))
                {
                    verified = true;
                    SetStatus(gen, ButtonConnectionState.Connected, $"Connected to button on {portName}", portName);
                    TryWrite(port, RecordingCommand);
                    lastHeartbeat = DateTime.UtcNow;
                }
                else if (verified && string.Equals(line?.Trim(), IdentMessage, StringComparison.OrdinalIgnoreCase))
                {
                    // ENDO_BTN again = the board was reset and lost its LED state.
                    TryWrite(port, RecordingCommand);
                    lastHeartbeat = DateTime.UtcNow;
                }

                HandleLine(line, gen);
            }
        }

        private static bool IsDeviceMessage(string? raw)
        {
            string line = (raw ?? "").Trim().ToUpperInvariant();
            return line == IdentMessage || line == "SHORT" || line == "LONG";
        }

        private void HandleLine(string? raw, int gen)
        {
            if (gen != Volatile.Read(ref _generation)) return;
            string line = (raw ?? "").Trim().ToUpperInvariant();
            if (line.Length == 0) return;
            if (line.StartsWith('#')) return; // Arduino debug output (see EndoButton.ino)

            LastMessage = $"{line}  ({DateTime.Now:HH:mm:ss})";

            Action? handler = line switch
            {
                "SHORT" => ShortPressed,
                "LONG" => LongPressed,
                _ => null // ENDO_BTN or anything unknown - only shown in Settings
            };

            RunOnUi(() =>
            {
                StatusChanged?.Invoke();
                handler?.Invoke();
            });
        }

        /// <summary>
        /// Tries every COM port and returns the first one that identifies itself
        /// as the endoscopy button. Returns null if none found.
        /// </summary>
        private SerialPort? DetectDevice(CancellationToken token)
        {
            foreach (var name in GetAvailablePorts().Where(p => !IsBluetoothPort(p)))
            {
                if (token.IsCancellationRequested) return null;

                SerialPort? port = null;
                try
                {
                    port = OpenPort(name);

                    // Opening the port resets an Arduino Uno; the bootloader takes ~1.5-2 s.
                    // The sketch prints ENDO_BTN on boot. If the board did not reset
                    // (e.g. other boards) we also ask it with "?".
                    var deadline = DateTime.UtcNow.AddMilliseconds(3500);
                    bool asked = false;
                    while (DateTime.UtcNow < deadline && !token.IsCancellationRequested)
                    {
                        if (!asked && DateTime.UtcNow > deadline.AddMilliseconds(-1500))
                        {
                            try { port.WriteLine("?"); } catch { /* ignore */ }
                            asked = true;
                        }

                        try
                        {
                            string reply = port.ReadLine().Trim().ToUpperInvariant();
                            if (reply == IdentMessage)
                                return port; // keep it open
                        }
                        catch (TimeoutException) { }
                    }
                }
                catch
                {
                    // busy / not a serial device we can open - try the next one
                }

                SafeClose(port);
            }
            return null;
        }

        private static SerialPort OpenPort(string name)
        {
            var port = new SerialPort(name, BaudRate, Parity.None, 8, StopBits.One)
            {
                NewLine = "\n",
                ReadTimeout = 500,
                WriteTimeout = 500,
                DtrEnable = true,   // required by many boards (Leonardo/Micro) to send data
                RtsEnable = true,
                Handshake = Handshake.None
            };
            port.Open();
            port.DiscardInBuffer();
            return port;
        }

        // ------------------------------------------------------------------
        // Helpers
        // ------------------------------------------------------------------

        private void StopWorker()
        {
            _cts?.Cancel();
            SafeClose(_port);     // unblocks a pending ReadLine immediately
            _port = null;
            try { _worker?.Wait(1500); } catch { /* ignore */ }
            _cts?.Dispose();
            _cts = null;
            _worker = null;
        }

        private static void SafeClose(SerialPort? port)
        {
            if (port == null) return;
            try { if (port.IsOpen) port.Close(); } catch { }
            try { port.Dispose(); } catch { }
        }

        private static void Wait(CancellationToken token, int ms) => token.WaitHandle.WaitOne(ms);

        private void SetStatus(int gen, ButtonConnectionState state, string message, string? port)
        {
            if (gen != Volatile.Read(ref _generation)) return; // stale worker
            State = state;
            StatusMessage = message;
            ConnectedPort = port;
            RunOnUi(() => StatusChanged?.Invoke());
        }

        private static void RunOnUi(Action action)
        {
            var dispatcher = Application.Current?.Dispatcher;
            if (dispatcher == null) return;
            if (dispatcher.CheckAccess()) action();
            else dispatcher.BeginInvoke(action);
        }

        public void Dispose()
        {
            lock (_sync)
            {
                // App is closing: turn the recording LED off before releasing the port.
                _recordingLed = false;
                var port = _port;
                if (port != null) TryWrite(port, "0");
                StopWorker();
            }
        }
    }
}
