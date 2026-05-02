using System;
using System.IO.Ports;
using System.Threading;
using System.Diagnostics;

namespace EndoscopyApp.Services
{
    public class FootPedalService : IDisposable
    {
        private SerialPort? _serialPort;
        private Timer? _reconnectTimer;
        private readonly string _comPort;

        public event Action? PedalPressed;

        public bool IsConnected => _serialPort != null && _serialPort.IsOpen;

        public FootPedalService(string comPort)
        {
            _comPort = comPort;
            
            if (!string.IsNullOrWhiteSpace(_comPort) && _comPort != "None")
            {
                Connect();
                
                // Set up a timer to check connection health every 3 seconds
                _reconnectTimer = new Timer(CheckConnection, null, 3000, 3000);
            }
        }

        private void Connect()
        {
            try
            {
                if (_serialPort != null && _serialPort.IsOpen)
                {
                    return;
                }

                _serialPort = new SerialPort(_comPort, 115200);
                _serialPort.DataReceived += DataReceivedHandler;
                _serialPort.Open();
                Debug.WriteLine($"Successfully connected to foot pedal on {_comPort}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Could not connect to foot pedal on {_comPort}: {ex.Message}");
                // Disconnection or failure is handled silently to avoid interrupting the doctor
            }
        }

        private void CheckConnection(object? state)
        {
            if (_serialPort == null) return;

            try
            {
                if (!_serialPort.IsOpen)
                {
                    Debug.WriteLine("Foot pedal disconnected. Attempting to reconnect...");
                    Connect();
                }
            }
            catch
            {
                // Ignore exceptions during status check
            }
        }

        private void DataReceivedHandler(object sender, SerialDataReceivedEventArgs e)
        {
            try
            {
                if (_serialPort == null || !_serialPort.IsOpen) return;

                string indata = _serialPort.ReadLine();
                if (indata.Contains("SNAPSHOT"))
                {
                    // Trigger the event safely
                    PedalPressed?.Invoke();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error reading from pedal: {ex.Message}");
            }
        }

        public void Dispose()
        {
            _reconnectTimer?.Dispose();
            if (_serialPort != null)
            {
                if (_serialPort.IsOpen)
                {
                    _serialPort.DataReceived -= DataReceivedHandler;
                    _serialPort.Close();
                }
                _serialPort.Dispose();
            }
        }
    }
}
