import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, X, CheckCircle, Disc } from 'lucide-react'; // Using Lucide icons for UI match
import './LiveView.css';

const LiveView = () => {
    const videoRef = React.useRef(null);
    const mediaRecorderRef = React.useRef(null);
    const [recording, setRecording] = React.useState(false);
    const [timer, setTimer] = React.useState(0);
    const location = useLocation();
    const navigate = useNavigate();

    // Get patient from navigation state OR mock if direct access
    // If mocking, we normally wouldn't, but to prevent crash if accessed directly:
    const patient = location.state?.patient || { name: 'Unknown Patient', patientId: '---' };

    React.useEffect(() => {
        let stream = null;

        async function startCamera() {
            try {
                // In production, we would list devices and pick the HDMI capture card.
                // For now, we use default video input.
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 1920, height: 1080 }
                });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (err) {
                console.error("Error accessing camera:", err);
            }
        }
        startCamera();

        return () => {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, []);

    // Timer Logic
    React.useEffect(() => {
        let interval;
        if (recording) {
            interval = setInterval(() => setTimer(prev => prev + 1), 1000);
        } else {
            setTimer(0);
        }
        return () => clearInterval(interval);
    }, [recording]);

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const handleStartRecording = () => {
        if (!videoRef.current?.srcObject) return;
        if (recording) return;

        const stream = videoRef.current.srcObject;
        const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });
        mediaRecorderRef.current = mediaRecorder;

        const chunks = [];
        mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

        mediaRecorder.onstop = async () => {
            // Save logic is handled optionally in stopAndSave or just handled here if triggered
            // We want to differentiate between "Cancel" (discard) and "Save"
            // For now, let's assume if this stops naturally or via Save, we save.
            // But we need a flag to know if we should discard.
            // Implementing a simple ref to track "save" intent could work, 
            // but standard pattern is to have the Stop button trigger the save logic explicitly.
        };

        // We'll attach the logic directly to the buttons instead of onstop for cleaner control
        // But MediaRecorder `ondataavailable` captures the data.

        // Simpler approach: Always capture. If "Cancel" is clicked, we just don't save the blob.

        mediaRecorder.start();
        setRecording(true);
    };

    const handleStopAndSave = async () => {
        if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') {
            // If not recording, maybe just snapshot? Or start recording?
            // Design implies this is "Stop & Save", so it assumes recording is active.
            // If not active, maybe it should be "Start Recording"? 
            // The image shows "Recording" in red top right, implying it IS recording.
            // But if we are in "Pre-record" state, maybe we need a "Start" button?
            // The image shows "Recording" (red box) at the top. This might be a status indicator.
            // The controls are "Stop & Save" and "Cancel". 
            // This implies the user is ALREADY recording or the flow starts immediately?
            // OR, the page IS the "Live Recording" mode. 
            // Let's implement a toggle: If not recording, button says "Start Recording".
            // IF recording, button says "Stop & Save".
            // BUT the prompt says "generate live recording screen like this", showing the ACTIVE recording state.
            // I will implement it such that you can Start, and then you see these buttons.

            // Wait, looking at image: Top right says "Recording". It looks like a status tag.
            // Bottom says "Stop & Save" / "Cancel".
            // This strongly implies the system is currently recording.
            // I will AUTO-START recording when this page loads, or have a "Start" button that shifts to this view.
            // For safety, I'll add a "Start" button if not recording.
            if (!recording) {
                handleStartRecording();
            }
            return;
        }

        mediaRecorderRef.current.stop();
        setRecording(false);

        // We need to wait for the final chunk. Since we can't easily await inside onstop from here without a promise wrapper,
        // We'll do a quick promise wrapper for the data.

        // Actually, simpler: define onstop behaviour to check a 'shouldSave' flag.
        // But for now, let's just use the previous logic but inside the standard flow.

        await new Promise(resolve => {
            mediaRecorderRef.current.onstop = resolve; // Wait for stop event
        });

        // Current implementation of chunks moved to scope? No, ref is better.
        // Let's rewrite the recorder logic slightly to be robust. 
    };

    // Robust Recording Logic
    const chunks = React.useRef([]);

    const startPropRecording = () => {
        if (!videoRef.current?.srcObject) return;
        chunks.current = [];
        const stream = videoRef.current.srcObject;
        const recorder = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });

        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.current.push(e.data);
        };

        recorder.start();
        setRecording(true);
        mediaRecorderRef.current = recorder;
    };

    const stopAndSave = async () => {
        if (!mediaRecorderRef.current) return;

        const recorder = mediaRecorderRef.current;

        return new Promise((resolve) => {
            recorder.onstop = async () => {
                const blob = new Blob(chunks.current, { type: 'video/webm' });
                const buffer = await blob.arrayBuffer();
                const pId = patient.patientId === '---' ? 'ANONYMOUS' : patient.patientId;
                const filename = `REC_${pId}_${Date.now()}.webm`;

                console.log('Saving video...', filename);
                if (window.electronAPI?.saveBuffer) {
                    await window.electronAPI.saveBuffer(buffer, filename);
                    await window.electronAPI.saveMedia({
                        patientId: pId,
                        type: 'video',
                        filePath: filename,
                        timestamp: new Date().toISOString()
                    });
                    alert(`Video Saved: ${filename}`);
                }
                setRecording(false);
                resolve();
            };
            recorder.stop();
        });
    };

    const cancelRecording = () => {
        if (mediaRecorderRef.current && recording) {
            mediaRecorderRef.current.onstop = null; // Remove save handler
            mediaRecorderRef.current.stop();
        }
        setRecording(false);
        chunks.current = []; // Clear data
        navigate(-1); // Go back
    };

    // Auto-start recording on mount? Or manual?
    // User request: "generate live recording screen like this".
    // I will add a "Start" state for usability, but if they click "Start", it looks like the image.

    return (
        <div className="live-recording-container">
            <header className="live-header">
                <div className="header-left">
                    <button className="back-button" onClick={() => navigate(-1)}>
                        <ArrowLeft size={32} />
                    </button>
                    <div className="title-section">
                        <h1>Live Recording</h1>
                        <p className="patient-info">Patient: {patient.name} ({patient.patientId})</p>
                    </div>
                </div>

                <div className={`recording-status ${recording ? 'active' : ''}`}>
                    {recording ? (
                        <>
                            <span className="status-dot"></span>
                            <span>Recording {formatTime(timer)}</span>
                        </>
                    ) : (
                        <span>Ready to Record</span>
                    )}
                </div>
            </header>

            <main className="video-section">
                <video
                    ref={videoRef}
                    autoPlay
                    muted
                    className="video-feed"
                />
                {!videoRef.current?.srcObject && <div className="placeholder-text">Initializing Camera...</div>}
            </main>

            <footer className="controls-section">
                {!recording ? (
                    <button className="control-btn btn-save" onClick={startPropRecording} style={{ background: '#2563eb' }}>
                        <Disc size={20} />
                        Start Recording
                    </button>
                ) : (
                    <>
                        <button className="control-btn btn-save" onClick={stopAndSave}>
                            <CheckCircle size={20} />
                            Stop & Save
                        </button>
                        <button className="control-btn btn-cancel" onClick={cancelRecording}>
                            <X size={20} />
                            Cancel
                        </button>
                    </>
                )}
            </footer>
        </div>
    );
};

export default LiveView;
