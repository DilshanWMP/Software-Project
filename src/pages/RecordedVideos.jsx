import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Film, Image as ImageIcon } from 'lucide-react';
import './RecordedVideos.css';

const RecordedVideos = () => {
    const [mediaList, setMediaList] = React.useState([]);
    const navigate = useNavigate();

    React.useEffect(() => {
        const loadMedia = async () => {
            if (window.electronAPI) {
                try {
                    const data = await window.electronAPI.getAllMedia();
                    setMediaList(data);
                } catch (err) {
                    console.error("Failed to load media", err);
                }
            }
        };
        loadMedia();
    }, []);

    const handleOpenMedia = (filePath) => {
        // In a real app, we'd open a modal or use the system player.
        // Since we are in Electron, we can't easily validly src="file://..." due to security without handling protocols.
        // For now, let's just alert the path, or if we had a protocol set up we'd use it.
        // We will just show the info.
        alert(`Opening ${filePath} (Player Implementation Needed)`);
    };

    return (
        <div className="videos-container">
            <header className="videos-header">
                <button className="back-btn" onClick={() => navigate('/home')}>
                    <ArrowLeft size={28} />
                </button>
                <h1 style={{ margin: 0 }}>Recorded Library</h1>
            </header>

            <div className="videos-grid">
                {mediaList.map((media) => (
                    <div key={media.id} className="video-card" onClick={() => handleOpenMedia(media.filePath)}>
                        <div className="video-thumbnail">
                            {/* In production, generate actual thumbnails. For now, icon */}
                            {media.type === 'video' ? <Film size={40} /> : <ImageIcon size={40} />}
                        </div>
                        <div className="video-info">
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                                <span className="type-badge">{media.type}</span>
                                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{new Date(media.timestamp).toLocaleDateString()}</span>
                            </div>
                            <h3>{media.patientName || 'Anonymous'}</h3>
                            <div className="video-meta">
                                <span>ID: {media.patientId}</span>
                            </div>
                            <p style={{ fontSize: '0.75rem', color: '#cbd5e1', marginTop: '8px', wordBreak: 'break-all' }}>
                                {media.filePath}
                            </p>
                        </div>
                    </div>
                ))}
            </div>

            {mediaList.length === 0 && (
                <div style={{ textAlign: 'center', marginTop: '100px', color: '#94a3b8' }}>
                    <p>No recordings found.</p>
                </div>
            )}
        </div>
    );
};

export default RecordedVideos;
