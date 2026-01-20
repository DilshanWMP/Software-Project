import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Settings, LogOut, User, Video, Folder, Globe } from 'lucide-react';
import './Home.css';

const Home = () => {
    const navigate = useNavigate();

    return (
        <div className="home-container">
            <header className="home-header">
                <div className="logo-section">
                    <div className="logo-placeholder">
                        <Globe size={40} strokeWidth={1.5} />
                        <span>BMI</span>
                    </div>
                </div>
                <div className="header-actions">
                    <button className="action-btn btn-settings">
                        <Settings size={20} />
                        Settings
                    </button>
                    <button className="action-btn btn-logout" onClick={() => navigate('/')}>
                        <LogOut size={20} />
                        Logout
                    </button>
                </div>
            </header>

            <main className="cards-container">
                <div className="home-card card-patient" onClick={() => navigate('/patients')}>
                    <div className="icon-wrapper">
                        <User size={40} color="white" />
                    </div>
                    <h2>Patient Details</h2>
                    <p>Add and manage patient information</p>
                </div>

                <div className="home-card card-live" onClick={() => navigate('/live/select')}>
                    <div className="icon-wrapper">
                        <Video size={40} color="white" />
                    </div>
                    <h2>Live</h2>
                    <p>Record live endoscopy</p>
                </div>

                <div className="home-card card-videos" onClick={() => navigate('/videos')}>
                    <div className="icon-wrapper">
                        <Folder size={40} color="white" />
                    </div>
                    <h2>Recorded Videos</h2>
                    <p>View and manage recorded procedures</p>
                </div>
            </main>
        </div>
    );
};

export default Home;
