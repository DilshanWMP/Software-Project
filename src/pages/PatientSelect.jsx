import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, User } from 'lucide-react';
import './PatientSelect.css';

const PatientSelect = () => {
    const [patients, setPatients] = React.useState([]);
    const navigate = useNavigate();

    React.useEffect(() => {
        const loadPatients = async () => {
            if (window.electronAPI) {
                const data = await window.electronAPI.getPatients();
                setPatients(data);
            }
        };
        loadPatients();
    }, []);

    const handleSelect = (patient) => {
        navigate('/live', { state: { patient } });
    };

    return (
        <div className="select-container">
            <header className="select-header">
                <button className="back-btn" onClick={() => navigate('/home')}>
                    <ArrowLeft size={28} />
                </button>
                <h1>Live Video</h1>
            </header>

            <h2 className="section-title">Select a Patient</h2>

            <div className="patient-grid">
                {patients.map(p => (
                    <div key={p.id || p.patientId} className="patient-card" onClick={() => handleSelect(p)}>
                        <div className="card-avatar">
                            <User size={24} />
                        </div>
                        <div className="card-info">
                            <h3 className="card-name">{p.name}</h3>
                            <span className="card-id">ID: {p.patientId}</span>
                            <div className="card-meta">
                                <span>{p.age} years</span>
                                <span>{p.gender}</span>
                            </div>
                            <span className="card-phone">{p.phone}</span>
                        </div>
                    </div>
                ))}
            </div>
            {patients.length === 0 && <p style={{ color: '#6b7280', marginLeft: '10px' }}>No patients found. Please add patients first.</p>}
        </div>
    );
};

export default PatientSelect;
