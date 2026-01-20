import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Plus, PenSquare, Trash2, X } from 'lucide-react';
import './Patients.css';

const Patients = () => {
    const [patients, setPatients] = React.useState([]);
    const [showModal, setShowModal] = React.useState(false);
    const [isEditing, setIsEditing] = React.useState(false);
    const [formData, setFormData] = React.useState({
        patientId: '', name: '', age: '', gender: 'Male', phone: ''
    });

    const navigate = useNavigate();

    const fetchPatients = async () => {
        if (window.electronAPI) {
            const data = await window.electronAPI.getPatients();
            setPatients(data);
        }
    };

    React.useEffect(() => {
        fetchPatients();
    }, []);

    const handleOpenAdd = () => {
        setIsEditing(false);
        const newId = 'P-' + Math.floor(1000 + Math.random() * 9000);
        setFormData({
            patientId: newId,
            name: '', age: '', gender: 'Male', phone: '',
            dateAdded: new Date().toISOString().slice(0, 10)
        });
        setShowModal(true);
    };

    const handleOpenEdit = (patient) => {
        setIsEditing(true);
        setFormData(patient);
        setShowModal(true);
    };

    const handleDelete = async (patientId) => {
        if (!confirm('Are you sure you want to delete this patient?')) return;

        if (window.electronAPI) {
            const result = await window.electronAPI.deletePatient(patientId);
            if (result.success) {
                fetchPatients();
            } else {
                alert('Error deleting: ' + result.error);
            }
        } else {
            setPatients(patients.filter(p => p.patientId !== patientId));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (window.electronAPI) {
            let result;
            if (isEditing) {
                result = await window.electronAPI.updatePatient(formData);
            } else {
                result = await window.electronAPI.addPatient(formData);
            }

            if (result.success) {
                setShowModal(false);
                fetchPatients();
            } else {
                alert('Error processing: ' + result.error);
            }
        } else {
            // Mock
            if (isEditing) {
                setPatients(patients.map(p => p.patientId === formData.patientId ? formData : p));
            } else {
                setPatients([...patients, { ...formData, id: Date.now() }]);
            }
            setShowModal(false);
        }
    };

    return (
        <div className="patients-container">
            <header className="page-header">
                <div className="header-left">
                    <button className="back-btn" onClick={() => navigate('/home')}>
                        <ArrowLeft size={28} />
                    </button>
                    <h1>Patient Details</h1>
                </div>
                <button
                    className="add-btn"
                    onClick={handleOpenAdd}
                >
                    <Plus size={20} />
                    Add Patients
                </button>
            </header>

            <div className="table-container">
                <table className="patients-table">
                    <thead>
                        <tr>
                            <th>Patient ID</th>
                            <th>Name</th>
                            <th>Age</th>
                            <th>Gender</th>
                            <th>Phone</th>
                            <th>Date Added</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {patients.map(p => (
                            <tr key={p.id || p.patientId}>
                                <td>{p.patientId}</td>
                                <td>{p.name}</td>
                                <td>{p.age}</td>
                                <td>{p.gender}</td>
                                <td>{p.phone}</td>
                                <td>{p.dateAdded}</td>
                                <td>
                                    <div className="action-icons">
                                        <button
                                            className="icon-btn edit"
                                            onClick={() => handleOpenEdit(p)}
                                            title="Edit"
                                        >
                                            <PenSquare size={18} />
                                        </button>
                                        <button
                                            className="icon-btn delete"
                                            onClick={() => handleDelete(p.patientId)}
                                            title="Delete"
                                        >
                                            <Trash2 size={18} />
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                        {patients.length === 0 && (
                            <tr>
                                <td colSpan="7" style={{ textAlign: 'center', padding: '30px', background: 'transparent' }}>
                                    No patients found. Click "Add Patients" to begin.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {showModal && (
                <div className="modal-overlay">
                    <div className="form-container">
                        <div className="form-header">
                            <h2 style={{ margin: 0 }}>{isEditing ? 'Edit Patient' : 'Add New Patient'}</h2>
                            <button onClick={() => setShowModal(false)} style={{ background: 'none', border: 'none', cursor: 'pointer' }}>
                                <X size={24} color="#6b7280" />
                            </button>
                        </div>
                        <form onSubmit={handleSubmit} className="form-grid">
                            <div style={{ gridColumn: 'span 2' }}>
                                <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9rem', color: '#6b7280' }}>Full Name</label>
                                <input
                                    className="form-input"
                                    required
                                    value={formData.name}
                                    onChange={e => setFormData({ ...formData, name: e.target.value })}
                                />
                            </div>

                            <div>
                                <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9rem', color: '#6b7280' }}>Age</label>
                                <input
                                    className="form-input"
                                    required
                                    type="number"
                                    value={formData.age}
                                    onChange={e => setFormData({ ...formData, age: e.target.value })}
                                />
                            </div>

                            <div>
                                <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9rem', color: '#6b7280' }}>Gender</label>
                                <select
                                    className="form-select"
                                    value={formData.gender}
                                    onChange={e => setFormData({ ...formData, gender: e.target.value })}
                                >
                                    <option>Male</option>
                                    <option>Female</option>
                                    <option>Other</option>
                                </select>
                            </div>

                            <div style={{ gridColumn: 'span 2' }}>
                                <label style={{ display: 'block', marginBottom: '5px', fontSize: '0.9rem', color: '#6b7280' }}>Phone Number</label>
                                <input
                                    className="form-input"
                                    value={formData.phone}
                                    onChange={e => setFormData({ ...formData, phone: e.target.value })}
                                />
                            </div>

                            <div className="form-actions">
                                <button type="button" onClick={() => setShowModal(false)} className="cancel-btn">Cancel</button>
                                <button type="submit" className="save-btn">{isEditing ? 'Update' : 'Save'}</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Patients;
