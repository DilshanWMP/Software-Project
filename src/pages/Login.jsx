import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, Eye, EyeOff } from 'lucide-react';
import './Login.css';

const Login = () => {
    const navigate = useNavigate();
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [error, setError] = useState('');

    const handleLogin = (e) => {
        e.preventDefault();
        if (password === 'USERSTAFF') {
            navigate('/home');
        } else {
            setError('Invalid Password. Access Denied.');
        }
    };

    return (
        <div className="login-container">
            <div className="login-card">
                <div className="icon-circle">
                    <Lock size={28} />
                </div>

                <h1 className="login-title">Endoscope Recording System</h1>
                <p className="login-subtitle">Staff Access Only</p>

                <form onSubmit={handleLogin}>
                    <div className="input-group">
                        <label className="input-label">Password</label>
                        <div className="password-input-wrapper">
                            <input
                                type={showPassword ? "text" : "password"}
                                placeholder="Enter password"
                                className="login-input"
                                value={password}
                                onChange={(e) => {
                                    setPassword(e.target.value);
                                    setError('');
                                }}
                            />
                            <button
                                type="button"
                                className="toggle-password"
                                onClick={() => setShowPassword(!showPassword)}
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                        {error && <p style={{ color: '#ef4444', fontSize: '0.875rem', marginTop: '5px', textAlign: 'left' }}>{error}</p>}
                    </div>

                    <button type="submit" className="btn-login">
                        Login
                    </button>

                    <a href="#" className="forgot-password">Change Password</a>
                </form>
            </div>
        </div>
    );
};

export default Login;
