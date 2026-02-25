import React from 'react';
import { useNavigate } from 'react-router-dom';
import { LayoutDashboard, Users, Video, LogOut } from 'lucide-react';

const DashboardLayout = ({ children }) => {
    const navigate = useNavigate();
    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', path: '/dashboard' },
        { icon: Video, label: 'Live View', path: '/live' },
        { icon: Users, label: 'Patients', path: '/patients' },
    ];

    return (
        <div style={{ display: 'flex', height: '100vh' }}>
            <aside style={{ width: '250px', background: 'var(--color-bg-secondary)', padding: '20px', display: 'flex', flexDirection: 'column', borderRight: '1px solid var(--color-bg-card)' }}>
                <h2 style={{ marginBottom: '40px', color: 'var(--color-accent)' }}>EndoRecord</h2>
                <nav style={{ flex: 1 }}>
                    {navItems.map((item) => (
                        <div
                            key={item.path}
                            onClick={() => navigate(item.path)}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '10px', padding: '12px',
                                marginBottom: '5px', borderRadius: 'var(--radius)', cursor: 'pointer',
                                color: 'var(--color-text-secondary)',
                                transition: '0.2s'
                            }}
                            onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--color-bg-card)'; e.currentTarget.style.color = 'white' }}
                            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--color-text-secondary)' }}
                        >
                            <item.icon size={20} />
                            {item.label}
                        </div>
                    ))}
                </nav>
                <div onClick={() => navigate('/')} style={{ cursor: 'pointer', display: 'flex', gap: '10px', color: 'var(--color-danger)', padding: '12px' }}>
                    <LogOut size={20} /> Logout
                </div>
            </aside>
            <main style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
                {children}
            </main>
        </div>
    );
};

export default DashboardLayout;
