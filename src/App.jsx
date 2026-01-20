import React from 'react';
import { HashRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import Home from './pages/Home';
import Patients from './pages/Patients';
import LiveView from './pages/LiveView';
import PatientSelect from './pages/PatientSelect';

import Login from './pages/Login';
import RecordedVideos from './pages/RecordedVideos';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/home" element={<Home />} />
        <Route path="/live" element={<LiveView />} />
        <Route path="/live/select" element={<PatientSelect />} />
        <Route path="/patients" element={<Patients />} />
        <Route path="/videos" element={<RecordedVideos />} />
      </Routes>
    </Router>
  );
}

export default App;
