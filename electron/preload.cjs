const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Patient Operations
    getPatients: () => ipcRenderer.invoke('get-patients'),
    addPatient: (patient) => ipcRenderer.invoke('add-patient', patient),
    updatePatient: (patient) => ipcRenderer.invoke('update-patient', patient),
    deletePatient: (patientId) => ipcRenderer.invoke('delete-patient', patientId),

    // Media Operations
    saveMedia: (media) => ipcRenderer.invoke('save-media', media),
    getPatientMedia: (patientId) => ipcRenderer.invoke('get-patient-media', patientId),
    getAllMedia: () => ipcRenderer.invoke('get-all-media'),

    // Save File (Actual file writing will be done in Main or via Renderer logic? 
    // For videos, we might record in Renderer using MediaRecorder and send Blob to Main to save.
    // For now, let's assume Renderer handles basic logic, but Main saves the buffer.)
    saveBuffer: (buffer, filePath) => ipcRenderer.invoke('save-buffer', buffer, filePath), // To implemented
});
