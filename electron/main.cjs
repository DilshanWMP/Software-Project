const { app, BrowserWindow, ipcMain, protocol } = require('electron');
const path = require('path');
const fs = require('fs');
const mysql = require('mysql2/promise');

// MySQL Connection Pool
const mysqlPool = mysql.createPool({
  host: 'localhost',
  user: 'root',
  password: 'Kavindu@128',
  database: 'endoscopy',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// Test MySQL connection and Initialize DB tables
mysqlPool.getConnection()
  .then(async connection => {
    console.log('Connected to MySQL database: endoscopy');

    try {
      await connection.query(`
        CREATE TABLE IF NOT EXISTS patients (
          id INT AUTO_INCREMENT PRIMARY KEY,
          patientId VARCHAR(255) UNIQUE,
          name VARCHAR(255),
          age INT,
          gender VARCHAR(50),
          phone VARCHAR(50),
          dateAdded VARCHAR(255)
        )
      `);

      await connection.query(`
        CREATE TABLE IF NOT EXISTS media (
          id INT AUTO_INCREMENT PRIMARY KEY,
          patientId VARCHAR(255),
          type VARCHAR(50), -- 'video' or 'image'
          filePath TEXT,
          timestamp VARCHAR(255),
          FOREIGN KEY(patientId) REFERENCES patients(patientId)
        )
      `);
      console.log('Database tables initialized');
    } catch (err) {
      console.error('Error initializing tables:', err);
    } finally {
      connection.release();
    }
  })
  .catch(err => {
    console.error('Error connecting to MySQL:', err);
  });

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false // Allow loading local files for now (videos)
    },
    title: "Endoscopy Recording System",
    backgroundColor: '#1a1b26', // Dark theme background
  });

  // Check if in development
  const isDev = !app.isPackaged;

  if (isDev) {
    win.loadURL('http://localhost:5173');
    win.webContents.openDevTools();
  } else {
    win.loadFile(path.join(__dirname, '../dist/index.html'));
  }

  // Hide menu bar
  win.setMenuBarVisibility(false);
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC Handlers

// Get all patients
ipcMain.handle('get-patients', async () => {
  try {
    const [rows] = await mysqlPool.query('SELECT * FROM patients ORDER BY dateAdded DESC');
    return rows;
  } catch (err) {
    console.error(err);
    return [];
  }
});

// Add patient
ipcMain.handle('add-patient', async (event, patient) => {
  try {
    const sql = `
      INSERT INTO patients (patientId, name, age, gender, phone, dateAdded)
      VALUES (?, ?, ?, ?, ?, ?)
    `;
    const [result] = await mysqlPool.execute(sql, [
      patient.patientId,
      patient.name,
      patient.age,
      patient.gender,
      patient.phone,
      patient.dateAdded
    ]);
    return { success: true, id: result.insertId };
  } catch (err) {
    console.error(err);
    return { success: false, error: err.message };
  }
});

// Update patient
ipcMain.handle('update-patient', async (event, patient) => {
  try {
    const sql = `
      UPDATE patients 
      SET name = ?, age = ?, gender = ?, phone = ? 
      WHERE patientId = ?
    `;
    const [result] = await mysqlPool.execute(sql, [
      patient.name,
      patient.age,
      patient.gender,
      patient.phone,
      patient.patientId
    ]);
    return { success: true, changes: result.affectedRows };
  } catch (err) {
    console.error(err);
    return { success: false, error: err.message };
  }
});

// Delete patient
ipcMain.handle('delete-patient', async (event, patientId) => {
  try {
    const sql = 'DELETE FROM patients WHERE patientId = ?';
    const [result] = await mysqlPool.execute(sql, [patientId]);
    return { success: true, changes: result.affectedRows };
  } catch (err) {
    console.error(err);
    return { success: false, error: err.message };
  }
});

// Save Media Metadata
ipcMain.handle('save-media', async (event, media) => {
  try {
    const sql = `
      INSERT INTO media (patientId, type, filePath, timestamp)
      VALUES (?, ?, ?, ?)
    `;
    await mysqlPool.execute(sql, [
      media.patientId,
      media.type,
      media.filePath,
      media.timestamp
    ]);
    return { success: true };
  } catch (err) {
    console.error(err);
    return { success: false, error: err.message };
  }
});

// Get Media for Patient
ipcMain.handle('get-patient-media', async (event, patientId) => {
  try {
    const sql = 'SELECT * FROM media WHERE patientId = ? ORDER BY timestamp DESC';
    const [rows] = await mysqlPool.execute(sql, [patientId]);
    return rows;
  } catch (err) {
    console.error(err);
    return [];
  }
});

// Get All Media (For Gallery)
ipcMain.handle('get-all-media', async () => {
  try {
    const sql = 'SELECT m.*, p.name as patientName FROM media m LEFT JOIN patients p ON m.patientId = p.patientId ORDER BY m.timestamp DESC';
    const [rows] = await mysqlPool.query(sql);
    return rows;
  } catch (err) {
    console.error(err);
    return [];
  }
});

// Save Buffer (Video/Image) to Disk
ipcMain.handle('save-buffer', async (event, buffer, filename) => {
  const savePath = path.join(app.getPath('userData'), 'recordings');
  if (!fs.existsSync(savePath)) {
    fs.mkdirSync(savePath, { recursive: true });
  }
  const fullPath = path.join(savePath, filename);
  fs.writeFileSync(fullPath, buffer);
  return { success: true, path: fullPath };
});
