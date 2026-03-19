import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import isDev from 'electron-is-dev';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false, // Required to load local files via file://
    },
    title: "Enemy Localization Model Tester",
    backgroundColor: '#020617',
  });

  const startURL = isDev
    ? 'http://localhost:5173'
    : `file://${path.join(__dirname, 'dist/index.html')}`;

  mainWindow.loadURL(startURL);

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => (mainWindow = null));
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// IPC handlers
ipcMain.handle('select-image', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Images', extensions: ['jpg', 'png', 'jpeg', 'bmp'] }]
  });
  if (canceled) {
    return null;
  } else {
    return filePaths[0];
  }
});

ipcMain.handle('run-prediction', async (event, imagePath) => {
  return new Promise((resolve, reject) => {
    const projectRoot = path.join(__dirname, '..');
    const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(projectRoot, 'src', 'predict_cli.py');

    const pythonProcess = spawn(pythonExe, [scriptPath, imagePath], {
      cwd: projectRoot,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let dataString = '';
    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        resolve({ error: `Process exited with code ${code}` });
      } else {
        try {
          const result = JSON.parse(dataString);
          resolve(result);
        } catch (e) {
          resolve({ error: "Failed to parse Python output" });
        }
      }
    });
  });
});

ipcMain.handle('clear-videos', async (event) => {
  try {
    const projectRoot = path.join(__dirname, '..');
    const videosDir = path.join(projectRoot, 'src', 'videos');
    if (fs.existsSync(videosDir)) {
      const files = fs.readdirSync(videosDir);
      for (const file of files) {
        if (file.endsWith('.mp4') || file.endsWith('.webm') || file.endsWith('.mkv') || file.endsWith('.part')) {
          fs.unlinkSync(path.join(videosDir, file));
        }
      }
    }
    return { success: true };
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('save-links', async (event, linksText) => {
  try {
    const projectRoot = path.join(__dirname, '..');
    const videosDir = path.join(projectRoot, 'src', 'videos');
    if (!fs.existsSync(videosDir)) {
      fs.mkdirSync(videosDir, { recursive: true });
    }
    const linksPath = path.join(videosDir, 'links.txt');
    fs.writeFileSync(linksPath, linksText, 'utf-8');
    return { success: true };
  } catch (error) {
    return { error: error.message };
  }
});

ipcMain.handle('run-pipeline-step', async (event, scriptName, argsArray = []) => {
  if (!mainWindow) return { error: "No window" };
  
  return new Promise((resolve, reject) => {
    const projectRoot = path.join(__dirname, '..');
    const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
    
    let scriptPath;
    if (scriptName === 'run_pipeline.py' || scriptName === 'reset_project.py') {
        scriptPath = path.join(projectRoot, scriptName);
    } else {
        scriptPath = path.join(projectRoot, 'src', scriptName);
    }

    const pythonProcess = spawn(pythonExe, [scriptPath, ...argsArray], {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    pythonProcess.stdout.on('data', (data) => {
      mainWindow.webContents.send('pipeline-output', { type: 'stdout', msg: data.toString() });
    });

    pythonProcess.stderr.on('data', (data) => {
      mainWindow.webContents.send('pipeline-output', { type: 'stderr', msg: data.toString() });
    });

    pythonProcess.on('close', (code) => {
      mainWindow.webContents.send('pipeline-output', { type: 'exit', msg: `Process exited with code ${code}` });
      if (code !== 0) {
        resolve({ error: `Process exited with code ${code}` });
      } else {
        resolve({ success: true });
      }
    });
    
    pythonProcess.on('error', (err) => {
      mainWindow.webContents.send('pipeline-output', { type: 'stderr', msg: `Failed to start process: ${err.message}` });
      resolve({ error: err.message });
    });
  });
});
