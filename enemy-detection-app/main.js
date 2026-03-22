import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import isDev from 'electron-is-dev';
import fs from 'fs';
import process from 'process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let mainWindow;
let currentProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    frame: false, // Make window frameless
    titleBarStyle: 'hidden',
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

// IPC handlers for Window Controls
ipcMain.on('window-minimize', () => mainWindow.minimize());
ipcMain.on('window-maximize', () => {
  if (mainWindow.isMaximized()) mainWindow.unmaximize();
  else mainWindow.maximize();
});
ipcMain.on('window-close', () => mainWindow.close());

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

    // Get downloads path and create a unique filename
    const downloadsPath = app.getPath('downloads');
    const timestamp = new Date().getTime();
    const savePath = path.join(downloadsPath, `prediction_${timestamp}.png`);

    const pythonProcess = spawn(pythonExe, [scriptPath, imagePath, '--save_path', savePath], {
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

ipcMain.handle('cancel-pipeline', async () => {
  if (currentProcess) {
    spawn("taskkill", ["/pid", currentProcess.pid, '/f', '/t']);
    currentProcess = null;
    return { success: true };
  }
  return { error: 'No process running' };
});

ipcMain.handle('run-pipeline-step', async (event, scriptName, argsArray = []) => {
  // ... (keep existing implementation)
});

// --- NEW DATASET MANAGEMENT HANDLERS ---

ipcMain.handle('list-datasets', async () => {
    try {
        const projectRoot = path.join(__dirname, '..');
        const dataSetsDir = path.join(projectRoot, 'data_sets');
        if (!fs.existsSync(dataSetsDir)) return [];

        const results = [];
        const items = fs.readdirSync(dataSetsDir, { withFileTypes: true });

        for (const item of items) {
            if (item.isDirectory()) {
                const itemPath = path.join(dataSetsDir, item.name);
                // Look for CSV files in this directory or subdirectories
                const files = fs.readdirSync(itemPath);
                const csvFiles = files.filter(f => f.endsWith('.csv') || f.endsWith('.csv.backup'));
                
                if (csvFiles.length > 0) {
                    results.push({
                        name: item.name,
                        path: itemPath,
                        csvs: csvFiles
                    });
                } else {
                    // Check one level deeper (e.g., train/labels.csv)
                    const subdirs = fs.readdirSync(itemPath, { withFileTypes: true })
                        .filter(sd => sd.isDirectory());
                    
                    for (const sd of subdirs) {
                        const sdPath = path.join(itemPath, sd.name);
                        const sdFiles = fs.readdirSync(sdPath);
                        const sdCsvs = sdFiles.filter(f => f.endsWith('.csv') || f.endsWith('.csv.backup'));
                        if (sdCsvs.length > 0) {
                            results.push({
                                name: `${item.name}/${sd.name}`,
                                path: sdPath,
                                csvs: sdCsvs
                            });
                        }
                    }
                }
            }
        }
        return results;
    } catch (error) {
        console.error("Error listing datasets:", error);
        return [];
    }
});

ipcMain.handle('analyze-dataset-bias', async (event, datasetPath, csvName) => {
    return new Promise((resolve) => {
        const projectRoot = path.join(__dirname, '..');
        const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
        const scriptPath = path.join(projectRoot, 'src', 'visualize_dataset.py');
        const csvPath = path.join(datasetPath, csvName);
        
        // Output image in the dataset folder for persistence
        const outputPath = path.join(datasetPath, 'bias_visual_app.png');

        const pythonProcess = spawn(pythonExe, [scriptPath, '--csv', csvPath, '--output', outputPath], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) resolve({ success: true, imagePath: outputPath });
            else resolve({ error: `Analysis failed with code ${code}` });
        });
    });
});

ipcMain.handle('run-bias-fix', async (event, datasetPath, csvName) => {
    return new Promise((resolve) => {
        const projectRoot = path.join(__dirname, '..');
        const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
        const cleanScript = path.join(projectRoot, 'src', 'clean_dataset_remove_bias.py');
        const vizScript = path.join(projectRoot, 'src', 'visualize_dataset.py');
        
        const inputCsv = path.join(datasetPath, csvName);
        const imgDir = path.join(datasetPath, 'images');
        
        // Output to a "cleaned" subfolder
        const outputDir = path.join(datasetPath, 'cleaned_balanced');
        const cleanedCsv = path.join(outputDir, 'labels_cleaned.csv');
        const finalVizPath = path.join(outputDir, 'bias_after_fix.png');

        // 1. Run Cleaning
        const cleanProcess = spawn(pythonExe, [cleanScript, '--csv', inputCsv, '--img_dir', imgDir, '--output_dir', outputDir], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
        });

        cleanProcess.on('close', (cleanCode) => {
            if (cleanCode !== 0) return resolve({ error: `Cleaning failed with code ${cleanCode}` });

            // 2. Run Visualization on Cleaned Data
            const vizProcess = spawn(pythonExe, [vizScript, '--csv', cleanedCsv, '--output', finalVizPath], {
                cwd: projectRoot,
                env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
            });

            vizProcess.on('close', (vizCode) => {
                if (vizCode === 0) resolve({ success: true, imagePath: finalVizPath, csvPath: cleanedCsv });
                else resolve({ error: `Visualization failed with code ${vizCode}` });
            });
        });
    });
});

ipcMain.handle('run-training', async (event, datasetPath, csvName, epochs = 10) => {
    return new Promise((resolve) => {
        const projectRoot = path.join(__dirname, '..');
        const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
        const trainScript = path.join(projectRoot, 'src', 'train.py');
        
        const csvPath = path.join(datasetPath, csvName);
        const imgDir = path.join(datasetPath, 'images');
        
        // Check for validation data in cleaned_balanced subfolder
        const cleanedDir = path.join(datasetPath, 'cleaned_balanced');
        let valCsv = null;
        let valDir = null;
        
        if (fs.existsSync(cleanedDir)) {
            const cleanedFiles = fs.readdirSync(cleanedDir);
            const valCsvFile = cleanedFiles.find(f => f.includes('labels') && f.endsWith('.csv'));
            if (valCsvFile) {
                valCsv = path.join(cleanedDir, valCsvFile);
                valDir = path.join(cleanedDir, 'images');
            }
        }
        
        // If no validation data found, let training script handle auto-split
        const trainArgs = [
            trainScript, 
            '--train_dir', imgDir, 
            '--train_csv', csvPath,
            '--epochs', epochs.toString()
        ];
        
        if (valCsv && valDir) {
            trainArgs.push('--val_csv', valCsv);
            trainArgs.push('--val_dir', valDir);
        }

        const trainProcess = spawn(pythonExe, trainArgs, {
            cwd: projectRoot,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
        });

        trainProcess.stdout.on('data', (data) => {
            mainWindow.webContents.send('pipeline-output', { type: 'stdout', msg: data.toString() });
        });

        trainProcess.stderr.on('data', (data) => {
            mainWindow.webContents.send('pipeline-output', { type: 'stderr', msg: data.toString() });
        });

        trainProcess.on('close', (code) => {
            if (code === 0) resolve({ success: true });
            else resolve({ error: `Training failed with code ${code}` });
        });
    });
});

ipcMain.handle('list-videos', async () => {
    try {
        const projectRoot = path.join(__dirname, '..');
        const videosDir = path.join(projectRoot, 'src', 'videos');
        if (!fs.existsSync(videosDir)) return [];

        const files = fs.readdirSync(videosDir);
        return files.filter(f => f.endsWith('.mp4') || f.endsWith('.mov') || f.endsWith('.avi'));
    } catch (error) {
        console.error("Error listing videos:", error);
        return [];
    }
});

ipcMain.handle('run-data-collection', async (event, videoName, datasetName) => {
    return new Promise((resolve) => {
        const projectRoot = path.join(__dirname, '..');
        const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
        const scriptPath = path.join(projectRoot, 'src', 'process_video_improved.py');
        const videoPath = path.join(projectRoot, 'src', 'videos', videoName);
        
        // Default name if none provided
        const finalName = datasetName || `collected_${new Date().toISOString().split('T')[0]}_${Math.floor(Math.random() * 1000)}`;
        const outputDir = path.join(projectRoot, 'data_sets', finalName);

        const child = spawn(pythonExe, [
            scriptPath,
            '--video_file', videoPath,
            '--videos_dir', path.join(projectRoot, 'src', 'videos'),
            '--output_dir', outputDir,
            '--auto_skip'
        ], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
        });

        currentProcess = child;

        child.stdout.on('data', (data) => {
            mainWindow.webContents.send('pipeline-output', { type: 'stdout', msg: data.toString() });
        });

        child.stderr.on('data', (data) => {
            mainWindow.webContents.send('pipeline-output', { type: 'stderr', msg: data.toString() });
        });

        child.on('close', (code) => {
            currentProcess = null;
            if (code === 0) resolve({ success: true });
            else resolve({ error: `Process failed with code ${code}` });
        });
    });
});

ipcMain.handle('run-augmentation', async (event, datasetPath, csvName, outputDatasetName) => {
    return new Promise((resolve) => {
        const projectRoot = path.join(__dirname, '..');
        const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
        const scriptPath = path.join(projectRoot, 'src', 'augment_dataset_improved.py');
        
        const inputCsv = path.join(datasetPath, csvName);
        const inputDir = path.join(datasetPath, 'images');
        
        // Default name if none provided
        const finalName = outputDatasetName || `${path.basename(datasetPath)}_augmented`;
        const outputDir = path.join(projectRoot, 'data_sets', finalName);

        const child = spawn(pythonExe, [
            scriptPath,
            '--input_csv', inputCsv,
            '--input_dir', inputDir,
            '--output_dir', outputDir
        ], {
            cwd: projectRoot,
            env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
        });

        currentProcess = child;

        child.stdout.on('data', (data) => {
            mainWindow.webContents.send('pipeline-output', { type: 'stdout', msg: data.toString() });
        });

        child.stderr.on('data', (data) => {
            mainWindow.webContents.send('pipeline-output', { type: 'stderr', msg: data.toString() });
        });

        child.on('close', (code) => {
            currentProcess = null;
            if (code === 0) resolve({ success: true, outputDir });
            else resolve({ error: `Augmentation failed with code ${code}` });
        });
    });
});
