import { app, BrowserWindow, ipcMain, dialog, screen } from 'electron';
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
const collectionSessions = new Map();
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp']);
const ANNOTATION_HEADERS = [
  'filename', 'class_id', 'class_name', 'has_enemy', 'x_center',
  'y_center', 'width', 'height', 'video_id', 'frame_idx',
  'timestamp', 'confidence', 'auto_labeled', 'bbox_source', 'aug_type'
];

function parseCsvLine(line) {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      values.push(current);
      current = '';
      continue;
    }

    current += char;
  }

  values.push(current);
  return values;
}

function escapeCsvValue(value) {
  const text = String(value ?? '');
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

function readAnnotationRows(csvPath) {
  if (!fs.existsSync(csvPath)) return [];

  const raw = fs.readFileSync(csvPath, 'utf-8').replace(/^\uFEFF/, '').trim();
  if (!raw) return [];

  const lines = raw.split(/\r?\n/).filter(Boolean);
  if (lines.length <= 1) return [];

  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? '';
    });
    return row;
  });
}

function writeAnnotationRows(csvPath, rows) {
  const ordered = [...rows].sort((a, b) => {
    const aFile = String(a.filename || '');
    const bFile = String(b.filename || '');
    if (aFile !== bFile) return aFile.localeCompare(bFile, undefined, { numeric: true, sensitivity: 'base' });
    const aFrame = Number(a.frame_idx || 0);
    const bFrame = Number(b.frame_idx || 0);
    if (aFrame !== bFrame) return aFrame - bFrame;
    return Number(a.annotation_idx || 0) - Number(b.annotation_idx || 0);
  });

  const lines = [
    ANNOTATION_HEADERS.join(','),
    ...ordered.map((row) => ANNOTATION_HEADERS.map((header) => escapeCsvValue(row[header] ?? '')).join(',')),
  ];
  fs.writeFileSync(csvPath, `${lines.join('\n')}\n`, 'utf-8');
}

function getDatasetImagesDir(datasetPath) {
  return path.join(datasetPath, 'images');
}

function getDatasetLabelsDir(datasetPath) {
  return path.join(datasetPath, 'labels');
}

function getImageBasename(filename) {
  return path.parse(filename).name;
}

function findDatasetImagePath(datasetPath, filename) {
  const imagesDir = getDatasetImagesDir(datasetPath);
  const direct = path.join(imagesDir, filename);
  if (fs.existsSync(direct)) return direct;

  if (!fs.existsSync(imagesDir)) return null;
  const matches = fs.readdirSync(imagesDir).filter((entry) => entry === path.basename(filename));
  if (matches.length > 0) {
    return path.join(imagesDir, matches[0]);
  }
  return null;
}

function normalizeEditorBoxes(boxes) {
  return (Array.isArray(boxes) ? boxes : [])
    .filter(Boolean)
    .map((item) => ({
      class_id: Number(item.class_id ?? 0),
      class_name: String(item.class_name ?? (Number(item.class_id ?? 0) === 1 ? 'player' : 'enemy')),
      x_center: Number(item.x_center ?? 0.5),
      y_center: Number(item.y_center ?? 0.5),
      width: Number(item.width ?? 0),
      height: Number(item.height ?? 0),
      confidence: Number(item.confidence ?? 1),
    }))
    .filter((item) => Number.isFinite(item.x_center) && Number.isFinite(item.y_center) && Number.isFinite(item.width) && Number.isFinite(item.height));
}

function rowsToBoxes(rows) {
  return rows
    .filter((row) => Number(row.has_enemy ?? 1) !== 0)
    .map((row, index) => ({
      id: `${row.filename || 'box'}-${index}`,
      class_id: Number(row.class_id ?? 0),
      class_name: String(row.class_name ?? (Number(row.class_id ?? 0) === 1 ? 'player' : 'enemy')),
      x_center: Number(row.x_center ?? 0.5),
      y_center: Number(row.y_center ?? 0.5),
      width: Number(row.width ?? 0),
      height: Number(row.height ?? 0),
      confidence: Number(row.confidence ?? 1),
    }));
}

function buildAnnotationRow(template, filename, box, annotationIndex, bboxSource) {
  return {
    filename,
    class_id: String(box.class_id),
    class_name: box.class_name,
    has_enemy: '1',
    x_center: box.x_center.toFixed(6),
    y_center: box.y_center.toFixed(6),
    width: box.width.toFixed(6),
    height: box.height.toFixed(6),
    video_id: String(template.video_id ?? path.parse(filename).name),
    frame_idx: String(template.frame_idx ?? 0),
    timestamp: Number(template.timestamp ?? 0).toFixed(6),
    confidence: Number(box.confidence ?? template.confidence ?? 1).toFixed(4),
    auto_labeled: 'False',
    bbox_source: bboxSource,
    aug_type: String(template.aug_type ?? ''),
    annotation_idx: annotationIndex,
  };
}

function buildNegativeRow(template, filename) {
  return {
    filename,
    class_id: '0',
    class_name: 'enemy',
    has_enemy: '0',
    x_center: '0.500000',
    y_center: '0.500000',
    width: '0.000000',
    height: '0.000000',
    video_id: String(template.video_id ?? path.parse(filename).name),
    frame_idx: String(template.frame_idx ?? 0),
    timestamp: Number(template.timestamp ?? 0).toFixed(6),
    confidence: '0.0000',
    auto_labeled: 'False',
    bbox_source: 'editor_negative',
    aug_type: String(template.aug_type ?? ''),
    annotation_idx: 0,
  };
}

function writeYoloLabelFile(labelPath, boxes) {
  fs.mkdirSync(path.dirname(labelPath), { recursive: true });
  const lines = normalizeEditorBoxes(boxes)
    .map((item) => `${item.class_id} ${item.x_center.toFixed(6)} ${item.y_center.toFixed(6)} ${item.width.toFixed(6)} ${item.height.toFixed(6)}`);
  fs.writeFileSync(labelPath, lines.length > 0 ? `${lines.join('\n')}\n` : '', 'utf-8');
}

function sanitizeDatasetName(name) {
  const fallback = `collected_${new Date().toISOString().split('T')[0]}`;
  return (name || fallback)
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, '_')
    .replace(/\s+/g, '_')
    .slice(0, 80) || fallback;
}

function getCollectionSession(projectRoot, datasetName, videoName) {
  const safeName = sanitizeDatasetName(datasetName);
  const datasetPath = path.join(projectRoot, 'data_sets', safeName);
  const imagesDir = path.join(datasetPath, 'images');
  const labelsDir = path.join(datasetPath, 'labels');
  fs.mkdirSync(imagesDir, { recursive: true });
  fs.mkdirSync(labelsDir, { recursive: true });

  const key = datasetPath;
  if (!collectionSessions.has(key)) {
    collectionSessions.set(key, {
      datasetName: safeName,
      datasetPath,
      imagesDir,
      labelsDir,
      videoId: path.parse(videoName).name,
      rowsByFilename: new Map(),
    });
  }

  return collectionSessions.get(key);
}

function writeCollectionCsv(session) {
  const csvPath = path.join(session.datasetPath, 'labels_enhanced.csv');
  const headers = [
    'filename', 'class_id', 'class_name', 'has_enemy', 'x_center',
    'y_center', 'width', 'height', 'video_id', 'frame_idx',
    'timestamp', 'confidence', 'auto_labeled', 'bbox_source', 'aug_type'
  ];

  const rows = [...session.rowsByFilename.values()]
    .flat()
    .sort((a, b) => {
      if (a.frame_idx !== b.frame_idx) return a.frame_idx - b.frame_idx;
      return (a.annotation_idx || 0) - (b.annotation_idx || 0);
    });
  const csvLines = [
    headers.join(','),
    ...rows.map((row) => headers.map((header) => row[header]).join(',')),
  ];
  fs.writeFileSync(csvPath, `${csvLines.join('\n')}\n`, 'utf-8');
}

function getSessionAnnotationCount(session) {
  return [...session.rowsByFilename.values()].reduce((total, rows) => total + rows.length, 0);
}

function getInitialWindowBounds() {
  const { width: workWidth, height: workHeight } = screen.getPrimaryDisplay().workAreaSize;
  const maxWidth = Math.max(1280, Math.floor(workWidth * 0.96));
  const maxHeight = Math.max(720, Math.floor(workHeight * 0.94));

  let width = Math.min(1728, maxWidth);
  let height = Math.round(width * 9 / 16);

  if (height > maxHeight) {
    height = maxHeight;
    width = Math.round(height * 16 / 9);
  }

  return {
    width: Math.min(width, workWidth),
    height: Math.min(height, workHeight),
  };
}

function createWindow() {
  const { width, height } = getInitialWindowBounds();
  mainWindow = new BrowserWindow({
    width,
    height,
    minWidth: 1100,
    minHeight: 680,
    frame: false, // Make window frameless
    titleBarStyle: 'hidden',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false, // Required to load local files via file://
    },
    title: "Enemy Detection Model Tester",
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
  return new Promise((resolve) => {
    const projectRoot = path.join(__dirname, '..');
    const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');

    const candidatePaths = [
      path.join(projectRoot, scriptName),
      path.join(projectRoot, 'src', scriptName),
      path.join(projectRoot, 'code_archive', scriptName),
    ];
    const scriptPath = candidatePaths.find((candidate) => fs.existsSync(candidate));

    if (!fs.existsSync(pythonExe)) {
      resolve({ error: `Python environment not found at ${pythonExe}. Run setup_venv.ps1 first.` });
      return;
    }

    if (!scriptPath) {
      resolve({ error: `Script not found: ${scriptName}` });
      return;
    }

    const child = spawn(pythonExe, [scriptPath, ...argsArray], {
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

    child.on('error', (error) => {
      currentProcess = null;
      resolve({ error: error.message });
    });

    child.on('close', (code) => {
      currentProcess = null;
      if (code === 0) resolve({ success: true, scriptPath });
      else resolve({ error: `${path.basename(scriptPath)} failed with code ${code}` });
    });
  });
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

ipcMain.handle('list-dataset-images', async (event, datasetPath, csvName) => {
    try {
        const csvPath = path.join(datasetPath, csvName);
        const imagesDir = getDatasetImagesDir(datasetPath);
        const rows = readAnnotationRows(csvPath);
        const rowsByFilename = new Map();

        rows.forEach((row) => {
            const filename = String(row.filename || '').trim();
            if (!filename) return;
            if (!rowsByFilename.has(filename)) {
                rowsByFilename.set(filename, []);
            }
            rowsByFilename.get(filename).push(row);
        });

        const imageNames = new Set();
        if (fs.existsSync(imagesDir)) {
            fs.readdirSync(imagesDir)
                .filter((entry) => IMAGE_EXTENSIONS.has(path.extname(entry).toLowerCase()))
                .forEach((entry) => imageNames.add(entry));
        }
        rowsByFilename.forEach((_, filename) => imageNames.add(filename));

        const entries = [...imageNames]
            .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
            .map((filename) => {
                const imagePath = findDatasetImagePath(datasetPath, filename);
                const imageRows = rowsByFilename.get(filename) || [];
                const boxes = rowsToBoxes(imageRows);
                return {
                    filename,
                    imagePath,
                    boxes,
                    boxCount: boxes.length,
                    missingImage: !imagePath,
                };
            });

        return {
            success: true,
            entries,
            datasetPath,
            csvName,
        };
    } catch (error) {
        return { error: error.message };
    }
});

ipcMain.handle('save-dataset-image-annotations', async (event, payload) => {
    try {
        const {
            datasetPath,
            csvName,
            filename,
            boxes,
            bboxSource = 'editor_box',
        } = payload;

        const csvPath = path.join(datasetPath, csvName);
        const labelsDir = getDatasetLabelsDir(datasetPath);
        const labelPath = path.join(labelsDir, `${getImageBasename(filename)}.txt`);
        const existingRows = readAnnotationRows(csvPath);
        const retainedRows = existingRows.filter((row) => row.filename !== filename);
        const sourceRows = existingRows.filter((row) => row.filename === filename);
        const template = sourceRows[0] || {
            filename,
            video_id: path.parse(filename).name,
            frame_idx: 0,
            timestamp: 0,
            confidence: 1,
            aug_type: '',
        };

        const normalizedBoxes = normalizeEditorBoxes(boxes);
        const replacementRows = normalizedBoxes.length > 0
            ? normalizedBoxes.map((box, index) => buildAnnotationRow(template, filename, box, index, bboxSource))
            : [buildNegativeRow(template, filename)];

        writeAnnotationRows(csvPath, [...retainedRows, ...replacementRows]);
        writeYoloLabelFile(labelPath, normalizedBoxes);

        return {
            success: true,
            filename,
            boxCount: normalizedBoxes.length,
            boxes: normalizedBoxes,
        };
    } catch (error) {
        return { error: error.message };
    }
});

ipcMain.handle('delete-dataset-image', async (event, datasetPath, csvName, filename) => {
    try {
        const csvPath = path.join(datasetPath, csvName);
        const labelPath = path.join(getDatasetLabelsDir(datasetPath), `${getImageBasename(filename)}.txt`);
        const imagePath = findDatasetImagePath(datasetPath, filename);

        if (imagePath && fs.existsSync(imagePath)) {
            fs.unlinkSync(imagePath);
        }
        if (fs.existsSync(labelPath)) {
            fs.unlinkSync(labelPath);
        }

        const rows = readAnnotationRows(csvPath).filter((row) => row.filename !== filename);
        writeAnnotationRows(csvPath, rows);

        return { success: true, filename };
    } catch (error) {
        return { error: error.message };
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

ipcMain.handle('run-training', async (event, payload = {}) => {
    return new Promise((resolve) => {
        const projectRoot = path.join(__dirname, '..');
        const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
        const trainScript = path.join(projectRoot, 'src', 'train.py');
        const {
            datasetPath,
            csvName,
            epochs = 10,
            batchSize = 16,
            imageSize = 640,
            deviceMode = 'cuda',
            modelChoice = 'yolov8n',
        } = payload;

        if (!datasetPath || !csvName) {
            resolve({ error: 'Training requires both a dataset path and CSV name.' });
            return;
        }
        
        const csvPath = path.join(datasetPath, csvName);
        const imgDir = path.join(datasetPath, 'images');

        const trainArgs = [
            trainScript, 
            '--train_dir', imgDir, 
            '--train_csv', csvPath,
            '--epochs', epochs.toString(),
            '--batch_size', batchSize.toString(),
            '--imgsz', imageSize.toString(),
            '--device_mode', deviceMode,
            '--model', modelChoice
        ];

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
        return files
            .filter(f => f.endsWith('.mp4') || f.endsWith('.mov') || f.endsWith('.avi'))
            .map((name) => ({
                name,
                path: path.join(videosDir, name),
            }));
    } catch (error) {
        console.error("Error listing videos:", error);
        return [];
    }
});

ipcMain.handle('start-manual-collection', async (event, videoName, datasetName) => {
    try {
        const projectRoot = path.join(__dirname, '..');
        const videoPath = path.join(projectRoot, 'src', 'videos', videoName);
        if (!fs.existsSync(videoPath)) {
            return { error: `Video not found: ${videoName}` };
        }

        const session = getCollectionSession(projectRoot, datasetName, videoName);
        writeCollectionCsv(session);

        return {
            success: true,
            datasetName: session.datasetName,
            datasetPath: session.datasetPath,
            videoName,
            videoPath,
            savedCount: getSessionAnnotationCount(session),
        };
    } catch (error) {
        return { error: error.message };
    }
});

ipcMain.handle('save-manual-annotation', async (event, payload) => {
    try {
        const {
            datasetPath,
            datasetName,
            videoName,
            frameIndex,
            timestamp,
            bbox,
            boxes,
            imageDataUrl,
            bboxSource = 'manual_box',
        } = payload;

        const projectRoot = path.join(__dirname, '..');
        const session = getCollectionSession(projectRoot, datasetName || path.basename(datasetPath), videoName);
        const videoId = path.parse(videoName).name;
        const safeFrameIndex = String(frameIndex).padStart(6, '0');
        const filename = `${videoId}_${safeFrameIndex}.png`;

        const base64Data = imageDataUrl.split(',')[1];
        if (!base64Data) {
            return { error: 'Invalid frame image payload.' };
        }

        const imageBuffer = Buffer.from(base64Data, 'base64');
        fs.writeFileSync(path.join(session.imagesDir, filename), imageBuffer);

        const normalizedBoxes = (Array.isArray(boxes) && boxes.length > 0 ? boxes : [bbox])
            .filter(Boolean)
            .map((item) => ({
                class_id: Number(item.class_id ?? 0),
                class_name: String(item.class_name ?? (Number(item.class_id ?? 0) === 1 ? 'player' : 'enemy')),
                x_center: Number(item.x_center),
                y_center: Number(item.y_center),
                width: Number(item.width),
                height: Number(item.height),
                confidence: Number(item.confidence ?? 1),
            }));

        const labelPath = path.join(session.labelsDir, `${path.parse(filename).name}.txt`);
        const imagePath = path.join(session.imagesDir, filename);

        if (normalizedBoxes.length === 0) {
            session.rowsByFilename.delete(filename);
            if (fs.existsSync(labelPath)) {
                fs.unlinkSync(labelPath);
            }
            if (fs.existsSync(imagePath)) {
                fs.unlinkSync(imagePath);
            }
            writeCollectionCsv(session);
            return {
                success: true,
                filename,
                savedCount: getSessionAnnotationCount(session),
                frameBoxCount: 0,
                datasetPath: session.datasetPath,
            };
        }

        const labelContents = normalizedBoxes
            .map((item) => `${item.class_id} ${item.x_center.toFixed(6)} ${item.y_center.toFixed(6)} ${item.width.toFixed(6)} ${item.height.toFixed(6)}`)
            .join('\n');
        fs.writeFileSync(labelPath, `${labelContents}\n`, 'utf-8');

        session.rowsByFilename.set(filename, normalizedBoxes.map((item, index) => ({
            filename,
            class_id: item.class_id,
            class_name: item.class_name,
            has_enemy: 1,
            x_center: item.x_center.toFixed(6),
            y_center: item.y_center.toFixed(6),
            width: item.width.toFixed(6),
            height: item.height.toFixed(6),
            video_id: videoId,
            frame_idx: Number(frameIndex),
            timestamp: Number(timestamp).toFixed(6),
            confidence: item.confidence.toFixed(4),
            auto_labeled: 'False',
            bbox_source: bboxSource,
            aug_type: '',
            annotation_idx: index,
        })));

        writeCollectionCsv(session);

        return {
            success: true,
            filename,
            savedCount: getSessionAnnotationCount(session),
            frameBoxCount: normalizedBoxes.length,
            datasetPath: session.datasetPath,
        };
    } catch (error) {
        return { error: error.message };
    }
});

ipcMain.handle('finish-manual-collection', async (event, datasetPath) => {
    try {
        const session = collectionSessions.get(datasetPath);
        if (!session) {
            return { success: true, savedCount: 0 };
        }
        writeCollectionCsv(session);
        collectionSessions.delete(datasetPath);
        return { success: true, savedCount: getSessionAnnotationCount(session), datasetPath };
    } catch (error) {
        return { error: error.message };
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
