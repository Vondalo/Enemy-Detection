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
let videoPredictionProcess = null;
let videoPredictionSessionId = null;
let videoPredictionStopRequested = false;
const collectionSessions = new Map();
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.webp']);
const VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'webm', 'mkv'];
const MODEL_HISTORY_DIRNAME = 'history';
const ACTIVE_MODEL_FILENAME = 'active_model.json';
const IMAGE_ARTIFACT_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.bmp', '.webp']);
const ANNOTATION_HEADERS = [
  'filename', 'class_id', 'class_name', 'has_enemy', 'x_center',
  'y_center', 'width', 'height', 'video_id', 'frame_idx',
  'timestamp', 'confidence', 'auto_labeled', 'bbox_source', 'aug_type'
];

function sendPipelineOutput(type, msg) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('pipeline-output', { type, msg });
  }
}

function sendVideoPredictionEvent(payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('video-prediction-event', payload);
  }
}

function cleanupVideoPredictionState() {
  videoPredictionProcess = null;
  videoPredictionSessionId = null;
  videoPredictionStopRequested = false;
}

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

function resolveProjectPath(projectRoot, candidatePath) {
  if (!candidatePath) return null;
  return path.isAbsolute(candidatePath) ? candidatePath : path.join(projectRoot, candidatePath);
}

function getModelsRoot(projectRoot) {
  return path.join(projectRoot, 'models');
}

function getModelHistoryRoot(projectRoot) {
  return path.join(getModelsRoot(projectRoot), MODEL_HISTORY_DIRNAME);
}

function getActiveModelConfigPath(projectRoot) {
  return path.join(getModelsRoot(projectRoot), ACTIVE_MODEL_FILENAME);
}

function relativeProjectPath(projectRoot, targetPath) {
  if (!targetPath) return null;
  return path.relative(path.resolve(projectRoot), path.resolve(targetPath));
}

function resolveSummaryArtifactPath(projectRoot, summaryDir, candidatePath) {
  if (!candidatePath) return null;
  if (path.isAbsolute(candidatePath)) return candidatePath;

  if (summaryDir) {
    const fromSummaryDir = path.join(summaryDir, candidatePath);
    if (fs.existsSync(fromSummaryDir)) {
      return fromSummaryDir;
    }
  }

  return path.join(projectRoot, candidatePath);
}

function readJsonFileSafe(filePath) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch {
    return null;
  }
}

function readCsvFileSafe(csvPath) {
  if (!csvPath || !fs.existsSync(csvPath)) return [];

  const raw = fs.readFileSync(csvPath, 'utf-8').replace(/^\uFEFF/, '').trim();
  if (!raw) return [];

  const lines = raw.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) return [];

  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? '';
    });
    return row;
  });
}

function resolveTrainingSummaryPaths(projectRoot, summaryPath, summary) {
  if (!summary || typeof summary !== 'object') return summary;

  const summaryDir = summaryPath ? path.dirname(summaryPath) : projectRoot;
  const nextSummary = JSON.parse(JSON.stringify(summary));
  if (nextSummary.best_weights) {
    nextSummary.best_weights = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.best_weights);
  }
  if (nextSummary.stable_best_model) {
    nextSummary.stable_best_model = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.stable_best_model);
  }
  if (nextSummary.ultralytics_run_dir) {
    nextSummary.ultralytics_run_dir = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.ultralytics_run_dir);
  }
  if (nextSummary.artifact_manifest_path) {
    nextSummary.artifact_manifest_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.artifact_manifest_path);
  }
  if (nextSummary.dataset?.data_yaml) {
    nextSummary.dataset.data_yaml = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.dataset.data_yaml);
  }
  if (nextSummary.dataset?.test_images_dir) {
    nextSummary.dataset.test_images_dir = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.dataset.test_images_dir);
  }
  if (nextSummary.dataset?.test_labels_csv) {
    nextSummary.dataset.test_labels_csv = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.dataset.test_labels_csv);
  }
  if (nextSummary.evaluation?.metrics_path) {
    nextSummary.evaluation.metrics_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.evaluation.metrics_path);
  }
  if (nextSummary.evaluation?.review_manifest_path) {
    nextSummary.evaluation.review_manifest_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.evaluation.review_manifest_path);
  }
  if (nextSummary.evaluation?.review_csv_path) {
    nextSummary.evaluation.review_csv_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.evaluation.review_csv_path);
  }
  if (nextSummary.evaluation?.review_images_dir) {
    nextSummary.evaluation.review_images_dir = resolveSummaryArtifactPath(projectRoot, summaryDir, nextSummary.evaluation.review_images_dir);
  }
  return nextSummary;
}

function resolveArtifactManifestPaths(projectRoot, summaryPath, artifactManifest) {
  if (!artifactManifest || typeof artifactManifest !== 'object') return artifactManifest;

  const summaryDir = summaryPath ? path.dirname(summaryPath) : projectRoot;
  const nextManifest = JSON.parse(JSON.stringify(artifactManifest));
  if (nextManifest.ultralytics_run_dir) {
    nextManifest.ultralytics_run_dir = resolveSummaryArtifactPath(projectRoot, summaryDir, nextManifest.ultralytics_run_dir);
  }

  ['images', 'tables', 'weights'].forEach((collectionKey) => {
    if (!Array.isArray(nextManifest[collectionKey])) return;
    nextManifest[collectionKey] = nextManifest[collectionKey].map((item) => ({
      ...item,
      path: resolveSummaryArtifactPath(projectRoot, summaryDir, item.path),
    }));
  });

  if (nextManifest.evaluation?.metrics_path) {
    nextManifest.evaluation.metrics_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextManifest.evaluation.metrics_path);
  }
  if (nextManifest.evaluation?.review_manifest_path) {
    nextManifest.evaluation.review_manifest_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextManifest.evaluation.review_manifest_path);
  }
  if (nextManifest.evaluation?.review_csv_path) {
    nextManifest.evaluation.review_csv_path = resolveSummaryArtifactPath(projectRoot, summaryDir, nextManifest.evaluation.review_csv_path);
  }
  if (nextManifest.evaluation?.review_images_dir) {
    nextManifest.evaluation.review_images_dir = resolveSummaryArtifactPath(projectRoot, summaryDir, nextManifest.evaluation.review_images_dir);
  }

  return nextManifest;
}

function resolveReviewManifestPaths(projectRoot, summaryPath, reviewManifest) {
  if (!reviewManifest || typeof reviewManifest !== 'object') return reviewManifest;

  const summaryDir = summaryPath ? path.dirname(summaryPath) : projectRoot;
  const nextManifest = JSON.parse(JSON.stringify(reviewManifest));
  if (!Array.isArray(nextManifest.entries)) {
    return nextManifest;
  }

  nextManifest.entries = nextManifest.entries.map((entry) => ({
    ...entry,
    image_path: resolveSummaryArtifactPath(projectRoot, summaryDir, entry.image_path),
    review_image_path: resolveSummaryArtifactPath(projectRoot, summaryDir, entry.review_image_path),
  }));
  return nextManifest;
}

function createArtifactKey(relativePath) {
  return String(relativePath || '')
    .replace(/[\\/]/g, '_')
    .replace(/[^a-zA-Z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .toLowerCase();
}

function humanizeArtifactName(relativePath) {
  const baseName = path.basename(relativePath, path.extname(relativePath));
  const withSpaces = baseName
    .replace(/[_-]+/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .replace(/\bpr\b/gi, 'PR')
    .replace(/\bf1\b/gi, 'F1');

  return withSpaces.replace(/\b\w/g, (char) => char.toUpperCase());
}

function classifyImageArtifact(relativePath) {
  const normalized = String(relativePath || '').replace(/\\/g, '/').toLowerCase();
  if (normalized.includes('curve') || normalized.includes('confusion_matrix') || normalized === 'results.png') {
    return 'plot';
  }
  if (normalized.includes('batch') || normalized === 'labels.jpg') {
    return 'preview';
  }
  return 'image';
}

function listFilesRecursive(rootDir) {
  if (!rootDir || !fs.existsSync(rootDir)) return [];

  const results = [];
  const stack = [rootDir];
  while (stack.length > 0) {
    const currentDir = stack.pop();
    fs.readdirSync(currentDir, { withFileTypes: true }).forEach((entry) => {
      const entryPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        stack.push(entryPath);
      } else {
        results.push(entryPath);
      }
    });
  }

  return results.sort((left, right) => left.localeCompare(right, undefined, { numeric: true, sensitivity: 'base' }));
}

function resolveUltralyticsRunDir(projectRoot, summaryPath, summary) {
  if (summary?.ultralytics_run_dir && fs.existsSync(summary.ultralytics_run_dir)) {
    return summary.ultralytics_run_dir;
  }

  const summaryDir = summaryPath ? path.dirname(summaryPath) : projectRoot;
  const archiveRunDir = path.join(summaryDir, 'ultralytics_run');
  if (fs.existsSync(archiveRunDir)) {
    return archiveRunDir;
  }

  if (summary?.best_weights && fs.existsSync(summary.best_weights)) {
    const weightsDir = path.dirname(summary.best_weights);
    if (path.basename(weightsDir).toLowerCase() === 'weights') {
      const runDir = path.dirname(weightsDir);
      if (fs.existsSync(runDir)) {
        return runDir;
      }
    }
  }

  return null;
}

function mapExistingArtifactPath(targetPath, pathMapper) {
  if (!targetPath || !fs.existsSync(targetPath)) return null;
  return pathMapper(targetPath);
}

function collectRunArtifacts(ultralyticsRunDir, pathMapper) {
  const images = [];
  const tables = [];
  const weights = [];

  listFilesRecursive(ultralyticsRunDir).forEach((filePath) => {
    const relativePath = path.relative(ultralyticsRunDir, filePath);
    const ext = path.extname(filePath).toLowerCase();
    const baseEntry = {
      key: createArtifactKey(relativePath),
      label: humanizeArtifactName(relativePath),
      path: pathMapper(filePath),
      relative_path: relativePath,
    };

    if (IMAGE_ARTIFACT_EXTENSIONS.has(ext)) {
      images.push({
        ...baseEntry,
        category: classifyImageArtifact(relativePath),
      });
      return;
    }

    if (ext === '.pt') {
      weights.push(baseEntry);
      return;
    }

    if (['.csv', '.yaml', '.yml', '.json', '.txt'].includes(ext)) {
      tables.push({
        ...baseEntry,
        format: ext.slice(1).toUpperCase(),
      });
    }
  });

  return { images, tables, weights };
}

function buildArtifactManifestPayload({
  ultralyticsRunDir = null,
  stableBestModel = null,
  bestWeights = null,
  evaluation = {},
  pathMapper = (value) => value,
}) {
  const manifest = {
    generated_at: new Date().toISOString(),
    ultralytics_run_dir: mapExistingArtifactPath(ultralyticsRunDir, pathMapper),
    images: [],
    tables: [],
    weights: [],
    evaluation: {
      metrics_path: mapExistingArtifactPath(evaluation.metrics_path, pathMapper),
      review_manifest_path: mapExistingArtifactPath(evaluation.review_manifest_path, pathMapper),
      review_csv_path: mapExistingArtifactPath(evaluation.review_csv_path, pathMapper),
      review_images_dir: mapExistingArtifactPath(evaluation.review_images_dir, pathMapper),
    },
    counts: {
      imageCount: 0,
      plotCount: 0,
      previewCount: 0,
      tableCount: 0,
      weightCount: 0,
    },
  };

  if (ultralyticsRunDir && fs.existsSync(ultralyticsRunDir)) {
    const collected = collectRunArtifacts(ultralyticsRunDir, pathMapper);
    manifest.images = collected.images;
    manifest.tables = collected.tables;
    manifest.weights = collected.weights;
  }

  const stableBestEntry = mapExistingArtifactPath(stableBestModel, pathMapper);
  if (stableBestEntry) {
    manifest.weights.unshift({
      key: 'stable_best_model',
      label: 'Stable Best Model',
      path: stableBestEntry,
      relative_path: path.basename(stableBestModel),
    });
  }

  const bestWeightsEntry = mapExistingArtifactPath(bestWeights, pathMapper);
  if (bestWeightsEntry && bestWeightsEntry !== stableBestEntry) {
    manifest.weights.unshift({
      key: 'best_weights',
      label: 'Best Training Weights',
      path: bestWeightsEntry,
      relative_path: ultralyticsRunDir && bestWeights.startsWith(ultralyticsRunDir)
        ? path.relative(ultralyticsRunDir, bestWeights)
        : path.basename(bestWeights),
    });
  }

  manifest.counts.imageCount = manifest.images.length;
  manifest.counts.plotCount = manifest.images.filter((item) => item.category === 'plot').length;
  manifest.counts.previewCount = manifest.images.filter((item) => item.category === 'preview').length;
  manifest.counts.tableCount = manifest.tables.length;
  manifest.counts.weightCount = manifest.weights.length;
  return manifest;
}

function inferArtifactManifest(projectRoot, summaryPath, summary) {
  const ultralyticsRunDir = resolveUltralyticsRunDir(projectRoot, summaryPath, summary);
  return buildArtifactManifestPayload({
    ultralyticsRunDir,
    stableBestModel: summary?.stable_best_model || null,
    bestWeights: summary?.best_weights || null,
    evaluation: summary?.evaluation || {},
  });
}

function readActiveModelSelection(projectRoot) {
  return readJsonFileSafe(getActiveModelConfigPath(projectRoot));
}

function writeActiveModelSelection(projectRoot, payload) {
  const activeModelPath = getActiveModelConfigPath(projectRoot);
  fs.mkdirSync(path.dirname(activeModelPath), { recursive: true });
  fs.writeFileSync(activeModelPath, JSON.stringify(payload, null, 2), 'utf-8');
}

function resolveActiveModel(projectRoot) {
  const activeSelection = readActiveModelSelection(projectRoot);
  if (activeSelection && typeof activeSelection === 'object') {
    const summaryPath = resolveProjectPath(projectRoot, activeSelection.summary_path || activeSelection.summaryPath || null);
    const weightsPath = resolveProjectPath(projectRoot, activeSelection.weights_path || activeSelection.weightsPath || activeSelection.model_path || null);

    if (weightsPath && fs.existsSync(weightsPath)) {
      return {
        ...activeSelection,
        summaryPath: summaryPath && fs.existsSync(summaryPath) ? path.resolve(summaryPath) : null,
        weightsPath: path.resolve(weightsPath),
        activatedAt: activeSelection.activated_at || activeSelection.activatedAt || null,
      };
    }
  }

  const fallbackWeightsPath = path.join(getModelsRoot(projectRoot), 'best_model.pt');
  const fallbackSummaryPath = path.join(getModelsRoot(projectRoot), 'training_summary.json');
  if (!fs.existsSync(fallbackWeightsPath)) {
    return null;
  }

  return {
    source: 'current',
    summaryPath: fs.existsSync(fallbackSummaryPath) ? path.resolve(fallbackSummaryPath) : null,
    weightsPath: path.resolve(fallbackWeightsPath),
    activatedAt: null,
    fallback: true,
  };
}

function loadTrainingArtifactsFromSummaryPath(projectRoot, summaryPath) {
  const resolvedSummaryPath = path.resolve(summaryPath);
  const summary = readJsonFileSafe(resolvedSummaryPath);
  if (!summary) {
    return { error: `Training summary not found at ${resolvedSummaryPath}.` };
  }

  const resolvedSummary = resolveTrainingSummaryPaths(projectRoot, resolvedSummaryPath, summary);
  const reviewManifestPath = resolvedSummary?.evaluation?.review_manifest_path || null;
  const metricsPath = resolvedSummary?.evaluation?.metrics_path || null;
  const summaryDir = path.dirname(resolvedSummaryPath);
  const artifactManifestPath = resolvedSummary?.artifact_manifest_path || path.join(summaryDir, 'artifact_manifest.json');
  const artifactManifest = resolveArtifactManifestPaths(
    projectRoot,
    resolvedSummaryPath,
    readJsonFileSafe(artifactManifestPath) || inferArtifactManifest(projectRoot, resolvedSummaryPath, resolvedSummary),
  );
  const trainingResultsCsvPath = artifactManifest?.tables?.find((item) => item.key === 'results_csv')?.path || null;

  return {
    summary: resolvedSummary,
    summaryPath: resolvedSummaryPath,
    reviewManifest: resolveReviewManifestPaths(projectRoot, resolvedSummaryPath, readJsonFileSafe(reviewManifestPath)),
    testMetrics: readJsonFileSafe(metricsPath),
    artifactManifest,
    trainingResultsRows: readCsvFileSafe(trainingResultsCsvPath),
  };
}

function loadTrainingArtifacts(projectRoot, outputDir = 'models') {
  return loadTrainingArtifactsFromSummaryPath(projectRoot, path.join(projectRoot, outputDir, 'training_summary.json'));
}

function copyFileIfExists(sourcePath, destinationPath) {
  if (!sourcePath || !fs.existsSync(sourcePath)) return false;
  fs.mkdirSync(path.dirname(destinationPath), { recursive: true });
  fs.copyFileSync(sourcePath, destinationPath);
  return true;
}

function copyDirectoryRecursive(sourcePath, destinationPath) {
  if (!sourcePath || !fs.existsSync(sourcePath)) return false;

  const stats = fs.statSync(sourcePath);
  if (stats.isDirectory()) {
    fs.mkdirSync(destinationPath, { recursive: true });
    for (const entry of fs.readdirSync(sourcePath, { withFileTypes: true })) {
      copyDirectoryRecursive(path.join(sourcePath, entry.name), path.join(destinationPath, entry.name));
    }
    return true;
  }

  fs.mkdirSync(path.dirname(destinationPath), { recursive: true });
  fs.copyFileSync(sourcePath, destinationPath);
  return true;
}

function sanitizeRunSegment(value, fallback = 'run') {
  return String(value || fallback)
    .trim()
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, '_')
    .replace(/\s+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48) || fallback;
}

function formatRunTimestamp(value) {
  const date = value ? new Date(value) : new Date();
  if (Number.isNaN(date.getTime())) {
    return new Date().toISOString().replace(/[:.]/g, '-');
  }

  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, '0');
  const day = String(date.getUTCDate()).padStart(2, '0');
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  const seconds = String(date.getUTCSeconds()).padStart(2, '0');
  return `${year}${month}${day}T${hours}${minutes}${seconds}Z`;
}

function ensureUniqueDirectory(parentDir, preferredName) {
  let finalName = preferredName;
  let suffix = 2;
  let candidate = path.join(parentDir, finalName);

  while (fs.existsSync(candidate)) {
    finalName = `${preferredName}_${suffix}`;
    suffix += 1;
    candidate = path.join(parentDir, finalName);
  }

  fs.mkdirSync(candidate, { recursive: true });
  return candidate;
}

function buildTrainingRunDedupKey(summary) {
  return summary?.run_id
    || [
      summary?.created_at || '',
      summary?.model_choice || summary?.chosen_model || '',
      summary?.training_source?.dataset_name || '',
      summary?.epochs ?? '',
      summary?.batch_size ?? '',
      summary?.imgsz ?? '',
    ].join('|');
}

function activateTrainingRunSelection(projectRoot, artifacts, source = 'history') {
  if (!artifacts?.summary || !artifacts?.summaryPath) {
    throw new Error('Cannot activate a run without loaded training artifacts.');
  }

  const weightsPath = artifacts.summary?.stable_best_model;
  if (!weightsPath || !fs.existsSync(weightsPath)) {
    throw new Error(`Active model weights were not found at ${weightsPath || 'unknown path'}.`);
  }

  const payload = {
    activated_at: new Date().toISOString(),
    source,
    run_id: artifacts.summary?.run_id || null,
    summary_path: relativeProjectPath(projectRoot, artifacts.summaryPath),
    weights_path: relativeProjectPath(projectRoot, weightsPath),
    model_choice: artifacts.summary?.model_choice || null,
    dataset_name: artifacts.summary?.training_source?.dataset_name || null,
  };
  writeActiveModelSelection(projectRoot, payload);
  return resolveActiveModel(projectRoot);
}

function buildTrainingRunRecordFromArtifacts(artifacts, options = {}) {
  if (!artifacts?.summary || !artifacts?.summaryPath) return null;

  const {
    source = 'history',
    isActive = false,
    activatedAt = null,
  } = options;
  const summary = artifacts.summary;
  const metrics = summary?.evaluation?.metrics || artifacts.testMetrics?.aggregate_metrics || {};
  const reviewSummary = summary?.evaluation?.review_summary || artifacts.reviewManifest?.summary || {};
  const artifactCounts = artifacts?.artifactManifest?.counts || {};
  const modelChoice = summary?.model_choice || null;
  const fallbackModelLabel = summary?.chosen_model
    ? path.parse(summary.chosen_model).name
    : (summary?.stable_best_model ? path.parse(summary.stable_best_model).name : 'saved-model');

  return {
    id: `${source}:${summary?.run_id || artifacts.summaryPath}`,
    source,
    runId: summary?.run_id || null,
    summaryPath: artifacts.summaryPath,
    createdAt: summary?.created_at || (fs.existsSync(artifacts.summaryPath) ? fs.statSync(artifacts.summaryPath).mtime.toISOString() : null),
    datasetName: summary?.training_source?.dataset_name || 'Unknown dataset',
    csvName: summary?.training_source?.csv_name || null,
    modelChoice,
    modelLabel: modelChoice || fallbackModelLabel,
    modelSource: summary?.chosen_model || null,
    stableBestModel: summary?.stable_best_model || null,
    epochs: Number(summary?.epochs ?? 0) || null,
    batchSize: Number(summary?.batch_size ?? 0) || null,
    imageSize: Number(summary?.imgsz ?? 0) || null,
    requestedDeviceMode: summary?.requested_device_mode || null,
    device: summary?.device || null,
    isActive,
    activatedAt,
    reviewConfidenceThreshold: summary?.evaluation?.confidence_threshold ?? null,
    metrics: {
      precision: typeof metrics?.precision === 'number' ? metrics.precision : null,
      recall: typeof metrics?.recall === 'number' ? metrics.recall : null,
      map50: typeof metrics?.map50 === 'number' ? metrics.map50 : null,
      map75: typeof metrics?.map75 === 'number' ? metrics.map75 : null,
      map50_95: typeof metrics?.map50_95 === 'number' ? metrics.map50_95 : null,
      fitness: typeof metrics?.fitness === 'number' ? metrics.fitness : null,
    },
    perClass: Array.isArray(metrics?.per_class) ? metrics.per_class : [],
    resultsDict: metrics?.results_dict && typeof metrics.results_dict === 'object' ? metrics.results_dict : {},
    reviewSummary: {
      images: Number(reviewSummary?.images ?? 0) || 0,
      matched_boxes: Number(reviewSummary?.matched_boxes ?? 0) || 0,
      false_positive_boxes: Number(reviewSummary?.false_positive_boxes ?? 0) || 0,
      missed_ground_truth_boxes: Number(reviewSummary?.missed_ground_truth_boxes ?? 0) || 0,
      status_counts: reviewSummary?.status_counts && typeof reviewSummary.status_counts === 'object'
        ? reviewSummary.status_counts
        : {},
    },
    artifactSummary: {
      imageCount: Number(artifactCounts.imageCount ?? 0) || 0,
      plotCount: Number(artifactCounts.plotCount ?? 0) || 0,
      previewCount: Number(artifactCounts.previewCount ?? 0) || 0,
      tableCount: Number(artifactCounts.tableCount ?? 0) || 0,
      weightCount: Number(artifactCounts.weightCount ?? 0) || 0,
    },
  };
}

function archiveTrainingArtifacts(projectRoot, artifacts) {
  if (!artifacts?.summary || !artifacts?.summaryPath) {
    throw new Error('Cannot archive a training run without loaded artifacts.');
  }

  const stableBestModel = artifacts.summary?.stable_best_model;
  if (!stableBestModel || !fs.existsSync(stableBestModel)) {
    throw new Error(`Stable best model not found at ${stableBestModel || 'unknown path'}.`);
  }

  const historyRoot = getModelHistoryRoot(projectRoot);
  fs.mkdirSync(historyRoot, { recursive: true });

  const archiveName = [
    formatRunTimestamp(artifacts.summary?.created_at),
    sanitizeRunSegment(artifacts.summary?.training_source?.dataset_name, 'dataset'),
    sanitizeRunSegment(artifacts.summary?.model_choice || path.parse(stableBestModel).name, 'model'),
  ].join('_');
  const archiveDir = ensureUniqueDirectory(historyRoot, archiveName);
  const archiveEvaluationDir = path.join(archiveDir, 'test_evaluation');
  const archiveUltralyticsRunDir = path.join(archiveDir, 'ultralytics_run');
  const sourceUltralyticsRunDir = resolveUltralyticsRunDir(projectRoot, artifacts.summaryPath, artifacts.summary);

  copyFileIfExists(stableBestModel, path.join(archiveDir, 'best_model.pt'));
  if (sourceUltralyticsRunDir) {
    copyDirectoryRecursive(sourceUltralyticsRunDir, archiveUltralyticsRunDir);
  }

  if (artifacts.summary?.evaluation?.metrics_path) {
    copyFileIfExists(artifacts.summary.evaluation.metrics_path, path.join(archiveEvaluationDir, 'test_metrics.json'));
  }
  if (artifacts.summary?.evaluation?.review_csv_path) {
    copyFileIfExists(artifacts.summary.evaluation.review_csv_path, path.join(archiveEvaluationDir, 'test_predictions.csv'));
  }
  if (artifacts.summary?.evaluation?.review_images_dir) {
    copyDirectoryRecursive(
      artifacts.summary.evaluation.review_images_dir,
      path.join(archiveEvaluationDir, 'test_review_images'),
    );
  }

  if (artifacts.reviewManifest) {
    const archivedManifest = JSON.parse(JSON.stringify(artifacts.reviewManifest));
    if (Array.isArray(archivedManifest.entries)) {
      archivedManifest.entries = archivedManifest.entries.map((entry) => ({
        ...entry,
        review_image_path: entry.review_image_path
          ? path.join('test_evaluation', 'test_review_images', path.basename(entry.review_image_path))
          : entry.review_image_path,
      }));
    }

    fs.mkdirSync(archiveEvaluationDir, { recursive: true });
    fs.writeFileSync(
      path.join(archiveEvaluationDir, 'test_review_manifest.json'),
      JSON.stringify(archivedManifest, null, 2),
      'utf-8',
    );
  } else if (artifacts.summary?.evaluation?.review_manifest_path) {
    copyFileIfExists(
      artifacts.summary.evaluation.review_manifest_path,
      path.join(archiveEvaluationDir, 'test_review_manifest.json'),
    );
  }

  const archivedSummary = JSON.parse(JSON.stringify(artifacts.summary));
  archivedSummary.best_weights = fs.existsSync(path.join(archiveUltralyticsRunDir, 'weights', 'best.pt'))
    ? path.join('ultralytics_run', 'weights', 'best.pt')
    : 'best_model.pt';
  archivedSummary.stable_best_model = 'best_model.pt';
  if (fs.existsSync(archiveUltralyticsRunDir)) {
    archivedSummary.ultralytics_run_dir = 'ultralytics_run';
  } else {
    delete archivedSummary.ultralytics_run_dir;
  }
  archivedSummary.archive = {
    archived_at: new Date().toISOString(),
    source_summary_path: path.relative(projectRoot, artifacts.summaryPath),
    source: 'history',
  };

  if (archivedSummary.evaluation) {
    if (fs.existsSync(path.join(archiveEvaluationDir, 'test_metrics.json'))) {
      archivedSummary.evaluation.metrics_path = path.join('test_evaluation', 'test_metrics.json');
    }
    if (fs.existsSync(path.join(archiveEvaluationDir, 'test_review_manifest.json'))) {
      archivedSummary.evaluation.review_manifest_path = path.join('test_evaluation', 'test_review_manifest.json');
    }
    if (fs.existsSync(path.join(archiveEvaluationDir, 'test_predictions.csv'))) {
      archivedSummary.evaluation.review_csv_path = path.join('test_evaluation', 'test_predictions.csv');
    }
    if (fs.existsSync(path.join(archiveEvaluationDir, 'test_review_images'))) {
      archivedSummary.evaluation.review_images_dir = path.join('test_evaluation', 'test_review_images');
    }
  }

  const archiveArtifactManifestPath = path.join(archiveDir, 'artifact_manifest.json');
  const artifactManifest = buildArtifactManifestPayload({
    ultralyticsRunDir: fs.existsSync(archiveUltralyticsRunDir) ? archiveUltralyticsRunDir : null,
    stableBestModel: path.join(archiveDir, 'best_model.pt'),
    bestWeights: fs.existsSync(path.join(archiveUltralyticsRunDir, 'weights', 'best.pt'))
      ? path.join(archiveUltralyticsRunDir, 'weights', 'best.pt')
      : path.join(archiveDir, 'best_model.pt'),
    evaluation: {
      metrics_path: path.join(archiveEvaluationDir, 'test_metrics.json'),
      review_manifest_path: path.join(archiveEvaluationDir, 'test_review_manifest.json'),
      review_csv_path: path.join(archiveEvaluationDir, 'test_predictions.csv'),
      review_images_dir: path.join(archiveEvaluationDir, 'test_review_images'),
    },
    pathMapper: (targetPath) => path.relative(archiveDir, targetPath),
  });
  fs.writeFileSync(archiveArtifactManifestPath, JSON.stringify(artifactManifest, null, 2), 'utf-8');
  archivedSummary.artifact_manifest_path = 'artifact_manifest.json';

  const archiveSummaryPath = path.join(archiveDir, 'training_summary.json');
  fs.writeFileSync(archiveSummaryPath, JSON.stringify(archivedSummary, null, 2), 'utf-8');
  return loadTrainingArtifactsFromSummaryPath(projectRoot, archiveSummaryPath);
}

function listTrainingRuns(projectRoot) {
  const runs = [];
  const seen = new Set();
  const activeModel = resolveActiveModel(projectRoot);
  const activeSummaryPath = activeModel?.summaryPath ? path.resolve(activeModel.summaryPath) : null;
  let activeDedupKey = null;
  if (activeSummaryPath && fs.existsSync(activeSummaryPath)) {
    const activeArtifacts = loadTrainingArtifactsFromSummaryPath(projectRoot, activeSummaryPath);
    if (!activeArtifacts.error) {
      activeDedupKey = buildTrainingRunDedupKey(activeArtifacts.summary);
    }
  }
  const historyRoot = getModelHistoryRoot(projectRoot);

  const appendRun = (artifacts, source) => {
    if (artifacts.error) return;

    const dedupKey = buildTrainingRunDedupKey(artifacts.summary);
    if (seen.has(dedupKey)) return;
    seen.add(dedupKey);

    const isActive = (activeSummaryPath && path.resolve(artifacts.summaryPath) === activeSummaryPath)
      || (activeDedupKey && dedupKey === activeDedupKey);
    const record = buildTrainingRunRecordFromArtifacts(artifacts, {
      source,
      isActive,
      activatedAt: isActive ? activeModel?.activatedAt || null : null,
    });
    if (record) runs.push(record);
  };

  if (fs.existsSync(historyRoot)) {
    const historyDirs = fs.readdirSync(historyRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => path.join(historyRoot, entry.name));

    historyDirs.forEach((dirPath) => {
      appendRun(loadTrainingArtifactsFromSummaryPath(projectRoot, path.join(dirPath, 'training_summary.json')), 'history');
    });
  }

  const currentArtifacts = loadTrainingArtifacts(projectRoot);
  appendRun(currentArtifacts, 'current');

  return runs.sort((left, right) => {
    const leftTime = left?.createdAt ? new Date(left.createdAt).getTime() : 0;
    const rightTime = right?.createdAt ? new Date(right.createdAt).getTime() : 0;
    return rightTime - leftTime;
  });
}

function isPathInsideDirectory(parentDir, targetPath) {
  const relativePath = path.relative(path.resolve(parentDir), path.resolve(targetPath));
  return relativePath === '' || (!relativePath.startsWith('..') && !path.isAbsolute(relativePath));
}

function getDatasetImagesDir(datasetPath) {
  return path.join(datasetPath, 'images');
}

function getDatasetLabelsDir(datasetPath) {
  return path.join(datasetPath, 'labels');
}

function summarizeDatasetDirectory(datasetPath) {
  const imagesDir = getDatasetImagesDir(datasetPath);
  const labelsDir = getDatasetLabelsDir(datasetPath);
  const imageCount = fs.existsSync(imagesDir)
    ? fs.readdirSync(imagesDir).filter((entry) => IMAGE_EXTENSIONS.has(path.extname(entry).toLowerCase())).length
    : 0;
  const labelCount = fs.existsSync(labelsDir)
    ? fs.readdirSync(labelsDir).filter((entry) => path.extname(entry).toLowerCase() === '.txt').length
    : 0;

  return {
    imagesDir,
    labelsDir,
    imageCount,
    labelCount,
    hasImages: imageCount > 0,
    hasLabels: labelCount > 0,
  };
}

function getDatasetLabelPath(datasetPath, filename) {
  return path.join(getDatasetLabelsDir(datasetPath), `${getImageBasename(filename)}.txt`);
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

function ensureUniqueFilename(filename, usedNames) {
  const parsed = path.parse(filename);
  const safeBase = sanitizeDatasetName(parsed.name || 'image');
  const safeExt = parsed.ext || '.png';
  let candidate = `${safeBase}${safeExt}`;
  let counter = 2;

  while (usedNames.has(candidate.toLowerCase())) {
    candidate = `${safeBase}_${counter}${safeExt}`;
    counter += 1;
  }

  usedNames.add(candidate.toLowerCase());
  return candidate;
}

function cloneAnnotationRow(row) {
  const cloned = {};
  Object.entries(row || {}).forEach(([key, value]) => {
    cloned[key] = value ?? '';
  });
  return cloned;
}

function normalizeMergedSources(payloadSources) {
  return (Array.isArray(payloadSources) ? payloadSources : [])
    .filter(Boolean)
    .map((source) => ({
      datasetName: String(source.datasetName || '').trim(),
      datasetPath: String(source.datasetPath || '').trim(),
      csvName: String(source.csvName || '').trim(),
    }))
    .filter((source) => source.datasetName && source.datasetPath && source.csvName);
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

ipcMain.handle('select-video', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile'],
    filters: [{ name: 'Videos', extensions: VIDEO_EXTENSIONS }]
  });
  if (canceled) {
    return null;
  }
  return filePaths[0];
});

ipcMain.handle('run-prediction', async (event, imagePath) => {
  return new Promise((resolve, reject) => {
    const projectRoot = path.join(__dirname, '..');
    const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(projectRoot, 'src', 'predict_cli.py');
    const activeModel = resolveActiveModel(projectRoot);

    // Get downloads path and create a unique filename
    const downloadsPath = app.getPath('downloads');
    const timestamp = new Date().getTime();
    const savePath = path.join(downloadsPath, `prediction_${timestamp}.png`);

    const pythonArgs = [scriptPath, imagePath, '--save_path', savePath];
    if (activeModel?.weightsPath) {
      pythonArgs.push('--model', activeModel.weightsPath);
    }

    const pythonProcess = spawn(pythonExe, pythonArgs, {
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

ipcMain.handle('start-video-prediction', async (event, payload = {}) => {
  return new Promise((resolve) => {
    if (videoPredictionProcess) {
      resolve({ error: 'A video prediction session is already running. Stop it before starting another.' });
      return;
    }

    const projectRoot = path.join(__dirname, '..');
    const pythonExe = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(projectRoot, 'src', 'predict_video_cli.py');
    const videoPath = String(payload.videoPath || '').trim();
    const mode = String(payload.mode || 'precompute').trim().toLowerCase();
    const activeModel = resolveActiveModel(projectRoot);

    if (!videoPath) {
      resolve({ error: 'No video path was provided.' });
      return;
    }
    if (!['precompute', 'stream'].includes(mode)) {
      resolve({ error: `Unsupported video inference mode '${mode}'.` });
      return;
    }
    if (!fs.existsSync(videoPath)) {
      resolve({ error: `Video file not found: ${videoPath}` });
      return;
    }
    if (!fs.existsSync(pythonExe)) {
      resolve({ error: `Python environment not found at ${pythonExe}. Run setup_venv.ps1 first.` });
      return;
    }
    if (!fs.existsSync(scriptPath)) {
      resolve({ error: `Video prediction script not found at ${scriptPath}.` });
      return;
    }

    const sessionId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const pythonArgs = [scriptPath, videoPath, '--mode', mode];
    if (activeModel?.weightsPath) {
      pythonArgs.push('--model', activeModel.weightsPath);
    }

    const child = spawn(pythonExe, pythonArgs, {
      cwd: projectRoot,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }
    });

    videoPredictionProcess = child;
    videoPredictionSessionId = sessionId;
    videoPredictionStopRequested = false;

    let resolved = false;
    let stdoutBuffer = '';
    let lastLoggedProcessed = 0;

    const resolveOnce = (value) => {
      if (!resolved) {
        resolved = true;
        resolve(value);
      }
    };

    const handleVideoEvent = (eventPayload) => {
      const forwarded = { ...eventPayload, session_id: sessionId };
      sendVideoPredictionEvent(forwarded);

      if (eventPayload.type === 'started') {
        sendPipelineOutput(
          'stdout',
          `[Video] Started ${path.basename(String(eventPayload.video_path || videoPath))} | ${Number(eventPayload.frame_count || 0)} frame(s) @ ${Number(eventPayload.fps || 0).toFixed(2)} FPS on ${eventPayload.device_name || eventPayload.device || 'unknown'}\n`,
        );
      } else if (eventPayload.type === 'progress') {
        const processed = Number(eventPayload.processed_frames || 0);
        const total = Number(eventPayload.total_frames || 0);
        if (processed === 1 || total === processed || processed - lastLoggedProcessed >= 30) {
          lastLoggedProcessed = processed;
          sendPipelineOutput('stdout', `[Video] Processed ${processed}/${total || '?'} frames.\n`);
        }
      } else if (eventPayload.type === 'complete') {
        sendPipelineOutput('stdout', `[Video] Prediction complete. Processed ${Number(eventPayload.processed_frames || 0)} frame(s).\n`);
      } else if (eventPayload.type === 'error') {
        sendPipelineOutput('stderr', `[Video][Error] ${eventPayload.message || 'Unknown video prediction error.'}\n`);
      }
    };

    const flushStdoutBuffer = (includeTrailing = false) => {
      const lines = stdoutBuffer.split(/\r?\n/);
      stdoutBuffer = includeTrailing ? '' : (lines.pop() || '');
      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) continue;
        try {
          handleVideoEvent(JSON.parse(line));
        } catch {
          sendPipelineOutput('stdout', `${line}\n`);
        }
      }
    };

    child.on('spawn', () => {
      resolveOnce({ success: true, session_id: sessionId });
    });

    child.stdout.on('data', (data) => {
      stdoutBuffer += data.toString();
      flushStdoutBuffer();
    });

    child.stderr.on('data', (data) => {
      sendPipelineOutput('stderr', data.toString());
    });

    child.on('error', (error) => {
      sendVideoPredictionEvent({ type: 'error', session_id: sessionId, message: error.message });
      sendPipelineOutput('stderr', `[Video][Error] ${error.message}\n`);
      cleanupVideoPredictionState();
      resolveOnce({ error: error.message });
    });

    child.on('close', (code) => {
      flushStdoutBuffer(true);
      const closingSessionId = videoPredictionSessionId || sessionId;
      const stoppedByUser = videoPredictionStopRequested;

      if (stoppedByUser) {
        sendVideoPredictionEvent({ type: 'stopped', session_id: closingSessionId });
        sendPipelineOutput('stdout', '[Video] Prediction session stopped.\n');
      } else if (code !== 0) {
        const message = `Video prediction failed with code ${code}`;
        sendVideoPredictionEvent({ type: 'error', session_id: closingSessionId, message });
        sendPipelineOutput('stderr', `[Video][Error] ${message}\n`);
        resolveOnce({ error: message });
      }

      cleanupVideoPredictionState();
      resolveOnce({ success: true, session_id: sessionId });
    });
  });
});

ipcMain.handle('stop-video-prediction', async () => {
  if (!videoPredictionProcess) {
    return { success: true };
  }

  videoPredictionStopRequested = true;
  spawn('taskkill', ['/pid', String(videoPredictionProcess.pid), '/f', '/t']);
  return { success: true };
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
                    const datasetSummary = summarizeDatasetDirectory(itemPath);
                    results.push({
                        name: item.name,
                        path: itemPath,
                        csvs: csvFiles,
                        ...datasetSummary,
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
                            const datasetSummary = summarizeDatasetDirectory(sdPath);
                            results.push({
                                name: `${item.name}/${sd.name}`,
                                path: sdPath,
                                csvs: sdCsvs,
                                ...datasetSummary,
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

ipcMain.handle('merge-datasets', async (event, payload = {}) => {
    try {
        const projectRoot = path.join(__dirname, '..');
        const sources = normalizeMergedSources(payload.sources);
        if (sources.length < 2) {
            return { error: 'Choose at least two datasets to merge.' };
        }

        const outputDatasetName = sanitizeDatasetName(payload.outputDatasetName || `merged_${new Date().toISOString().split('T')[0]}`);
        const outputDatasetPath = path.join(projectRoot, 'data_sets', outputDatasetName);
        const outputImagesDir = path.join(outputDatasetPath, 'images');
        const outputLabelsDir = path.join(outputDatasetPath, 'labels');
        const outputCsvName = 'labels_merged.csv';
        const outputCsvPath = path.join(outputDatasetPath, outputCsvName);

        if (fs.existsSync(outputDatasetPath)) {
            const existingEntries = fs.readdirSync(outputDatasetPath);
            if (existingEntries.length > 0) {
                return { error: `Output dataset already exists and is not empty: ${outputDatasetName}` };
            }
        }

        fs.mkdirSync(outputImagesDir, { recursive: true });
        fs.mkdirSync(outputLabelsDir, { recursive: true });

        const mergedRows = [];
        const usedOutputNames = new Set();
        const mergeSummary = [];

        for (const source of sources) {
            const csvPath = path.join(source.datasetPath, source.csvName);
            if (!fs.existsSync(csvPath)) {
                return { error: `CSV not found for ${source.datasetName}: ${source.csvName}` };
            }

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

            let copiedImages = 0;
            let copiedAnnotations = 0;

            for (const [filename, filenameRows] of rowsByFilename.entries()) {
                const imagePath = findDatasetImagePath(source.datasetPath, filename);
                if (!imagePath || !fs.existsSync(imagePath)) {
                    continue;
                }

                const safeSourceName = sanitizeDatasetName(source.datasetName);
                const outputFilename = ensureUniqueFilename(`${safeSourceName}_${path.basename(filename)}`, usedOutputNames);
                const outputImagePath = path.join(outputImagesDir, outputFilename);
                const outputLabelPath = path.join(outputLabelsDir, `${getImageBasename(outputFilename)}.txt`);
                const sourceLabelPath = getDatasetLabelPath(source.datasetPath, filename);

                fs.copyFileSync(imagePath, outputImagePath);

                if (fs.existsSync(sourceLabelPath)) {
                    fs.copyFileSync(sourceLabelPath, outputLabelPath);
                } else {
                    writeYoloLabelFile(outputLabelPath, rowsToBoxes(filenameRows));
                }

                filenameRows.forEach((row) => {
                    const rewritten = cloneAnnotationRow(row);
                    rewritten.filename = outputFilename;
                    rewritten.video_id = `${safeSourceName}__${String(row.video_id || path.parse(filename).name)}`;
                    mergedRows.push(rewritten);
                    copiedAnnotations += 1;
                });

                copiedImages += 1;
            }

            mergeSummary.push({
                datasetName: source.datasetName,
                csvName: source.csvName,
                images: copiedImages,
                annotations: copiedAnnotations,
            });
        }

        if (mergedRows.length === 0) {
            return { error: 'No labeled images were merged. Check that the selected datasets contain images and CSV rows.' };
        }

        writeAnnotationRows(outputCsvPath, mergedRows);
        fs.writeFileSync(
            path.join(outputDatasetPath, 'merge_manifest.json'),
            JSON.stringify({
                outputDatasetName,
                outputCsvName,
                createdAt: new Date().toISOString(),
                sources: mergeSummary,
                totals: {
                    images: mergeSummary.reduce((sum, item) => sum + item.images, 0),
                    annotations: mergeSummary.reduce((sum, item) => sum + item.annotations, 0),
                },
            }, null, 2),
            'utf-8'
        );

        return {
            success: true,
            datasetName: outputDatasetName,
            datasetPath: outputDatasetPath,
            csvName: outputCsvName,
            summary: mergeSummary,
            totals: {
                images: mergeSummary.reduce((sum, item) => sum + item.images, 0),
                annotations: mergeSummary.reduce((sum, item) => sum + item.annotations, 0),
            },
        };
    } catch (error) {
        return { error: error.message };
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
            testSplit = 20,
            confidenceThreshold = 25,
        } = payload;

        if (!datasetPath || !csvName) {
            resolve({ error: 'Training requires both a dataset path and CSV name.' });
            return;
        }

        const csvPath = path.join(datasetPath, csvName);
        const imgDir = path.join(datasetPath, 'images');
        const normalizedTestSplit = Math.max(1, Math.min(90, Number(testSplit) || 20)) / 100;
        const normalizedConfidenceThreshold = Math.max(0, Math.min(100, Number(confidenceThreshold) || 25)) / 100;

        if (!fs.existsSync(csvPath)) {
            resolve({ error: `Training CSV not found: ${csvPath}` });
            return;
        }
        if (!fs.existsSync(imgDir)) {
            resolve({
                error: `This dataset is incomplete: the image folder is missing at ${imgDir}. Rebuild or re-export the dataset so it contains both images/ and labels/ before training.`,
            });
            return;
        }

        const imageCount = fs.readdirSync(imgDir).filter((entry) => IMAGE_EXTENSIONS.has(path.extname(entry).toLowerCase())).length;
        if (imageCount === 0) {
            resolve({
                error: `This dataset is incomplete: ${imgDir} exists but contains no images. Rebuild or re-export the dataset before training.`,
            });
            return;
        }

        const trainArgs = [
            trainScript, 
            '--train_dir', imgDir, 
            '--train_csv', csvPath,
            '--epochs', epochs.toString(),
            '--batch_size', batchSize.toString(),
            '--imgsz', imageSize.toString(),
            '--device_mode', deviceMode,
            '--model', modelChoice,
            '--test_split', normalizedTestSplit.toString(),
            '--review_confidence_threshold', normalizedConfidenceThreshold.toString(),
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
            if (code !== 0) {
                resolve({ error: `Training failed with code ${code}` });
                return;
            }

            const artifacts = loadTrainingArtifacts(projectRoot);
            if (artifacts.error) {
                resolve({ error: artifacts.error });
                return;
            }

            let resultArtifacts = artifacts;
            let historyWarning = null;
            let activationWarning = null;
            try {
                resultArtifacts = archiveTrainingArtifacts(projectRoot, artifacts);
            } catch (archiveError) {
                historyWarning = `Training finished, but saving the model history snapshot failed: ${archiveError.message}`;
            }

            let activeModel = null;
            try {
                const activationSource = resultArtifacts.summaryPath.includes(`${path.sep}${MODEL_HISTORY_DIRNAME}${path.sep}`) ? 'history' : 'current';
                activeModel = activateTrainingRunSelection(projectRoot, resultArtifacts, activationSource);
            } catch (activationError) {
                activationWarning = `Training finished, but activating the saved model failed: ${activationError.message}`;
            }

            resolve({
                success: true,
                summary: resultArtifacts.summary,
                summaryPath: resultArtifacts.summaryPath,
                reviewManifest: resultArtifacts.reviewManifest,
                testMetrics: resultArtifacts.testMetrics,
                artifactManifest: resultArtifacts.artifactManifest,
                trainingResultsRows: resultArtifacts.trainingResultsRows,
                savedRun: buildTrainingRunRecordFromArtifacts(
                    resultArtifacts,
                    {
                        source: resultArtifacts.summaryPath.includes(`${path.sep}${MODEL_HISTORY_DIRNAME}${path.sep}`) ? 'history' : 'current',
                        isActive: Boolean(activeModel),
                        activatedAt: activeModel?.activatedAt || null,
                    },
                ),
                historyWarning,
                activationWarning,
                activeModel,
            });
        });
    });
});

ipcMain.handle('list-training-runs', async () => {
    try {
        const projectRoot = path.join(__dirname, '..');
        return listTrainingRuns(projectRoot);
    } catch (error) {
        return [];
    }
});

ipcMain.handle('load-training-run', async (event, summaryPath) => {
    try {
        const projectRoot = path.join(__dirname, '..');
        const modelsRoot = getModelsRoot(projectRoot);
        const resolvedSummaryPath = path.resolve(String(summaryPath || ''));

        if (!resolvedSummaryPath) {
            return { error: 'A saved training run path is required.' };
        }
        if (!isPathInsideDirectory(modelsRoot, resolvedSummaryPath)) {
            return { error: 'Saved training runs can only be loaded from the models directory.' };
        }

        const artifacts = loadTrainingArtifactsFromSummaryPath(projectRoot, resolvedSummaryPath);
        if (artifacts.error) {
            return { error: artifacts.error };
        }

        const activeModel = resolveActiveModel(projectRoot);
        let activeDedupKey = null;
        if (activeModel?.summaryPath && fs.existsSync(activeModel.summaryPath)) {
            const activeArtifacts = loadTrainingArtifactsFromSummaryPath(projectRoot, activeModel.summaryPath);
            if (!activeArtifacts.error) {
                activeDedupKey = buildTrainingRunDedupKey(activeArtifacts.summary);
            }
        }
        const isActive = (activeModel?.summaryPath && path.resolve(artifacts.summaryPath) === path.resolve(activeModel.summaryPath))
            || (activeDedupKey && buildTrainingRunDedupKey(artifacts.summary) === activeDedupKey);

        return {
            success: true,
            summary: artifacts.summary,
            summaryPath: artifacts.summaryPath,
            reviewManifest: artifacts.reviewManifest,
            testMetrics: artifacts.testMetrics,
            artifactManifest: artifacts.artifactManifest,
            trainingResultsRows: artifacts.trainingResultsRows,
            savedRun: buildTrainingRunRecordFromArtifacts(
                artifacts,
                {
                    source: resolvedSummaryPath.includes(`${path.sep}${MODEL_HISTORY_DIRNAME}${path.sep}`) ? 'history' : 'current',
                    isActive,
                    activatedAt: isActive ? activeModel?.activatedAt || null : null,
                },
            ),
            activeModel,
        };
    } catch (error) {
        return { error: error.message };
    }
});

ipcMain.handle('activate-training-run', async (event, summaryPath) => {
    try {
        const projectRoot = path.join(__dirname, '..');
        const modelsRoot = getModelsRoot(projectRoot);
        const resolvedSummaryPath = path.resolve(String(summaryPath || ''));

        if (!resolvedSummaryPath) {
            return { error: 'A saved training run path is required.' };
        }
        if (!isPathInsideDirectory(modelsRoot, resolvedSummaryPath)) {
            return { error: 'Saved training runs can only be activated from the models directory.' };
        }

        const artifacts = loadTrainingArtifactsFromSummaryPath(projectRoot, resolvedSummaryPath);
        if (artifacts.error) {
            return { error: artifacts.error };
        }

        const activeModel = activateTrainingRunSelection(
            projectRoot,
            artifacts,
            resolvedSummaryPath.includes(`${path.sep}${MODEL_HISTORY_DIRNAME}${path.sep}`) ? 'history' : 'current',
        );

        return {
            success: true,
            summary: artifacts.summary,
            summaryPath: artifacts.summaryPath,
            reviewManifest: artifacts.reviewManifest,
            testMetrics: artifacts.testMetrics,
            artifactManifest: artifacts.artifactManifest,
            trainingResultsRows: artifacts.trainingResultsRows,
            savedRun: buildTrainingRunRecordFromArtifacts(
                artifacts,
                {
                    source: resolvedSummaryPath.includes(`${path.sep}${MODEL_HISTORY_DIRNAME}${path.sep}`) ? 'history' : 'current',
                    isActive: true,
                    activatedAt: activeModel?.activatedAt || null,
                },
            ),
            activeModel,
        };
    } catch (error) {
        return { error: error.message };
    }
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
