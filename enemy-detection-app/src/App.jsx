import { useState, useRef, useEffect } from 'react';
import { X, Minus, Square, Copy, Target, Gamepad2, MonitorPlay, Database, BrainCircuit, BarChart3, ChevronRight, ChevronLeft, ScanEye, Terminal, Layers, Cpu, CheckCircle, CloudDownload, Trash2, Image as ImageIcon } from 'lucide-react';
import './index.css';
import CollectorWorkspace from './CollectorWorkspace';
import DatasetViewerWorkspace from './DatasetViewerWorkspace';
import TrainingReviewPanel from './TrainingReviewPanel';
import VideoTesterWorkspace from './VideoTesterWorkspace';

const TitleBar = () => (
    <div className="h-8 bg-stone-950 border-b border-rose-950 flex items-center justify-between px-4 drag select-none">
        <div className="flex items-center gap-2">
            <ScanEye size={16} className="text-rose-500" />
            <span className="text-xs font-bold text-stone-300 uppercase tracking-widest">Enemy Detection Hub</span>
        </div>
        <div className="flex items-center no-drag">
            <button 
                onClick={() => window.electronAPI.minimizeWindow()}
                className="p-2 hover:bg-stone-900 text-stone-400 transition"
            >
                <Minus size={14} />
            </button>
            <button 
                onClick={() => window.electronAPI.maximizeWindow()}
                className="p-2 hover:bg-stone-900 text-stone-400 transition"
            >
                <Square size={12} />
            </button>
            <button 
                onClick={() => window.electronAPI.closeWindow()}
                className="p-2 hover:bg-rose-600 hover:text-white text-slate-400 transition"
            >
                <X size={14} />
            </button>
        </div>
    </div>
);

const DETECTOR_CHOICES = [
    {
        key: 'yolov8n',
        label: 'YOLOv8n',
        summary: 'Fastest option for quick iteration and weaker GPUs.',
    },
    {
        key: 'yolov8s',
        label: 'YOLOv8s',
        summary: 'Better small-target recall with a moderate speed cost.',
    },
    {
        key: 'yolov8m',
        label: 'YOLOv8m',
        summary: 'Stronger capacity for harder scenes if you have the VRAM.',
    },
    {
        key: 'rtdetr-l',
        label: 'RT-DETR-L',
        summary: 'Heavier transformer detector for stronger baseline comparisons.',
    },
];

const formatPercent = (value) => (
    typeof value === 'number' && !Number.isNaN(value) ? `${(value * 100).toFixed(1)}%` : 'n/a'
);

const formatTrainingTimestamp = (value) => {
    if (!value) return 'Unknown date';
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return 'Unknown date';
    return parsed.toLocaleString();
};

const getDetectorChoiceLabel = (modelChoice, fallbackLabel = 'Custom model') => (
    DETECTOR_CHOICES.find((choice) => choice.key === modelChoice)?.label
    || modelChoice
    || fallbackLabel
);

const getDetectionChrome = (className) => {
    const normalized = String(className || '').toLowerCase();
    if (normalized === 'player') {
        return {
            border: 'border-amber-400/90',
            fill: 'bg-amber-500/10',
            glow: 'shadow-[0_0_20px_rgba(251,191,36,0.35)]',
            badge: 'bg-amber-500/90',
        };
    }
    return {
        border: 'border-rose-500/90',
        fill: 'bg-rose-500/10',
        glow: 'shadow-[0_0_20px_rgba(244,63,94,0.35)]',
        badge: 'bg-rose-500/90',
    };
};

const createEmptyPredictionSummary = () => ({
    count: 0,
    topDetection: null,
    modelPath: null,
    savedImagePath: null,
    classSummary: {
        enemy: 0,
        player: 0,
        unknown: 0,
    },
});

const resolveDetectionClassKey = (detection) => {
    const normalized = String(detection?.class_key || detection?.class_name || '').trim().toLowerCase();
    if (normalized.includes('player')) return 'player';
    if (normalized.includes('enemy')) return 'enemy';
    return normalized || 'unknown';
};

const formatDetectionClassName = (detection) => {
    const classKey = resolveDetectionClassKey(detection);
    if (classKey === 'player') return 'Player';
    if (classKey === 'enemy') return 'Enemy';
    return detection?.class_name || 'Unknown';
};

const summarizeDetections = (detections) => {
    const summary = {
        enemy: 0,
        player: 0,
        unknown: 0,
    };

    detections.forEach((detection) => {
        const classKey = resolveDetectionClassKey(detection);
        if (Object.prototype.hasOwnProperty.call(summary, classKey)) {
            summary[classKey] += 1;
        } else {
            summary.unknown += 1;
        }
    });

    return summary;
};

const ANSI_ESCAPE_PATTERN = /\u001b\[[0-9;?]*[ -/]*[@-~]/g;

const normalizeLogChunk = (value) => String(value ?? '')
    .replace(ANSI_ESCAPE_PATTERN, '')
    .replace(/\r\n/g, '\n')
    .replace(/\r/g, '\n');

const Presentation = () => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [logs, setLogs] = useState([]);
    const [isRunning, setIsRunning] = useState(false);
    const [videoPredictionActive, setVideoPredictionActive] = useState(false);
    const [videoLinks, setVideoLinks] = useState('');
    const logsEndRef = useRef(null);

    // Predictor State
    const [imagePath, setImagePath] = useState(null);
    const [detections, setDetections] = useState([]);
    const [predicting, setPredicting] = useState(false);
    const [hasPredictionResult, setHasPredictionResult] = useState(false);
    const [predictionSummary, setPredictionSummary] = useState(createEmptyPredictionSummary);

    // Dataset Manager State
    const [datasets, setDatasets] = useState([]);
    const [selectedDataset, setSelectedDataset] = useState(null);
    const [selectedCsv, setSelectedCsv] = useState('');
    const [beforeImage, setBeforeImage] = useState(null);
    const [afterImage, setAfterImage] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [isFixing, setIsFixing] = useState(false);
    const [trainEpochs, setTrainEpochs] = useState(10);
    const [trainBatchSize, setTrainBatchSize] = useState(16);
    const [trainImageSize, setTrainImageSize] = useState(640);
    const [trainTestSplit, setTrainTestSplit] = useState(20);
    const [trainConfidenceThreshold, setTrainConfidenceThreshold] = useState(25);
    const [trainDeviceMode, setTrainDeviceMode] = useState('cuda');
    const [trainModel, setTrainModel] = useState('yolov8n');
    const [trainedModels, setTrainedModels] = useState([]);
    const [selectedTrainingSummaryPath, setSelectedTrainingSummaryPath] = useState('');
    const [isLoadingTrainingRun, setIsLoadingTrainingRun] = useState(false);
    const [isActivatingTrainingRun, setIsActivatingTrainingRun] = useState(false);
    const [trainingSummary, setTrainingSummary] = useState(null);
    const [testReviewManifest, setTestReviewManifest] = useState(null);
    const [trainingArtifactManifest, setTrainingArtifactManifest] = useState(null);
    const [trainingResultsRows, setTrainingResultsRows] = useState([]);
    const [mergeSelections, setMergeSelections] = useState({});
    const [mergeOutputName, setMergeOutputName] = useState('');

    // Data Collector State
    const [videos, setVideos] = useState([]);
    const [selectedVideo, setSelectedVideo] = useState('');
    const [collectionName, setCollectionName] = useState('');
    const [augmentationName, setAugmentationName] = useState('');
    const [collectorSession, setCollectorSession] = useState(null);
    const isCollectorWorkspaceActive = currentSlide === 1 && Boolean(collectorSession);
    const trainingLogsRef = useRef(null);
    const selectedDatasetTrainable = Boolean(selectedDataset?.hasImages);
    const activeTrainingRun = trainedModels.find((run) => run.isActive) || null;

    const appendLog = (...entries) => {
        const nextEntries = entries.flat().filter(Boolean);
        if (nextEntries.length === 0) return;
        setLogs(prev => [...prev, ...nextEntries]);
    };

    const selectedMergeSources = datasets.filter((dataset) => mergeSelections[dataset.name]?.enabled)
        .map((dataset) => ({
            datasetName: dataset.name,
            datasetPath: dataset.path,
            csvName: mergeSelections[dataset.name]?.csvName || dataset.csvs?.[0] || '',
        }))
        .filter((dataset) => dataset.csvName);

    const handleLoadTrainingRun = async (summaryPath, options = {}) => {
        if (!summaryPath) return false;

        const { appendSelectionLog = true } = options;
        setIsLoadingTrainingRun(true);
        try {
            const result = await window.electronAPI.loadTrainingRun(summaryPath);
            if (!result.success) {
                if (appendSelectionLog) {
                    appendLog(`[Error] ${result.error}`);
                }
                return false;
            }

            setSelectedTrainingSummaryPath(result.summaryPath || summaryPath);
            setTrainingSummary(result.summary || null);
            setTestReviewManifest(result.reviewManifest || null);
            setTrainingArtifactManifest(result.artifactManifest || null);
            setTrainingResultsRows(Array.isArray(result.trainingResultsRows) ? result.trainingResultsRows : []);

            if (appendSelectionLog) {
                const runLabel = result.savedRun?.datasetName || result.summary?.training_source?.dataset_name || 'saved model';
                appendLog(`[Model] Loaded ${runLabel} trained on ${formatTrainingTimestamp(result.savedRun?.createdAt || result.summary?.created_at)}.`);
            }
            return true;
        } finally {
            setIsLoadingTrainingRun(false);
        }
    };

    const handleFetchTrainingRuns = async (preferredSummaryPath = null, options = {}) => {
        const { autoLoad = true, appendSelectionLog = false } = options;
        const result = await window.electronAPI.listTrainingRuns();
        const runs = Array.isArray(result) ? result : [];
        setTrainedModels(runs);

        if (!runs.length) {
            setSelectedTrainingSummaryPath('');
            setTrainingSummary(null);
            setTestReviewManifest(null);
            setTrainingArtifactManifest(null);
            setTrainingResultsRows([]);
            return;
        }

        if (!autoLoad) return;

        const preferredRun = preferredSummaryPath
            ? runs.find((run) => run.summaryPath === preferredSummaryPath)
            : null;
        const existingRun = selectedTrainingSummaryPath
            ? runs.find((run) => run.summaryPath === selectedTrainingSummaryPath)
            : null;
        const activeRun = runs.find((run) => run.isActive) || null;
        const runToLoad = preferredRun || existingRun || activeRun || runs[0];

        if (!runToLoad) return;
        if (runToLoad.summaryPath === selectedTrainingSummaryPath && trainingSummary) return;
        await handleLoadTrainingRun(runToLoad.summaryPath, { appendSelectionLog });
    };

    const handleActivateTrainingRun = async (summaryPath) => {
        if (!summaryPath) return false;

        setIsActivatingTrainingRun(true);
        try {
            const result = await window.electronAPI.activateTrainingRun(summaryPath);
            if (!result.success) {
                appendLog(`[Error] ${result.error}`);
                return false;
            }

            setSelectedTrainingSummaryPath(result.summaryPath || summaryPath);
            setTrainingSummary(result.summary || null);
            setTestReviewManifest(result.reviewManifest || null);
            setTrainingArtifactManifest(result.artifactManifest || null);
            setTrainingResultsRows(Array.isArray(result.trainingResultsRows) ? result.trainingResultsRows : []);
            await handleFetchTrainingRuns(result.summaryPath || summaryPath, { autoLoad: false });

            const runLabel = result.savedRun?.datasetName || result.summary?.training_source?.dataset_name || 'saved model';
            appendLog(`[Model] Activated ${runLabel}. New predictions will use this model.`);
            return true;
        } finally {
            setIsActivatingTrainingRun(false);
        }
    };

    const handleFetchDatasets = async () => {
        const data = await window.electronAPI.listDatasets();
        setDatasets(data);
        setMergeSelections((prev) => {
            const next = {};
            data.forEach((dataset) => {
                next[dataset.name] = {
                    enabled: prev[dataset.name]?.enabled || false,
                    csvName: prev[dataset.name]?.csvName || dataset.csvs?.[0] || '',
                };
            });
            return next;
        });
        if (data.length > 0 && !selectedDataset) {
            const preferredDataset = data.find((dataset) => dataset.hasImages) || data[0];
            setSelectedDataset(preferredDataset);
            setSelectedCsv(preferredDataset.csvs[0]);
        }
    };

    const handleFetchVideos = async () => {
        const data = await window.electronAPI.listVideos();
        setVideos(data);
        if (data.length > 0 && !selectedVideo) {
            setSelectedVideo(data[0].name);
        } else if (data.length === 0) {
            setSelectedVideo('');
        }
    };

    useEffect(() => {
        if (window.electronAPI?.onPipelineOutput) {
            window.electronAPI.onPipelineOutput((data) => {
                const message = normalizeLogChunk(data.msg);
                if (!message) return;
                setLogs(prev => [...prev, message]);
            });
        }
        handleFetchDatasets();
        handleFetchVideos();
        handleFetchTrainingRuns();
        return () => {
            if (window.electronAPI?.removePipelineOutputListener) {
                window.electronAPI.removePipelineOutputListener();
            }
        };
    }, []);

    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    const handleRunStep = async (scriptName, args = []) => {
        if (isRunning) return;
        setIsRunning(true);
        setLogs([`> Starting ${scriptName} ${args.join(' ')}...`]);
        try {
            const result = await window.electronAPI.runPipelineStep(scriptName, args);
            if (result.error) {
                setLogs(prev => [...prev, `\n[Error] ${result.error}`]);
            } else {
                setLogs(prev => [...prev, '\n[Success] Process completed naturally.']);
            }
        } catch (err) {
            setLogs(prev => [...prev, `\n[Exception] ${err.message}`]);
        } finally {
            setIsRunning(false);
        }
    };

    const handleDownloadStep = async () => {
        if (isRunning) return;
        if (videoLinks.trim()) {
            await window.electronAPI.saveLinks(videoLinks);
        }
        handleRunStep('download_videos.py', []);
    };

    const handleRunDataCollection = async (videoName) => {
        if (!videoName || isRunning) return;
        setIsRunning(true);
        const finalName = collectionName.trim() || `collected_${new Date().toISOString().split('T')[0]}`;
        setLogs([
            `> Opening in-app data collection for video: ${videoName}...`,
            `> Dataset Name: ${finalName}`,
            `> Mode: Manual multi-class annotation inside Electron`,
            `> Choose enemy or player, draw boxes, and step through frames with the keyboard.`
        ]);
        const result = await window.electronAPI.startManualCollection(videoName, finalName);
        if (result.success) {
            setCollectorSession(result);
            appendLog(`[Ready] Manual annotator loaded inside the app for ${result.videoName}.`);
            appendLog(`[Hint] Press 1 for enemy, 2 for player, Shift+X to add another box on the same frame, and X to save and move on.`);
        } else {
            appendLog(`\n[Error] ${result.error}`);
        }
        setIsRunning(false);
    };

    const handleCollectorSessionUpdate = (updates) => {
        setCollectorSession(prev => prev ? { ...prev, ...updates } : prev);
    };

    const handleCollectorClosed = (result) => {
        if (result?.success && collectorSession) {
            appendLog(`\n[Success] Manual collection complete. Saved ${result.savedCount} annotation(s) to data_sets/${collectorSession.datasetName}.`);
            appendLog('[System] The dataset is ready for review in Dataset Manager.');
        }
        setCollectorSession(null);
        handleFetchDatasets();
    };

    const handleRunAugmentation = async () => {
        if (!selectedDataset || !selectedCsv || isRunning) return;
        setIsRunning(true);
        const finalName = augmentationName.trim() || `${selectedDataset.name}_augmented`;
        setLogs([`> Starting data augmentation for ${selectedDataset.name}...`, `> Source CSV: ${selectedCsv}`, `> Output Name: ${finalName}`]);
        const result = await window.electronAPI.runAugmentation(selectedDataset.path, selectedCsv, finalName);
        if (result.success) {
            setLogs(prev => [...prev, `\n[Success] Augmentation complete!`, `[System] Augmented dataset saved to data_sets/${finalName}`]);
            handleFetchDatasets();
        } else {
            setLogs(prev => [...prev, `\n[Error] ${result.error}`]);
        }
        setIsRunning(false);
    };

    const handleToggleMergeDataset = (datasetName) => {
        setMergeSelections((prev) => ({
            ...prev,
            [datasetName]: {
                enabled: !prev[datasetName]?.enabled,
                csvName: prev[datasetName]?.csvName || datasets.find((dataset) => dataset.name === datasetName)?.csvs?.[0] || '',
            },
        }));
    };

    const handleMergeCsvChange = (datasetName, csvName) => {
        setMergeSelections((prev) => ({
            ...prev,
            [datasetName]: {
                enabled: prev[datasetName]?.enabled || false,
                csvName,
            },
        }));
    };

    const handleRunMerge = async () => {
        if (isRunning || selectedMergeSources.length < 2) return;

        setIsRunning(true);
        const finalName = mergeOutputName.trim() || `merged_${new Date().toISOString().split('T')[0]}`;
        setLogs([
            `> Starting dataset merge...`,
            `> Output Dataset: ${finalName}`,
            ...selectedMergeSources.map((source, index) => `> Source ${index + 1}: ${source.datasetName} -> ${source.csvName}`),
        ]);

        try {
            const result = await window.electronAPI.mergeDatasets({
                outputDatasetName: finalName,
                sources: selectedMergeSources,
            });

            if (result.error) {
                setLogs((prev) => [...prev, `\n[Error] ${result.error}`]);
            } else {
                setLogs((prev) => [
                    ...prev,
                    `\n[Success] Merge complete.`,
                    `[System] Created ${result.datasetName} with ${result.totals.images} image(s) and ${result.totals.annotations} annotation row(s).`,
                ]);
                setMergeOutputName('');
                await handleFetchDatasets();
                const mergedDataset = datasets.find((dataset) => dataset.name === result.datasetName)
                    || { name: result.datasetName, path: result.datasetPath, csvs: [result.csvName] };
                setSelectedDataset(mergedDataset);
                setSelectedCsv(result.csvName);
            }
        } catch (err) {
            setLogs((prev) => [...prev, `\n[Exception] ${err.message}`]);
        } finally {
            setIsRunning(false);
        }
    };

    const handleCancel = async () => {
        if (!isRunning) return;
        if (videoPredictionActive) {
            setLogs(prev => [...prev, '\n[System] Stopping active video prediction session...']);
            await window.electronAPI.stopVideoPrediction();
            return;
        }
        setLogs(prev => [...prev, '\n[System] Aborting current pipeline process via KILL signal...']);
        await window.electronAPI.cancelPipeline();
        setIsRunning(false);
    };

    const handleSelectImage = async () => {
        const path = await window.electronAPI.selectImage();
        if (path) {
            setImagePath(path);
            setDetections([]);
            setHasPredictionResult(false);
            setPredictionSummary(createEmptyPredictionSummary());
        }
    };

    const handlePredict = async () => {
        if (!imagePath) return;
        setPredicting(true);
        setLogs([`> Analyzing image: ${imagePath}...`]);
        try {
            const result = await window.electronAPI.runPrediction(imagePath);
            if (result.error) {
                setLogs(prev => [...prev, `[Error] ${result.error}`]);
            } else {
                const nextDetections = (result.detections || []).map((detection) => ({
                    ...detection,
                    class_key: resolveDetectionClassKey(detection),
                    class_name: formatDetectionClassName(detection),
                }));
                const topDetection = result.top_detection
                    ? {
                        ...result.top_detection,
                        class_key: resolveDetectionClassKey(result.top_detection),
                        class_name: formatDetectionClassName(result.top_detection),
                    }
                    : null;
                const nextSummary = {
                    enemy: Number(result.class_summary?.enemy || 0),
                    player: Number(result.class_summary?.player || 0),
                    unknown: 0,
                    ...summarizeDetections(nextDetections),
                };

                setDetections(nextDetections);
                setHasPredictionResult(true);
                setPredictionSummary({
                    count: typeof result.count === 'number' ? result.count : nextDetections.length,
                    topDetection,
                    modelPath: result.model_path || null,
                    savedImagePath: result.saved_image_path || null,
                    classSummary: nextSummary,
                });
                if (topDetection) {
                    const top = topDetection;
                    setLogs(prev => [...prev, `[Success] Found ${result.count} detection(s). Top detection: ${top.class_name} ${(top.confidence * 100).toFixed(1)}% @ [${top.x_center.toFixed(3)}, ${top.y_center.toFixed(3)}]`]);
                } else {
                    setLogs(prev => [...prev, `[Success] No detections above threshold.`]);
                }
                if (result.saved_image_path) {
                    setLogs(prev => [...prev, `[System] Stamped image saved to: ${result.saved_image_path}`]);
                }
            }
        } catch (err) {
            setLogs(prev => [...prev, `[Exception] ${err.message}`]);
        } finally {
            setPredicting(false);
        }
    };

    const handleAnalyzeBias = async () => {
        if (!selectedDataset || !selectedCsv) return;
        setIsAnalyzing(true);
        setLogs([`> Analyzing bias for ${selectedDataset.name}...`]);
        const result = await window.electronAPI.analyzeDatasetBias(selectedDataset.path, selectedCsv);
        if (result.success) {
            setBeforeImage(result.imagePath);
            setLogs(prev => [...prev, `[Success] Before-fix visualization generated.`]);
        } else {
            setLogs(prev => [...prev, `[Error] ${result.error}`]);
        }
        setIsAnalyzing(false);
    };

    const handleRunFix = async () => {
        if (!selectedDataset || !selectedCsv) return;
        setIsFixing(true);
        setLogs([`> Applying anti-bias fix and rebalancing to ${selectedDataset.name}...`]);
        const result = await window.electronAPI.runBiasFix(selectedDataset.path, selectedCsv);
        if (result.success) {
            setAfterImage(result.imagePath);
            setLogs(prev => [...prev, `[Success] Anti-bias fix applied. Balanced dataset created.`, `[System] Result saved to: ${result.csvPath}`]);
            handleFetchDatasets();
        } else {
            setLogs(prev => [...prev, `[Error] ${result.error}`]);
        }
        setIsFixing(false);
    };

    const handleTrainOnDataset = async () => {
        if (!selectedDataset || !selectedCsv) return;
        trainingLogsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        setIsRunning(true);
        setTrainingSummary(null);
        setTestReviewManifest(null);
        setTrainingArtifactManifest(null);
        setTrainingResultsRows([]);
        setLogs([
            `> Starting training on dataset: ${selectedDataset.name}`,
            `> CSV: ${selectedCsv}`,
            `> Model Basis: ${trainModel}`,
            `> Epochs: ${trainEpochs}`,
            `> Batch Size: ${trainBatchSize}`,
            `> Image Size: ${trainImageSize}`,
            `> Test Split: ${trainTestSplit}%`,
            `> Review Confidence Threshold: ${trainConfidenceThreshold}%`,
            `> Device Mode: ${trainDeviceMode === 'cuda' ? 'CUDA / NVIDIA GPU only' : trainDeviceMode === 'auto' ? 'Auto (prefer CUDA)' : 'CPU only'}`,
        ]);
        const result = await window.electronAPI.runTraining({
            datasetPath: selectedDataset.path,
            csvName: selectedCsv,
            epochs: trainEpochs,
            batchSize: trainBatchSize,
            imageSize: trainImageSize,
            testSplit: trainTestSplit,
            confidenceThreshold: trainConfidenceThreshold,
            deviceMode: trainDeviceMode,
            modelChoice: trainModel,
        });
        if (result.success) {
            const savedSummaryPath = result.savedRun?.summaryPath || result.summaryPath || '';
            setSelectedTrainingSummaryPath(savedSummaryPath);
            setTrainingSummary(result.summary || null);
            setTestReviewManifest(result.reviewManifest || null);
            setTrainingArtifactManifest(result.artifactManifest || null);
            setTrainingResultsRows(Array.isArray(result.trainingResultsRows) ? result.trainingResultsRows : []);
            await handleFetchTrainingRuns(savedSummaryPath, { autoLoad: false });
            const metrics = result.summary?.evaluation?.metrics || result.testMetrics?.aggregate_metrics || {};
            setLogs(prev => [
                ...prev,
                `\n[Success] Training completed successfully with ${trainModel}.`,
                `[Eval] Precision: ${formatPercent(metrics.precision)}`,
                `[Eval] Recall: ${formatPercent(metrics.recall)}`,
                `[Eval] mAP50: ${formatPercent(metrics.map50)}`,
                `[Eval] mAP50-95: ${formatPercent(metrics.map50_95)}`,
                `[Model] Saved snapshot: ${savedSummaryPath || 'latest model history entry'}`,
                ...(result.historyWarning ? [`[Warning] ${result.historyWarning}`] : []),
                ...(result.activationWarning ? [`[Warning] ${result.activationWarning}`] : ['[Model] Activated the newly trained model for future predictions.']),
            ]);
        } else {
            setLogs(prev => [...prev, `\n[Error] ${result.error}`]);
        }
        setIsRunning(false);
    };

    const renderTerminal = (compact = false, options = {}) => {
        const {
            title = 'TERMINAL',
            emptyMessage = 'No output yet. Run a process to see logs here.',
            containerRef = null,
        } = options;

        return (
        <div
            ref={containerRef}
            className={`bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col font-mono text-sm ${compact ? 'mt-4 h-40 max-h-40' : 'mt-6 h-64 max-h-96'}`}
        >
            <div className="bg-slate-900 border-b border-slate-800 px-4 py-2 flex items-center justify-between">
                <div className="flex gap-2.5 items-center">
                    <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
                    <span className="ml-2 text-slate-500 text-xs tracking-wider">{title}</span>
                </div>
                {isRunning && (
                    <div className="flex items-center gap-4">
                        <button 
                            onClick={handleCancel} 
                            className="px-3 py-1 bg-red-600/20 text-red-500 hover:bg-red-600/40 rounded border border-red-500/30 text-xs font-bold uppercase tracking-wider transition-colors shadow-lg shadow-red-900/20"
                        >Force Stop</button>
                        <span className="text-amber-300 text-xs animate-pulse">Running...</span>
                    </div>
                )}
            </div>
            <div className="p-4 overflow-y-auto flex-1 text-slate-300 leading-relaxed break-all whitespace-pre-wrap">
                {logs.length === 0 ? (
                    <span className="text-slate-600 italic">{emptyMessage}</span>
                ) : (
                    logs.map((log, i) => (
                        <div key={i}>{log}</div>
                    ))
                )}
                <div ref={logsEndRef} />
            </div>
        </div>
        );
    };

    const slides = [
        {
            id: 'download',
            title: "Downloader",
            subtitle: "Gather gameplay",
            icon: <CloudDownload size={22} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-4">
                        <textarea
                            value={videoLinks}
                            onChange={(e) => setVideoLinks(e.target.value)}
                            placeholder="Enter YouTube links..."
                            className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-300 font-mono focus:outline-none focus:border-rose-500 transition-colors resize-none"
                        />
                        <button 
                            onClick={handleDownloadStep}
                            disabled={isRunning || !videoLinks.trim()}
                            className="px-6 py-2 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white font-bold flex justify-center items-center gap-2 shadow-lg shadow-rose-950/30"
                        >
                            {isRunning && currentSlide === 0 ? (
                                <>
                                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Downloading...
                                </>
                            ) : (
                                "Start Download"
                            )}
                        </button>
                        {isRunning && currentSlide === 0 && (
                            <div className="w-full bg-slate-900 rounded-full h-2 mt-2 overflow-hidden border border-slate-700">
                                <div className="bg-rose-500 h-full w-1/2 rounded-full animate-progress"></div>
                            </div>
                        )}
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'collector',
            title: "Data Collector",
            subtitle: "Label gameplay frames directly inside the app",
            icon: <MonitorPlay size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    {collectorSession ? (
                        <CollectorWorkspace
                            session={collectorSession}
                            appendLog={appendLog}
                            onClose={handleCollectorClosed}
                            onSessionUpdate={handleCollectorSessionUpdate}
                        />
                    ) : (
                        <div className="flex flex-col gap-6">
                            <div className="grid grid-cols-1 xl:grid-cols-[1.35fr_1fr] gap-6">
                                <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                                    <div className="flex flex-col gap-2">
                                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Select Video Source</label>
                                        <div className="flex gap-4">
                                            <select 
                                                className="flex-1 bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                                value={selectedVideo}
                                                onChange={(e) => setSelectedVideo(e.target.value)}
                                            >
                                                {videos.length === 0 && <option value="">No videos found in src/videos</option>}
                                                {videos.map((video) => (
                                                    <option key={video.name} value={video.name}>{video.name}</option>
                                                ))}
                                            </select>
                                            <button 
                                                onClick={handleFetchVideos}
                                                className="p-2.5 bg-slate-700 hover:bg-slate-600 rounded-lg transition"
                                                title="Refresh List"
                                            >
                                                <CloudDownload size={20} />
                                            </button>
                                        </div>
                                    </div>

                                    <div className="flex flex-col gap-2">
                                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">New Dataset Name</label>
                                        <input 
                                            type="text"
                                            placeholder="e.g. desert_outpost_labels"
                                            className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                            value={collectionName}
                                            onChange={(e) => setCollectionName(e.target.value)}
                                        />
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-4">
                                            <div className="text-[11px] font-black uppercase tracking-[0.24em] text-rose-400 mb-2">Manual Flow</div>
                                            <div className="text-sm text-slate-300 space-y-2">
                                                <p>Open the annotator inside the Electron app.</p>
                                                <p>Choose <span className="text-white font-semibold">enemy</span> for opponents and <span className="text-white font-semibold">player</span> for your own character.</p>
                                                <p>Drag a bounding box around the selected target, or single-click for a quick centered square box.</p>
                                                <p>Save useful frames and stack multiple labeled characters on the same frame without leaving the app.</p>
                                            </div>
                                        </div>
                                        <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-4">
                                            <div className="text-[11px] font-black uppercase tracking-[0.24em] text-orange-300 mb-2">Quick Controls</div>
                                            <div className="text-sm text-slate-300 space-y-2">
                                                <p><span className="text-white font-semibold">1</span> selects enemy and <span className="text-white font-semibold">2</span> selects player.</p>
                                                <p><span className="text-white font-semibold">Shift + X</span> adds another labeled box on the same frame, and <span className="text-white font-semibold">X</span> saves and advances.</p>
                                                <p><span className="text-white font-semibold">Arrow keys</span> move frame by frame, and <span className="text-white font-semibold">Shift + Arrows</span> jumps further.</p>
                                                <p><span className="text-white font-semibold">D</span> duplicates the last saved box and <span className="text-white font-semibold">S</span> skips the current frame.</p>
                                            </div>
                                        </div>
                                    </div>

                                    <button 
                                        onClick={() => handleRunDataCollection(selectedVideo)}
                                        disabled={isRunning || !selectedVideo}
                                        className="px-6 py-3 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-rose-950/30 text-sm font-bold flex justify-center items-center gap-2"
                                    >
                                        Open In-App Annotator <MonitorPlay size={18}/>
                                    </button>
                                </div>

                                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 flex flex-col gap-4">
                                    <div className="text-[11px] font-black uppercase tracking-[0.24em] text-violet-400">Why This Works Better</div>
                                    <div className="text-sm text-slate-300 space-y-3">
                                        <p>Bounding boxes are now drawn directly where you can see them, so there is no hidden OpenCV window and no guessing what hotkeys do.</p>
                                        <p>The annotator now supports explicit player-versus-enemy labeling, which is much better for a third-person game where your own avatar is always visible.</p>
                                        <p>You also get timeline scrubbing, adjustable frame stepping, duplicate-last-box, and mixed multi-character labeling on the same frame.</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                    {renderTerminal(Boolean(collectorSession))}
                </div>
            )
        },
        {
            id: 'datasets',
            title: "Dataset Manager",
            subtitle: "Analyze and neutralize positional bias",
            icon: <Database size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="flex flex-col gap-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Select Dataset</label>
                                <select 
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    value={selectedDataset?.name || ''}
                                    onChange={(e) => {
                                        const ds = datasets.find(d => d.name === e.target.value);
                                        setSelectedDataset(ds);
                                        setSelectedCsv(ds?.csvs[0] || '');
                                        setBeforeImage(null);
                                        setAfterImage(null);
                                    }}
                                >
                                    {datasets.map(ds => <option key={ds.name} value={ds.name}>{ds.name}</option>)}
                                </select>
                            </div>
                            <div className="flex flex-col gap-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">CSV Labels</label>
                                <select 
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    value={selectedCsv}
                                    onChange={(e) => setSelectedCsv(e.target.value)}
                                >
                                    {selectedDataset?.csvs.map(csv => <option key={csv} value={csv}>{csv}</option>)}
                                </select>
                            </div>
                        </div>

                        <div className="flex gap-4">
                            <button 
                                onClick={handleAnalyzeBias}
                                disabled={isAnalyzing || !selectedDataset}
                                className="flex-1 py-2.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg transition font-bold text-sm flex items-center justify-center gap-2"
                            >
                                {isAnalyzing ? 'Analyzing...' : 'Analyze Bias (Before)'}
                                <BarChart3 size={18} />
                            </button>
                            <button 
                                onClick={handleRunFix}
                                disabled={isFixing || !selectedDataset}
                                className="flex-1 py-2.5 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition font-bold text-sm text-white flex items-center justify-center gap-2 shadow-lg shadow-rose-950/30"
                            >
                                {isFixing ? 'Cleaning...' : 'Apply Anti-Bias Fix (After)'}
                                <BrainCircuit size={18} />
                            </button>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 min-h-[350px]">
                        <div className="bg-slate-900/50 rounded-xl border border-slate-800 flex flex-col items-center justify-center p-4 relative overflow-hidden group">
                            <div className="absolute top-4 left-4 z-10 bg-slate-800/80 backdrop-blur px-3 py-1 rounded text-[10px] font-black uppercase tracking-widest text-slate-400 border border-slate-700">Original Bias</div>
                            {beforeImage ? (
                                <img src={`file://${beforeImage}?t=${new Date().getTime()}`} className="max-w-full max-h-full object-contain rounded shadow-2xl" alt="Before" />
                            ) : (
                                <div className="text-slate-600 flex flex-col items-center gap-2">
                                    <BarChart3 size={32} className="opacity-20" />
                                    <span className="text-xs italic">Run analysis to see distribution</span>
                                </div>
                            )}
                        </div>
                        <div className="bg-slate-900/50 rounded-xl border border-slate-800 flex flex-col items-center justify-center p-4 relative overflow-hidden group">
                            <div className="absolute top-4 left-4 z-10 bg-rose-600/80 backdrop-blur px-3 py-1 rounded text-[10px] font-black uppercase tracking-widest text-white border border-rose-300/30">Neutralized & Balanced</div>
                            {afterImage ? (
                                <img src={`file://${afterImage}?t=${new Date().getTime()}`} className="max-w-full max-h-full object-contain rounded shadow-2xl" alt="After" />
                            ) : (
                                <div className="text-slate-600 flex flex-col items-center gap-2">
                                    <CheckCircle size={32} className="opacity-20" />
                                    <span className="text-xs italic">Apply fix to see results</span>
                                </div>
                            )}
                        </div>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'augmenter',
            title: "Dataset Augmenter",
            subtitle: "Boost dataset with spatial variations",
            icon: <Layers size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="flex flex-col gap-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Select Source Dataset</label>
                                <select 
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    value={selectedDataset?.name || ''}
                                    onChange={(e) => {
                                        const ds = datasets.find(d => d.name === e.target.value);
                                        setSelectedDataset(ds);
                                        setSelectedCsv(ds?.csvs[0] || '');
                                    }}
                                >
                                    {datasets.map(ds => <option key={ds.name} value={ds.name}>{ds.name}</option>)}
                                </select>
                            </div>
                            <div className="flex flex-col gap-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Source CSV</label>
                                <select 
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    value={selectedCsv}
                                    onChange={(e) => setSelectedCsv(e.target.value)}
                                >
                                    {selectedDataset?.csvs.map(csv => <option key={csv} value={csv}>{csv}</option>)}
                                </select>
                            </div>
                        </div>

                        <div className="flex flex-col gap-2">
                            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Augmented Dataset Name</label>
                            <input 
                                type="text"
                                placeholder="e.g. combined_augmented_v1"
                                className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-orange-400"
                                value={augmentationName}
                                onChange={(e) => setAugmentationName(e.target.value)}
                            />
                        </div>

                        <div className="bg-rose-900/20 border border-rose-800/50 p-4 rounded-lg">
                            <h4 className="text-rose-300 text-xs font-bold uppercase tracking-widest mb-2">Augmentation Strategy</h4>
                            <ul className="text-xs text-slate-400 space-y-1">
                                <li>• Spatially-aware augmentation (Edges/Corners priority)</li>
                                <li>• Automatic bounding-box transformation for rotates/flips</li>
                                <li>• HUD-aware "Masked Pan" relocation</li>
                                <li>• Pixel-level noise, blur and brightness shifts</li>
                            </ul>
                        </div>

                        <button 
                            onClick={handleRunAugmentation}
                            disabled={isRunning || !selectedDataset}
                            className="px-6 py-3 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-orange-950/30 text-sm font-bold flex justify-center items-center gap-2"
                        >
                            Run Augmentation Pipeline <Layers size={18}/>
                        </button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'merge',
            title: "Dataset Merger",
            subtitle: "Combine multiple labeled datasets into one",
            icon: <Copy size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="flex flex-col gap-2">
                            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Merged Dataset Name</label>
                            <input
                                type="text"
                                placeholder="e.g. merged_scrims_v1"
                                className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                value={mergeOutputName}
                                onChange={(e) => setMergeOutputName(e.target.value)}
                            />
                        </div>

                        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-sm text-slate-300 leading-relaxed">
                            Pick at least two source datasets below. The merger copies their images and labels into a new dataset folder, renames files to avoid collisions, and writes a combined CSV you can train on immediately.
                        </div>

                        <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
                            {datasets.length === 0 ? (
                                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-500">
                                    No datasets found yet.
                                </div>
                            ) : (
                                datasets.map((dataset) => {
                                    const selection = mergeSelections[dataset.name] || {};
                                    return (
                                        <div key={dataset.name} className={`rounded-xl border p-4 transition ${selection.enabled ? 'border-rose-500/40 bg-rose-500/10' : 'border-slate-800 bg-slate-900/60'}`}>
                                            <div className="flex flex-col lg:flex-row lg:items-center gap-3">
                                                <label className="flex items-center gap-3 flex-1 cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={Boolean(selection.enabled)}
                                                        onChange={() => handleToggleMergeDataset(dataset.name)}
                                                        className="h-4 w-4 rounded border-slate-600 bg-slate-900 text-rose-500 focus:ring-rose-500"
                                                    />
                                                    <div className="min-w-0">
                                                        <div className="text-sm font-semibold text-slate-100 truncate">{dataset.name}</div>
                                                        <div className="text-xs text-slate-500 truncate">{dataset.path}</div>
                                                    </div>
                                                </label>
                                                <div className="flex flex-col gap-1 min-w-[220px]">
                                                    <label className="text-[11px] font-bold text-slate-500 uppercase tracking-widest">CSV To Merge</label>
                                                    <select
                                                        value={selection.csvName || dataset.csvs?.[0] || ''}
                                                        onChange={(e) => handleMergeCsvChange(dataset.name, e.target.value)}
                                                        className="bg-slate-950 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                                    >
                                                        {dataset.csvs?.map((csv) => <option key={csv} value={csv}>{csv}</option>)}
                                                    </select>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })
                            )}
                        </div>

                        <div className="flex items-center gap-4 text-sm">
                            <span className="text-rose-400 font-bold whitespace-nowrap">Selected Sources:</span>
                            <div className="flex-1 px-3 py-2 bg-slate-900 rounded border border-slate-700 text-slate-300 font-mono text-xs overflow-hidden text-ellipsis italic">
                                {selectedMergeSources.length > 0
                                    ? selectedMergeSources.map((source) => `${source.datasetName} -> ${source.csvName}`).join(' | ')
                                    : 'Choose two or more datasets'}
                            </div>
                        </div>

                        <button
                            onClick={handleRunMerge}
                            disabled={isRunning || selectedMergeSources.length < 2}
                            className="px-6 py-3 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-rose-950/30 text-sm font-bold flex justify-center items-center gap-2"
                        >
                            Merge Selected Datasets <Copy size={18}/>
                        </button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'viewer',
            title: "Data Viewer",
            subtitle: "Browse, fix, and delete labeled images",
            icon: <ImageIcon size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="flex flex-col gap-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Select Dataset</label>
                                <select
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    value={selectedDataset?.name || ''}
                                    onChange={(e) => {
                                        const ds = datasets.find(d => d.name === e.target.value);
                                        setSelectedDataset(ds);
                                        setSelectedCsv(ds?.csvs[0] || '');
                                    }}
                                >
                                    {datasets.length === 0 && <option value="">No datasets found</option>}
                                    {datasets.map(ds => <option key={ds.name} value={ds.name}>{ds.name}</option>)}
                                </select>
                            </div>
                            <div className="flex flex-col gap-2">
                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">CSV Labels</label>
                                <select
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    value={selectedCsv}
                                    onChange={(e) => setSelectedCsv(e.target.value)}
                                >
                                    {selectedDataset?.csvs?.map(csv => <option key={csv} value={csv}>{csv}</option>)}
                                </select>
                            </div>
                        </div>

                        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-sm text-slate-300 leading-relaxed">
                            This viewer opens every image in the selected dataset so you can inspect labels, move or resize existing boxes, add new player or enemy boxes, save fixes back into the YOLO files and CSV, or delete bad images entirely.
                        </div>
                    </div>

                    <DatasetViewerWorkspace
                        dataset={selectedDataset}
                        csvName={selectedCsv}
                        appendLog={appendLog}
                        onDatasetChanged={handleFetchDatasets}
                    />
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'train',
            title: "Model Training",
            subtitle: "Train on selected dataset",
            icon: <BrainCircuit size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="flex flex-col gap-4">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div className="flex flex-col gap-2">
                                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Training Dataset</label>
                                    <select
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                        value={selectedDataset?.name || ''}
                                        onChange={(e) => {
                                            const ds = datasets.find(d => d.name === e.target.value);
                                            setSelectedDataset(ds);
                                            setSelectedCsv(ds?.csvs?.[0] || '');
                                        }}
                                    >
                                        {datasets.length === 0 && <option value="">No datasets found</option>}
                                        {datasets.map(ds => (
                                            <option key={ds.name} value={ds.name}>
                                                {ds.name}{ds.hasImages ? '' : ' (missing images)'}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div className="flex flex-col gap-2">
                                    <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Training CSV</label>
                                    <select
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                        value={selectedCsv}
                                        onChange={(e) => setSelectedCsv(e.target.value)}
                                    >
                                        {selectedDataset?.csvs?.length
                                            ? selectedDataset.csvs.map(csv => <option key={csv} value={csv}>{csv}</option>)
                                            : <option value="">No CSV files found</option>}
                                    </select>
                                </div>
                            </div>

                            <div className="flex items-center gap-4 text-sm">
                                <span className="text-rose-400 font-bold whitespace-nowrap">Training Source:</span>
                                <div className="flex-1 px-3 py-1.5 bg-slate-900 rounded border border-slate-700 text-slate-300 font-mono text-xs overflow-hidden text-ellipsis italic">
                                    {selectedDataset ? `${selectedDataset.name} -> ${selectedCsv || 'No CSV selected'}` : "None Selected"}
                                </div>
                            </div>
                            
                            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-6 gap-6">
                                <div className="flex flex-col gap-1.5">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Model Basis</label>
                                    <select
                                        value={trainModel}
                                        onChange={(e) => setTrainModel(e.target.value)}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    >
                                        {DETECTOR_CHOICES.map((choice) => (
                                            <option key={choice.key} value={choice.key}>{choice.label}</option>
                                        ))}
                                    </select>
                                    <p className="text-xs text-slate-500 leading-relaxed">
                                        {DETECTOR_CHOICES.find((choice) => choice.key === trainModel)?.summary}
                                    </p>
                                </div>
                                <div className="flex flex-col gap-1.5 flex-1">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Epochs Count</label>
                                    <input 
                                        type="number" 
                                        value={trainEpochs} 
                                        min="1"
                                        max="1000"
                                        onChange={(e) => setTrainEpochs(Math.max(1, parseInt(e.target.value || '10', 10) || 10))}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    />
                                </div>
                                <div className="flex flex-col gap-1.5 flex-1">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Batch Size</label>
                                    <input
                                        type="number"
                                        min="1"
                                        max="512"
                                        value={trainBatchSize}
                                        onChange={(e) => setTrainBatchSize(Math.max(1, parseInt(e.target.value || '16', 10) || 16))}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    />
                                </div>
                                <div className="flex flex-col gap-1.5 flex-1">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Image Size</label>
                                    <input
                                        type="number"
                                        min="320"
                                        step="32"
                                        value={trainImageSize}
                                        onChange={(e) => setTrainImageSize(Math.max(320, parseInt(e.target.value || '640', 10) || 640))}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    />
                                </div>
                                <div className="flex flex-col gap-1.5 flex-1">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Test Split %</label>
                                    <input
                                        type="number"
                                        min="1"
                                        max="90"
                                        value={trainTestSplit}
                                        onChange={(e) => setTrainTestSplit(Math.max(1, Math.min(90, parseInt(e.target.value || '20', 10) || 20)))}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    />
                                    <p className="text-xs text-slate-500 leading-relaxed">
                                        This is the held-out split used for training-time validation and the final review images shown below.
                                    </p>
                                </div>
                                <div className="flex flex-col gap-1.5 flex-1">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Review Conf %</label>
                                    <input
                                        type="number"
                                        min="0"
                                        max="100"
                                        value={trainConfidenceThreshold}
                                        onChange={(e) => setTrainConfidenceThreshold(Math.max(0, Math.min(100, parseInt(e.target.value || '25', 10) || 25)))}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    />
                                    <p className="text-xs text-slate-500 leading-relaxed">
                                        Used for the saved hold-out review images and per-image review counts. Core validation mAP metrics are still computed by Ultralytics.
                                    </p>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                <div className="flex flex-col gap-1.5">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Training Device</label>
                                    <select
                                        value={trainDeviceMode}
                                        onChange={(e) => setTrainDeviceMode(e.target.value)}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    >
                                        <option value="cuda">CUDA / NVIDIA GPU only</option>
                                        <option value="auto">Auto (prefer CUDA, fallback to CPU)</option>
                                        <option value="cpu">CPU only</option>
                                    </select>
                                    <p className="text-xs text-slate-500 leading-relaxed">
                                        CUDA mode will fail fast if no NVIDIA-capable PyTorch GPU is available, which is safer than silently training on CPU.
                                    </p>
                                </div>
                                <div className="flex flex-col gap-1.5">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">CUDA Notes</label>
                                    <div className="bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-400 leading-relaxed">
                                        Training logs will print the chosen device, GPU name, VRAM, CUDA version, and whether AMP / TF32 acceleration was enabled.
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-sm text-slate-300 leading-relaxed">
                            Training now uses the selected detector basis directly from the app, passes your batch size, image size, and saved-review confidence threshold through to Python, and lets the trainer create its own train/validation split from the chosen CSV.
                            Every successful run is archived into model history so you can reopen earlier checkpoints, browse their saved hold-out reviews, and compare stats without retraining.
                        </div>

                        <div className="bg-emerald-950/30 border border-emerald-700/40 rounded-xl p-4 text-sm text-emerald-100 leading-relaxed">
                            No separate system terminal is required here. Press <span className="font-bold text-white">Start Training</span> and the app will stream Python output into the training log panel below.
                        </div>

                        {!selectedDatasetTrainable && selectedDataset && (
                            <div className="bg-rose-950/40 border border-rose-700/40 rounded-xl p-4 text-sm text-rose-100 leading-relaxed">
                                This dataset is not trainable yet because its <span className="font-bold text-white">images</span> folder is missing or empty.
                                The app found {selectedDataset.imageCount ?? 0} image files and {selectedDataset.labelCount ?? 0} label files in this dataset.
                            </div>
                        )}

                        <button 
                            onClick={handleTrainOnDataset}
                            disabled={isRunning || !selectedDataset || !selectedDatasetTrainable}
                            className="px-6 py-3 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-rose-900/20 text-sm font-bold w-full flex justify-center items-center gap-2"
                        >Start Training <BrainCircuit size={18}/></button>

                        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
                                <div className="flex items-center gap-2 text-slate-200">
                                    <Layers size={16} className="text-rose-400" />
                                    <span className="text-xs font-black uppercase tracking-[0.24em]">Saved Models</span>
                                </div>
                                <div className="text-xs text-slate-500">
                                    {trainedModels.length} saved run{trainedModels.length === 1 ? '' : 's'}
                                </div>
                            </div>

                            {!trainedModels.length ? (
                                <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/60 p-5 text-sm text-slate-400">
                                    Your trained-model history will appear here after the first successful run.
                                </div>
                            ) : (
                                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
                                    {trainedModels.map((run) => {
                                        const isSelected = run.summaryPath === selectedTrainingSummaryPath;
                                        const runActionBusy = isLoadingTrainingRun || isActivatingTrainingRun;
                                        return (
                                            <div
                                                key={run.id}
                                                className={`text-left rounded-2xl border p-4 transition ${
                                                    isSelected
                                                        ? 'border-rose-500/40 bg-rose-500/10'
                                                        : 'border-slate-800 bg-slate-950/70 hover:bg-slate-900'
                                                } ${runActionBusy ? 'opacity-90' : ''}`}
                                            >
                                                <div className="flex items-start justify-between gap-3">
                                                    <div className="min-w-0">
                                                        <div className="truncate text-sm font-bold text-white">{run.datasetName || 'Unnamed dataset'}</div>
                                                        <div className="mt-1 text-xs text-slate-500 truncate">
                                                            {run.csvName || 'No CSV recorded'}
                                                        </div>
                                                    </div>
                                                    <div className="flex flex-col items-end gap-2">
                                                        <span className={`shrink-0 rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.2em] ${
                                                            run.isActive
                                                                ? 'border-emerald-400/30 bg-emerald-500/10 text-emerald-100'
                                                                : 'border-slate-700 bg-slate-900 text-slate-300'
                                                        }`}>
                                                            {run.isActive ? 'active' : 'inactive'}
                                                        </span>
                                                        <span className={`shrink-0 rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.2em] ${
                                                            run.source === 'current'
                                                                ? 'border-amber-400/30 bg-amber-500/10 text-amber-100'
                                                                : 'border-slate-700 bg-slate-900 text-slate-300'
                                                        }`}>
                                                            {run.source}
                                                        </span>
                                                    </div>
                                                </div>

                                                <div className="mt-3 text-sm text-slate-200 font-semibold">
                                                    {getDetectorChoiceLabel(run.modelChoice, run.modelLabel)}
                                                </div>

                                                <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
                                                    <div className="rounded-lg border border-slate-800 bg-black/30 p-2">
                                                        <div className="text-slate-500 uppercase tracking-widest">mAP50</div>
                                                        <div className="mt-1 text-white font-bold">{formatPercent(run.metrics?.map50)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-slate-800 bg-black/30 p-2">
                                                        <div className="text-slate-500 uppercase tracking-widest">mAP75</div>
                                                        <div className="mt-1 text-white font-bold">{formatPercent(run.metrics?.map75)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-slate-800 bg-black/30 p-2">
                                                        <div className="text-slate-500 uppercase tracking-widest">mAP50-95</div>
                                                        <div className="mt-1 text-white font-bold">{formatPercent(run.metrics?.map50_95)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-slate-800 bg-black/30 p-2">
                                                        <div className="text-slate-500 uppercase tracking-widest">Fitness</div>
                                                        <div className="mt-1 text-white font-bold">{formatPercent(run.metrics?.fitness)}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-slate-800 bg-black/30 p-2">
                                                        <div className="text-slate-500 uppercase tracking-widest">Epochs</div>
                                                        <div className="mt-1 text-white font-bold">{run.epochs ?? 'n/a'}</div>
                                                    </div>
                                                    <div className="rounded-lg border border-slate-800 bg-black/30 p-2">
                                                        <div className="text-slate-500 uppercase tracking-widest">Plots</div>
                                                        <div className="mt-1 text-white font-bold">
                                                            {run.artifactSummary?.plotCount ?? 0}
                                                        </div>
                                                    </div>
                                                </div>

                                                <div className="mt-3 text-[11px] text-slate-500 leading-relaxed">
                                                    Trained {formatTrainingTimestamp(run.createdAt)}
                                                </div>
                                                {run.isActive && run.activatedAt && (
                                                    <div className="mt-1 text-[11px] text-emerald-300 leading-relaxed">
                                                        Active since {formatTrainingTimestamp(run.activatedAt)}
                                                    </div>
                                                )}

                                                <div className="mt-4 flex gap-2">
                                                    <button
                                                        onClick={() => handleLoadTrainingRun(run.summaryPath)}
                                                        disabled={runActionBusy}
                                                        className="flex-1 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-50 px-3 py-2 text-xs font-bold uppercase tracking-[0.18em] text-slate-100 transition"
                                                    >
                                                        View Stats
                                                    </button>
                                                    <button
                                                        onClick={() => handleActivateTrainingRun(run.summaryPath)}
                                                        disabled={runActionBusy || run.isActive}
                                                        className={`flex-1 rounded-lg px-3 py-2 text-xs font-bold uppercase tracking-[0.18em] transition ${
                                                            run.isActive
                                                                ? 'bg-emerald-500/10 text-emerald-200 border border-emerald-500/30 cursor-default'
                                                                : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                                                        } disabled:opacity-50`}
                                                    >
                                                        {run.isActive ? 'Active' : 'Set Active'}
                                                    </button>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>

                        <TrainingReviewPanel
                            summary={trainingSummary}
                            reviewManifest={testReviewManifest}
                            artifactManifest={trainingArtifactManifest}
                            trainingResultsRows={trainingResultsRows}
                        />
                    </div>
                </div>
            )
        },
        {
            id: 'tester',
            title: "Model Tester",
            subtitle: "Verify predictions",
            icon: <ScanEye size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="flex gap-4 items-center bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                        <button 
                            onClick={handleSelectImage} 
                            disabled={predicting || isRunning}
                            className="px-6 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition text-sm font-bold"
                        >Select Image</button>
                        <button 
                            onClick={handlePredict} 
                            disabled={!imagePath || predicting || isRunning}
                            className="px-6 py-2 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-rose-950/30 text-sm font-bold"
                        >Detect Characters</button>
                    </div>

                    {!imagePath ? (
                        <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-slate-800 rounded-xl text-slate-500 p-12 min-h-[300px]">
                            <Target size={48} className="mb-4 opacity-50" />
                            <p>Select a screenshot to begin analysis</p>
                        </div>
                    ) : (
                        <div className="grid flex-1 min-h-[300px] grid-cols-1 xl:grid-cols-[minmax(0,1.45fr)_380px] gap-6">
                            <div className="rounded-xl border-2 border-slate-700 bg-black min-h-[300px] flex items-center justify-center overflow-hidden p-6 py-12">
                                <div className="relative inline-block max-w-full max-h-full shadow-2xl">
                                    <img 
                                        src={`file://${imagePath}`} 
                                        alt="Input" 
                                        className="max-w-full max-h-[45vh] object-contain block filter brightness-110 contrast-125 saturate-150 rounded" 
                                    />
                                    {detections.map((detection, index) => {
                                        const chrome = getDetectionChrome(detection.class_key || detection.class_name);
                                        return (
                                            <div
                                                key={`${index}-${detection.class_id}-${detection.confidence}`}
                                                className={`absolute border-2 ${chrome.border} ${chrome.fill} ${chrome.glow} z-10 transition-all duration-500 pointer-events-none`}
                                                style={{
                                                    left: `${(detection.x_center - detection.width / 2) * 100}%`,
                                                    top: `${(detection.y_center - detection.height / 2) * 100}%`,
                                                    width: `${detection.width * 100}%`,
                                                    height: `${detection.height * 100}%`,
                                                }}
                                            >
                                                <div className={`absolute -top-6 left-0 px-2 py-0.5 ${chrome.badge} text-[10px] font-bold uppercase tracking-wider text-white rounded`}>
                                                    {detection.class_name} {(detection.confidence * 100).toFixed(0)}%
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>

                            <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-5 flex flex-col gap-5 min-h-[300px]">
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4">
                                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-rose-300">Enemies</div>
                                        <div className="mt-2 text-3xl font-black text-white">{predictionSummary.classSummary.enemy}</div>
                                    </div>
                                    <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4">
                                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-amber-200">Players</div>
                                        <div className="mt-2 text-3xl font-black text-white">{predictionSummary.classSummary.player}</div>
                                    </div>
                                </div>

                                <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
                                    <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400">Result</div>
                                    {hasPredictionResult ? (
                                        <div className="mt-3 space-y-2 text-sm text-slate-200">
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="text-slate-400">Detections</span>
                                                <span className="font-bold text-white">{predictionSummary.count}</span>
                                            </div>
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="text-slate-400">Top Class</span>
                                                <span className="font-bold text-white">{predictionSummary.topDetection?.class_name || 'None'}</span>
                                            </div>
                                            <div className="flex items-center justify-between gap-3">
                                                <span className="text-slate-400">Top Confidence</span>
                                                <span className="font-bold text-white">
                                                    {predictionSummary.topDetection ? `${(predictionSummary.topDetection.confidence * 100).toFixed(1)}%` : 'n/a'}
                                                </span>
                                            </div>
                                        </div>
                                    ) : (
                                        <p className="mt-3 text-sm text-slate-400 leading-relaxed">
                                            Run the tester to classify each visible character as Enemy or Player and draw the predicted bounding boxes directly on the screenshot.
                                        </p>
                                    )}
                                </div>

                                <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 flex flex-col gap-3 min-h-0 flex-1">
                                    <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400">
                                        <ImageIcon size={14} />
                                        Detection List
                                    </div>
                                    {detections.length ? (
                                        <div className="flex flex-col gap-2 overflow-y-auto pr-1">
                                            {detections.map((detection, index) => {
                                                const chrome = getDetectionChrome(detection.class_key || detection.class_name);
                                                return (
                                                    <div
                                                        key={`detection-card-${index}-${detection.class_id}-${detection.confidence}`}
                                                        className="rounded-lg border border-slate-700 bg-slate-950/80 p-3"
                                                    >
                                                        <div className="flex items-start justify-between gap-3">
                                                            <div>
                                                                <div className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white ${chrome.badge}`}>
                                                                    {detection.class_name}
                                                                </div>
                                                                <div className="mt-2 text-xs text-slate-400">
                                                                    Center: [{detection.x_center.toFixed(3)}, {detection.y_center.toFixed(3)}]
                                                                </div>
                                                                <div className="text-xs text-slate-500">
                                                                    Size: {detection.width.toFixed(3)} x {detection.height.toFixed(3)}
                                                                </div>
                                                            </div>
                                                            <div className="text-right">
                                                                <div className="text-lg font-black text-white">
                                                                    {(detection.confidence * 100).toFixed(0)}%
                                                                </div>
                                                                <div className="text-[11px] uppercase tracking-widest text-slate-500">
                                                                    confidence
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    ) : (
                                        <div className="flex-1 flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 text-center text-slate-500 p-6">
                                            <ImageIcon size={24} className="mb-3 opacity-60" />
                                            <p className="text-sm">
                                                {hasPredictionResult ? 'No detections passed the current confidence threshold.' : 'Pick an image and run detection to populate the list.'}
                                            </p>
                                        </div>
                                    )}
                                </div>

                                {activeTrainingRun && (
                                    <div className="rounded-xl border border-emerald-700/30 bg-emerald-500/10 p-3 text-xs text-emerald-50 leading-relaxed">
                                        <div className="font-bold uppercase tracking-[0.2em] text-emerald-200 mb-2">Active Model</div>
                                        <div className="text-sm font-semibold text-white">
                                            {getDetectorChoiceLabel(activeTrainingRun.modelChoice, activeTrainingRun.modelLabel)}
                                        </div>
                                        <div className="mt-1 text-emerald-100/80">{activeTrainingRun.datasetName}</div>
                                        <div className="mt-2 break-all text-emerald-100/70">
                                            {activeTrainingRun.stableBestModel || activeTrainingRun.summaryPath}
                                        </div>
                                    </div>
                                )}

                                {predictionSummary.modelPath && (
                                    <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-3 text-xs text-slate-400 leading-relaxed">
                                        <div className="font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">Prediction Used</div>
                                        <div className="break-all">{predictionSummary.modelPath}</div>
                                    </div>
                                )}

                                {predictionSummary.savedImagePath && (
                                    <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-3 text-xs text-slate-400 leading-relaxed">
                                        <div className="font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">Annotated Render</div>
                                        <div className="break-all">{predictionSummary.savedImagePath}</div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'video-tester',
            title: "Video Tester",
            subtitle: "Run YOLO over gameplay clips in real time",
            icon: <Gamepad2 size={24} />,
            content: (
                <VideoTesterWorkspace
                    appendLog={appendLog}
                    renderTerminal={renderTerminal}
                    isAppBusy={isRunning}
                    setAppBusy={setIsRunning}
                    isVideoPredictionActive={videoPredictionActive}
                    setVideoPredictionActive={setVideoPredictionActive}
                    activeTrainingRun={activeTrainingRun}
                />
            )
        },
        {
            id: 'reset',
            title: "Reset",
            subtitle: "Wipe all data",
            icon: <Trash2 size={22} />,
            content: (
                <div className="bg-red-900/10 p-6 rounded-xl border border-red-700/50 flex flex-col gap-4">
                    <button 
                        onClick={() => window.confirm("Reset everything?") && handleRunStep('reset_project.py', ['--all', '--yes'])}
                        className="px-6 py-2 bg-red-600 hover:bg-red-500 rounded-lg transition text-white font-bold"
                    >FACTORY RESET</button>
                </div>
            )
        }
    ];

    return (
        <div className="flex flex-col h-screen w-full bg-stone-950 text-stone-100 overflow-hidden font-sans selection:bg-rose-500/30 border border-rose-950">
            <TitleBar />
            
            {/* Header/Progress */}
            <div className="h-1.5 bg-slate-800 w-full relative z-40 no-drag">
                <div 
                    className="h-full bg-gradient-to-r from-rose-600 via-red-500 to-orange-400 transition-all duration-500 ease-out"
                    style={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
                />
            </div>

            <div className={`flex-1 flex flex-col md:flex-row w-full ${isCollectorWorkspaceActive ? 'p-2 md:p-3 lg:p-4 gap-4' : 'p-4 md:p-6 lg:p-8 gap-6'} h-full overflow-hidden`}>
                
                {/* Sidebar Navigation */}
                {!isCollectorWorkspaceActive && <div className="hidden md:flex flex-col w-64 space-y-2 pr-4 border-r border-slate-800/50 no-drag">
                    <div className="mb-6 px-4">
                        <h1 className="text-lg font-black tracking-tight text-white mb-1">DSAI Dashboard</h1>
                        <p className="text-xs font-mono text-slate-500">v1.1.0-alpha</p>
                    </div>
                    {slides.map((slide, index) => (
                        <button
                            key={slide.id}
                            onClick={() => setCurrentSlide(index)}
                            className={`text-left px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-3
                                ${currentSlide === index 
                                    ? 'bg-rose-500/10 text-rose-300 border border-rose-500/20' 
                                    : 'text-slate-500 hover:bg-slate-900 hover:text-slate-300 border border-transparent'
                                }`}
                        >
                            {slide.icon}
                            {slide.title}
                        </button>
                    ))}
                </div>}

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col relative bg-slate-950/50 rounded-2xl md:border border-slate-800/50 shadow-2xl overflow-hidden h-full no-drag">
                    <div className={`${isCollectorWorkspaceActive ? 'p-5 md:p-6 pb-0' : 'p-8 pb-0'} animate-fade-in-down border-b border-transparent`}>
                        <div className="flex items-center gap-3 text-rose-400 mb-2">
                            {slides[currentSlide].icon}
                            <span className="text-xs font-mono uppercase tracking-widest opacity-75">Control Surface</span>
                        </div>
                        <h2 className={`${isCollectorWorkspaceActive ? 'text-2xl md:text-3xl' : 'text-3xl'} font-bold text-white`}>{slides[currentSlide].title}</h2>
                        <p className="text-slate-400 mt-1">{slides[currentSlide].subtitle}</p>
                    </div>

                    <div className={`flex-1 ${isCollectorWorkspaceActive ? 'p-4 md:p-5 lg:p-6' : 'p-8'} overflow-y-auto w-full max-h-full min-h-0`}>
                        <div className="h-full animate-fade-in">
                            {slides[currentSlide].content}
                        </div>
                    </div>
                </div>
            </div>
            
            <div className="fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black pointer-events-none" />
        </div>
    );
};

export default Presentation;
