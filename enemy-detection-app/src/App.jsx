import { useState, useRef, useEffect } from 'react';
import { X, Minus, Square, Copy, Target, Gamepad2, MonitorPlay, Database, BrainCircuit, BarChart3, ChevronRight, ChevronLeft, ScanEye, Terminal, Layers, Cpu, CheckCircle, CloudDownload, Trash2, Image as ImageIcon } from 'lucide-react';
import './index.css';
import CollectorWorkspace from './CollectorWorkspace';
import DatasetViewerWorkspace from './DatasetViewerWorkspace';

const TitleBar = () => (
    <div className="h-8 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-4 drag select-none">
        <div className="flex items-center gap-2">
            <ScanEye size={16} className="text-blue-500" />
            <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Enemy Detection Hub</span>
        </div>
        <div className="flex items-center no-drag">
            <button 
                onClick={() => window.electronAPI.minimizeWindow()}
                className="p-2 hover:bg-slate-800 text-slate-400 transition"
            >
                <Minus size={14} />
            </button>
            <button 
                onClick={() => window.electronAPI.maximizeWindow()}
                className="p-2 hover:bg-slate-800 text-slate-400 transition"
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

const getDetectionChrome = (className) => {
    const normalized = String(className || '').toLowerCase();
    if (normalized === 'player') {
        return {
            border: 'border-sky-400/90',
            fill: 'bg-sky-500/10',
            glow: 'shadow-[0_0_20px_rgba(14,165,233,0.35)]',
            badge: 'bg-sky-500/90',
        };
    }
    return {
        border: 'border-rose-500/90',
        fill: 'bg-rose-500/10',
        glow: 'shadow-[0_0_20px_rgba(244,63,94,0.35)]',
        badge: 'bg-rose-500/90',
    };
};

const Presentation = () => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [logs, setLogs] = useState([]);
    const [isRunning, setIsRunning] = useState(false);
    const [videoLinks, setVideoLinks] = useState('');
    const logsEndRef = useRef(null);

    // Predictor State
    const [imagePath, setImagePath] = useState(null);
    const [detections, setDetections] = useState([]);
    const [truth, setTruth] = useState(null);
    const [predicting, setPredicting] = useState(false);
    const [showDetectionOverlays, setShowDetectionOverlays] = useState(false);
    const imgRef = useRef(null);
    const [imgDims, setImgDims] = useState({ width: 0, height: 0 });

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
    const [trainDeviceMode, setTrainDeviceMode] = useState('cuda');
    const [trainModel, setTrainModel] = useState('yolov8n');

    // Data Collector State
    const [videos, setVideos] = useState([]);
    const [selectedVideo, setSelectedVideo] = useState('');
    const [collectionName, setCollectionName] = useState('');
    const [augmentationName, setAugmentationName] = useState('');
    const [collectorSession, setCollectorSession] = useState(null);
    const isCollectorWorkspaceActive = currentSlide === 1 && Boolean(collectorSession);

    const appendLog = (...entries) => {
        const nextEntries = entries.flat().filter(Boolean);
        if (nextEntries.length === 0) return;
        setLogs(prev => [...prev, ...nextEntries]);
    };

    const handleFetchDatasets = async () => {
        const data = await window.electronAPI.listDatasets();
        setDatasets(data);
        if (data.length > 0 && !selectedDataset) {
            setSelectedDataset(data[0]);
            setSelectedCsv(data[0].csvs[0]);
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
                setLogs(prev => [...prev, data.msg]);
            });
        }
        handleFetchDatasets();
        handleFetchVideos();
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
            appendLog(`[Hint] Press 1 for enemy, 2 for player, Shift+Enter to add another box on the same frame, and Enter to save and move on.`);
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

    const handleCancel = async () => {
        if (!isRunning) return;
        setLogs(prev => [...prev, '\n[System] Aborting current pipeline process via KILL signal...']);
        await window.electronAPI.cancelPipeline();
        setIsRunning(false);
    };

    const handleSelectImage = async () => {
        const path = await window.electronAPI.selectImage();
        if (path) {
            setImagePath(path);
            setDetections([]);
            setTruth(null);
            setShowDetectionOverlays(false);
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
                const nextDetections = result.detections || [];
                if (result.saved_image_path) {
                    setImagePath(result.saved_image_path);
                    setShowDetectionOverlays(false);
                } else {
                    setShowDetectionOverlays(true);
                }
                setDetections(nextDetections);
                setTruth(result.truth);
                if (result.top_detection) {
                    const top = result.top_detection;
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
        setIsRunning(true);
        setLogs([
            `> Starting training on dataset: ${selectedDataset.name}`,
            `> CSV: ${selectedCsv}`,
            `> Model Basis: ${trainModel}`,
            `> Epochs: ${trainEpochs}`,
            `> Batch Size: ${trainBatchSize}`,
            `> Image Size: ${trainImageSize}`,
            `> Device Mode: ${trainDeviceMode === 'cuda' ? 'CUDA / NVIDIA GPU only' : trainDeviceMode === 'auto' ? 'Auto (prefer CUDA)' : 'CPU only'}`,
        ]);
        const result = await window.electronAPI.runTraining({
            datasetPath: selectedDataset.path,
            csvName: selectedCsv,
            epochs: trainEpochs,
            batchSize: trainBatchSize,
            imageSize: trainImageSize,
            deviceMode: trainDeviceMode,
            modelChoice: trainModel,
        });
        if (result.success) {
            setLogs(prev => [...prev, `\n[Success] Training completed successfully with ${trainModel}.`]);
        } else {
            setLogs(prev => [...prev, `\n[Error] ${result.error}`]);
        }
        setIsRunning(false);
    };

    const updateImageDims = () => {
        if (imgRef.current) {
            setImgDims({
                width: imgRef.current.clientWidth,
                height: imgRef.current.clientHeight
            });
        }
    };

    useEffect(() => {
        window.addEventListener('resize', updateImageDims);
        return () => window.removeEventListener('resize', updateImageDims);
    }, []);

    const renderTerminal = (compact = false) => (
        <div className={`bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col font-mono text-sm ${compact ? 'mt-4 h-40 max-h-40' : 'mt-6 h-64 max-h-96'}`}>
            <div className="bg-slate-900 border-b border-slate-800 px-4 py-2 flex items-center justify-between">
                <div className="flex gap-2.5 items-center">
                    <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
                    <span className="ml-2 text-slate-500 text-xs tracking-wider">TERMINAL</span>
                </div>
                {isRunning && (
                    <div className="flex items-center gap-4">
                        <button 
                            onClick={handleCancel} 
                            className="px-3 py-1 bg-red-600/20 text-red-500 hover:bg-red-600/40 rounded border border-red-500/30 text-xs font-bold uppercase tracking-wider transition-colors shadow-lg shadow-red-900/20"
                        >Force Stop</button>
                        <span className="text-emerald-400 text-xs animate-pulse">Running...</span>
                    </div>
                )}
            </div>
            <div className="p-4 overflow-y-auto flex-1 text-slate-300 leading-relaxed break-all whitespace-pre-wrap">
                {logs.length === 0 ? (
                    <span className="text-slate-600 italic">No output yet. Run a process to see logs here.</span>
                ) : (
                    logs.map((log, i) => (
                        <div key={i}>{log}</div>
                    ))
                )}
                <div ref={logsEndRef} />
            </div>
        </div>
    );

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
                            className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-300 font-mono focus:outline-none focus:border-sky-500 transition-colors resize-none"
                        />
                        <button 
                            onClick={handleDownloadStep}
                            disabled={isRunning || !videoLinks.trim()}
                            className="px-6 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 rounded-lg transition text-white font-bold"
                        >Start Download</button>
                    </div>
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
                                                className="flex-1 bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                            className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                                            value={collectionName}
                                            onChange={(e) => setCollectionName(e.target.value)}
                                        />
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-4">
                                            <div className="text-[11px] font-black uppercase tracking-[0.24em] text-sky-400 mb-2">Manual Flow</div>
                                            <div className="text-sm text-slate-300 space-y-2">
                                                <p>Open the annotator inside the Electron app.</p>
                                                <p>Choose <span className="text-white font-semibold">enemy</span> for opponents and <span className="text-white font-semibold">player</span> for your own character.</p>
                                                <p>Drag a bounding box around the selected target, or single-click for a quick centered square box.</p>
                                                <p>Save useful frames and stack multiple labeled characters on the same frame without leaving the app.</p>
                                            </div>
                                        </div>
                                        <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-4">
                                            <div className="text-[11px] font-black uppercase tracking-[0.24em] text-emerald-400 mb-2">Quick Controls</div>
                                            <div className="text-sm text-slate-300 space-y-2">
                                                <p><span className="text-white font-semibold">1</span> selects enemy and <span className="text-white font-semibold">2</span> selects player.</p>
                                                <p><span className="text-white font-semibold">Shift + Enter</span> adds another labeled box on the same frame, and <span className="text-white font-semibold">Enter</span> saves and advances.</p>
                                                <p><span className="text-white font-semibold">Arrow keys</span> move frame by frame, and <span className="text-white font-semibold">Shift + Arrows</span> jumps further.</p>
                                                <p><span className="text-white font-semibold">D</span> duplicates the last saved box and <span className="text-white font-semibold">S</span> skips the current frame.</p>
                                            </div>
                                        </div>
                                    </div>

                                    <button 
                                        onClick={() => handleRunDataCollection(selectedVideo)}
                                        disabled={isRunning || !selectedVideo}
                                        className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-blue-900/20 text-sm font-bold flex justify-center items-center gap-2"
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
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                className="flex-1 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition font-bold text-sm text-white flex items-center justify-center gap-2 shadow-lg shadow-blue-900/20"
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
                            <div className="absolute top-4 left-4 z-10 bg-blue-600/80 backdrop-blur px-3 py-1 rounded text-[10px] font-black uppercase tracking-widest text-white border border-blue-400/30">Neutralized & Balanced</div>
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
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
                                value={augmentationName}
                                onChange={(e) => setAugmentationName(e.target.value)}
                            />
                        </div>

                        <div className="bg-blue-900/20 border border-blue-800/50 p-4 rounded-lg">
                            <h4 className="text-blue-400 text-xs font-bold uppercase tracking-widest mb-2">Augmentation Strategy</h4>
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
                            className="px-6 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-indigo-900/20 text-sm font-bold flex justify-center items-center gap-2"
                        >
                            Run Augmentation Pipeline <Layers size={18}/>
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
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                    className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
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
                                        {datasets.map(ds => <option key={ds.name} value={ds.name}>{ds.name}</option>)}
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
                            
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                            Training now uses the selected detector basis directly from the app, passes your batch size and image size through to Python, and lets the trainer create its own train/validation split from the chosen CSV.
                            That avoids leaking the same cleaned images into both train and validation, while making CUDA usage explicit instead of hidden.
                        </div>

                        <button 
                            onClick={handleTrainOnDataset}
                            disabled={isRunning || !selectedDataset}
                            className="px-6 py-3 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-rose-900/20 text-sm font-bold w-full flex justify-center items-center gap-2"
                        >Start Training <BrainCircuit size={18}/></button>
                    </div>
                    {renderTerminal()}
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
                            className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-blue-900/20 text-sm font-bold"
                        >Detect Characters</button>
                    </div>

                    {!imagePath ? (
                        <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-slate-800 rounded-xl text-slate-500 p-12 min-h-[300px]">
                            <Target size={48} className="mb-4 opacity-50" />
                            <p>Select a screenshot to begin analysis</p>
                        </div>
                    ) : (
                        <div className="rounded-xl border-2 border-slate-700 bg-black flex-1 min-h-[300px] flex items-center justify-center overflow-hidden p-6 py-12">
                            <div className="relative inline-block max-w-full max-h-full shadow-2xl">
                                <img 
                                    src={`file://${imagePath}`} 
                                    alt="Input" 
                                    className="max-w-full max-h-[45vh] object-contain block filter brightness-110 contrast-125 saturate-150 rounded" 
                                />
                                {showDetectionOverlays && detections.map((detection, index) => {
                                    const chrome = getDetectionChrome(detection.class_name);
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
                    )}
                    {renderTerminal()}
                </div>
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
        <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans selection:bg-blue-500/30 border border-slate-800">
            <TitleBar />
            
            {/* Header/Progress */}
            <div className="h-1.5 bg-slate-800 w-full relative z-40 no-drag">
                <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-500 ease-out"
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
                                    ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' 
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
                        <div className="flex items-center gap-3 text-blue-400 mb-2">
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
