import { useState, useRef, useEffect } from 'react';
import { X, Minus, Square, Copy, Target, Gamepad2, MonitorPlay, Database, BrainCircuit, BarChart3, ChevronRight, ChevronLeft, ScanEye, Terminal, Layers, Cpu, CheckCircle, CloudDownload, Trash2 } from 'lucide-react';
import './index.css';

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

const Presentation = () => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [logs, setLogs] = useState([]);
    const [isRunning, setIsRunning] = useState(false);
    const [videoLinks, setVideoLinks] = useState('');
    const logsEndRef = useRef(null);

    // Predictor State
    const [imagePath, setImagePath] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [truth, setTruth] = useState(null);
    const [predicting, setPredicting] = useState(false);
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

    // Data Collector State
    const [videos, setVideos] = useState([]);
    const [selectedVideo, setSelectedVideo] = useState('');
    const [collectionName, setCollectionName] = useState('');
    const [augmentationName, setAugmentationName] = useState('');

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
            setSelectedVideo(data[0]);
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
        setLogs([`> Starting data collection for video: ${videoName}...`, `> Dataset Name: ${finalName}`, `> Mode: Automated (Auto-Skip Low Confidence)`]);
        const result = await window.electronAPI.runDataCollection(videoName, finalName);
        if (result.success) {
            setLogs(prev => [...prev, `\n[Success] Data collection complete! Frames saved to data_sets/${finalName}`, `[System] You can now analyze this in the Dataset Manager.`]);
            handleFetchDatasets();
        } else {
            setLogs(prev => [...prev, `\n[Error] ${result.error}`]);
        }
        setIsRunning(false);
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
            setPrediction(null);
            setTruth(null);
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
                if (result.saved_image_path) {
                    setImagePath(result.saved_image_path);
                    setPrediction(null); // Dot is already stamped on the image
                } else {
                    setPrediction(result.prediction);
                }
                setTruth(result.truth);
                setLogs(prev => [...prev, `[Success] Found target at [${result.prediction[0].toFixed(3)}, ${result.prediction[1].toFixed(3)}]`]);
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
        setLogs([`> Starting training on dataset: ${selectedDataset.name}`, `> CSV: ${selectedCsv}`, `> Epochs: ${trainEpochs}`]);
        const result = await window.electronAPI.runTraining(selectedDataset.path, selectedCsv, trainEpochs);
        if (result.success) {
            setLogs(prev => [...prev, `\n[Success] Training completed successfully!`]);
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

    const renderTerminal = () => (
        <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-2xl mt-6 flex flex-col h-64 font-mono text-sm max-h-96">
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
            subtitle: "Select and process gameplay videos",
            icon: <MonitorPlay size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="flex flex-col gap-2">
                            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Select Video Source</label>
                            <div className="flex gap-4">
                                <select 
                                    className="flex-1 bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                                    value={selectedVideo}
                                    onChange={(e) => setSelectedVideo(e.target.value)}
                                >
                                    {videos.length === 0 && <option>No videos found in src/videos</option>}
                                    {videos.map(v => <option key={v} value={v}>{v}</option>)}
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

                        <button 
                            onClick={() => handleRunDataCollection(selectedVideo)}
                            disabled={isRunning || !selectedVideo}
                            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-blue-900/20 text-sm font-bold flex justify-center items-center gap-2"
                        >
                            Start Data Collection <MonitorPlay size={18}/>
                        </button>
                    </div>
                    {renderTerminal()}
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
                                <li>• Automatic coordinate transformation for rotates/flips</li>
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
            id: 'train',
            title: "Model Training",
            subtitle: "Train on selected dataset",
            icon: <BrainCircuit size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                        <div className="flex flex-col gap-4">
                            <div className="flex items-center gap-4 text-sm">
                                <span className="text-rose-400 font-bold whitespace-nowrap">Training Source:</span>
                                <div className="flex-1 px-3 py-1.5 bg-slate-900 rounded border border-slate-700 text-slate-300 font-mono text-xs overflow-hidden text-ellipsis italic">
                                    {selectedDataset ? `${selectedDataset.name} -> ${selectedCsv}` : "None Selected (Go to Dataset Manager)"}
                                </div>
                            </div>
                            
                            <div className="flex items-center gap-6">
                                <div className="flex flex-col gap-1.5 flex-1">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Epochs Count</label>
                                    <input 
                                        type="number" 
                                        value={trainEpochs} 
                                        onChange={(e) => setTrainEpochs(parseInt(e.target.value))}
                                        className="bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
                                    />
                                </div>
                                <div className="flex flex-col gap-1.5 flex-1 opacity-50">
                                    <label className="text-xs font-bold text-slate-500 uppercase tracking-widest">Batch Size</label>
                                    <input type="number" value="16" readOnly className="bg-slate-950 border border-slate-800 rounded-lg p-2 text-sm text-slate-500 cursor-not-allowed" />
                                </div>
                            </div>
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
                        >Detect Enemy</button>
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
                                {prediction && (
                                    <div 
                                        className="absolute w-9 h-9 border-2 border-red-500/80 bg-red-500/10 rounded-full shadow-[0_0_20px_#ef4444] z-10 transition-all duration-500 pointer-events-none"
                                        style={{ left: `${prediction[0] * 100}%`, top: `${prediction[1] * 100}%`, transform: 'translate(-50%, -50%)' }}
                                    >
                                        <div className="absolute top-1/2 left-1/2 w-1.5 h-1.5 bg-white rounded-full -translate-x-1/2 -translate-y-1/2 shadow-lg shadow-black group-hover:bg-red-200"></div>
                                        <div className="absolute top-1/2 left-1/2 w-full h-[1px] bg-red-500/50 -translate-x-1/2 -translate-y-1/2"></div>
                                        <div className="absolute top-1/2 left-1/2 w-[1px] h-full bg-red-500/50 -translate-x-1/2 -translate-y-1/2"></div>
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
        <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans selection:bg-blue-500/30 drag border border-slate-800">
            <TitleBar />
            
            {/* Header/Progress */}
            <div className="h-1.5 bg-slate-800 w-full relative z-40 no-drag">
                <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-500 ease-out"
                    style={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
                />
            </div>

            <div className="flex-1 flex flex-col md:flex-row max-w-7xl mx-auto w-full p-4 md:p-8 gap-6 h-full overflow-hidden">
                
                {/* Sidebar Navigation */}
                <div className="hidden md:flex flex-col w-64 space-y-2 pr-4 border-r border-slate-800/50 no-drag">
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
                </div>

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col relative bg-slate-950/50 rounded-2xl md:border border-slate-800/50 shadow-2xl overflow-hidden h-full no-drag">
                    <div className="p-8 pb-0 animate-fade-in-down border-b border-transparent">
                        <div className="flex items-center gap-3 text-blue-400 mb-2">
                            {slides[currentSlide].icon}
                            <span className="text-xs font-mono uppercase tracking-widest opacity-75">Control Surface</span>
                        </div>
                        <h2 className="text-3xl font-bold text-white">{slides[currentSlide].title}</h2>
                        <p className="text-slate-400 mt-1">{slides[currentSlide].subtitle}</p>
                    </div>

                    <div className="flex-1 p-8 overflow-y-auto w-full max-h-full min-h-0">
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
