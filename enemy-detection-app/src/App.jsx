import { useState, useRef, useEffect } from 'react';
import { Target, Gamepad2, MonitorPlay, Database, BrainCircuit, BarChart3, ChevronRight, ChevronLeft, ScanEye, Terminal, Layers, Cpu, CheckCircle, CloudDownload, Trash2 } from 'lucide-react';
import './index.css';

const Presentation = () => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [logs, setLogs] = useState([]);
    const [isRunning, setIsRunning] = useState(false);
    const [videoLinks, setVideoLinks] = useState('');
    const logsEndRef = useRef(null);

    const handleDownloadStep = async () => {
        if (isRunning) return;
        if (videoLinks.trim()) {
            await window.electronAPI.saveLinks(videoLinks);
        }
        handleRunStep('download_videos.py', []);
    };

    // Predictor State
    const [imagePath, setImagePath] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [truth, setTruth] = useState(null);
    const [predicting, setPredicting] = useState(false);
    const imgRef = useRef(null);
    const [imgDims, setImgDims] = useState({ width: 0, height: 0 });

    useEffect(() => {
        if (window.electronAPI?.onPipelineOutput) {
            window.electronAPI.onPipelineOutput((data) => {
                setLogs(prev => [...prev, data.msg]);
            });
        }
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
            title: "Video Downloader",
            subtitle: "Fetch raw YouTube footage",
            icon: <CloudDownload size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-4">
                        <div>
                            <h3 className="text-xl font-bold text-sky-400 mb-1">download_videos.py</h3>
                            <p className="text-slate-300 text-sm leading-relaxed">
                                Paste YouTube URLs below (one per line). The downloader will intelligently skip videos that already exist.
                            </p>
                        </div>
                        <textarea
                            value={videoLinks}
                            onChange={(e) => setVideoLinks(e.target.value)}
                            placeholder="https://youtube.com/watch?v=..."
                            className="w-full h-32 bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-300 font-mono focus:outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500 transition-colors resize-none"
                        />
                        <div className="flex gap-2">
                            <button 
                                onClick={handleDownloadStep}
                                disabled={isRunning || !videoLinks.trim()}
                                className="flex-1 px-6 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-sky-900/20 text-sm font-bold"
                            >Start Video Downloader</button>
                            <button
                                onClick={async () => {
                                    if (window.confirm("Are you sure you want to delete all downloaded videos?")) {
                                        await window.electronAPI.clearVideos();
                                        setLogs(prev => [...prev, "\n[System] Videos cleared successfully!"]);
                                    }
                                }}
                                disabled={isRunning}
                                className="px-6 py-2 bg-red-600/10 hover:bg-red-600/30 text-red-500 font-bold rounded-lg transition text-sm border border-red-500/30"
                            >Delete Videos</button>
                        </div>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'tester',
            title: "Model Tester",
            subtitle: "Test current ONNX locally",
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
                                    className="max-w-full max-h-[50vh] object-contain block filter brightness-110 contrast-125 saturate-150 rounded" 
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
                                {truth && (
                                    <div 
                                        className="absolute w-3 h-3 bg-emerald-500 rounded-full shadow-[0_0_15px_#10b981] z-10 transition-all duration-500 pointer-events-none"
                                        style={{ left: `${truth[0] * 100}%`, top: `${truth[1] * 100}%`, transform: 'translate(-50%, -50%)' }}
                                    />
                                )}
                            </div>
                        </div>
                    )}
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'collect',
            title: "Data Collection",
            subtitle: "Extract screens from raw gameplay",
            icon: <MonitorPlay size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-bold text-blue-400 mb-2">process_video_improved.py</h3>
                        <p className="text-slate-300 leading-relaxed mb-6">
                            Runs object tracking and saves raw screenshots to dataset/labeled alongside initial coordinate data.
                        </p>
                        <button 
                            onClick={() => handleRunStep('process_video_improved.py', ['--videos_dir', 'src/videos', '--output_dir', 'dataset/labeled', '--auto_skip'])}
                            disabled={isRunning}
                            className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg text-sm font-bold w-full"
                        >Start Collection Pipeline (Auto-Skip)</button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'augment',
            title: "Augmentation",
            subtitle: "Expand dataset and fix biases",
            icon: <Layers size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-bold text-emerald-400 mb-2">augment_dataset_improved.py</h3>
                        <p className="text-slate-300 leading-relaxed mb-6">
                            Applies random augmentations (flips, translations) to dramatically multiply the available training data and smooth distribution.
                        </p>
                        <button 
                            onClick={() => handleRunStep('augment_dataset_improved.py', ['--input_csv', 'dataset/labeled/labels_enhanced.csv', '--input_dir', 'dataset/labeled/images', '--output_dir', 'dataset/augmented'])}
                            disabled={isRunning}
                            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg text-sm font-bold w-full"
                        >Process Data Augmentation</button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'split',
            title: "Split & Analyze",
            subtitle: "Train/Val Separation & Visualizing",
            icon: <BarChart3 size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                            <h3 className="text-lg font-bold text-yellow-500 mb-2">visualize_dataset.py</h3>
                            <button 
                                onClick={() => handleRunStep('visualize_dataset.py', ['--csv', 'dataset/augmented/augmented_labels.csv', '--output', 'dataset/augmented/center_bias_heatmap.png'])}
                                disabled={isRunning}
                                className="px-4 py-2 bg-yellow-600/20 text-yellow-500 hover:bg-yellow-600/30 border border-yellow-600/50 disabled:opacity-50 rounded-lg transition text-sm font-bold w-full"
                            >Generate Bias Heatmap</button>
                        </div>
                        <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                            <h3 className="text-lg font-bold text-purple-400 mb-2">split_dataset.py</h3>
                            <button 
                                onClick={() => handleRunStep('split_dataset.py', ['--csv', 'dataset/augmented/augmented_labels.csv', '--img_dir', 'dataset/augmented/images', '--output_dir', 'dataset/final', '--val_ratio', '0.2', '--stratified'])}
                                disabled={isRunning}
                                className="px-4 py-2 bg-purple-600/20 text-purple-400 hover:bg-purple-600/30 border border-purple-600/50 disabled:opacity-50 rounded-lg transition text-sm font-bold w-full"
                            >Run Stratified Split</button>
                        </div>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'train',
            title: "Training",
            subtitle: "Build the ResNet CNN",
            icon: <BrainCircuit size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700">
                        <h3 className="text-xl font-bold text-rose-500 mb-2">train.py</h3>
                        <p className="text-slate-300 leading-relaxed mb-6">
                            Run PyTorch to train the architecture using MobileNet/ResNet backbone with MSE loss optimized by Adam.
                        </p>
                        <button 
                            onClick={() => handleRunStep('train.py', [
                                '--train_csv', 'src/fn-dataset/train/labels.csv', 
                                '--train_dir', 'src/fn-dataset/train/images', 
                                '--val_csv', 'src/fn-dataset/valid/labels.csv', 
                                '--val_dir', 'src/fn-dataset/valid/images',
                                '--epochs', '30', '--batch_size', '16', '--lr', '1e-4'
                            ])}
                            disabled={isRunning}
                            className="px-6 py-2 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg text-sm font-bold w-full flex justify-center items-center gap-2"
                        >Start Model Training <BrainCircuit size={18}/></button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'pipeline',
            title: "Full Pipeline",
            subtitle: "One-Click Automation",
            icon: <Terminal size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 bg-[linear-gradient(45deg,transparent_25%,rgba(59,130,246,0.05)_50%,transparent_75%,transparent_100%)] bg-[length:20px_20px]">
                        <h3 className="text-xl font-bold text-white mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">run_pipeline.py</h3>
                        <p className="text-slate-300 leading-relaxed mb-6">
                            Execute the entire E2E engine. This script triggers every stage automatically.
                        </p>
                        <button 
                            onClick={() => handleRunStep('run_pipeline.py', ['--videos_dir', 'src/videos', '--auto_skip', '--epochs', '30'])}
                            disabled={isRunning}
                            className="px-6 py-2 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 disabled:opacity-50 rounded-lg transition text-white shadow-xl shadow-blue-900/20 text-sm font-bold w-full"
                        >Execute Master Pipeline</button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        },
        {
            id: 'reset',
            title: "Factory Reset",
            subtitle: "Wipe models and data",
            icon: <Trash2 size={24} />,
            content: (
                <div className="flex flex-col h-full space-y-6">
                    <div className="bg-red-900/10 p-6 rounded-xl border border-red-700/50 flex flex-col gap-4">
                        <div>
                            <h3 className="text-xl font-bold text-red-500 mb-1">DANGER ZONE</h3>
                            <p className="text-red-300/80 text-sm leading-relaxed">
                                This will run <code>reset_project.py</code> to clear your dataset, models, and caches. By default, timestamped backups will be created.
                            </p>
                        </div>
                        <button 
                            onClick={() => {
                                if (window.confirm("Are you absolutely sure you want to reset the project data?")) {
                                    handleRunStep('reset_project.py', ['--all', '--yes']);
                                }
                            }}
                            disabled={isRunning}
                            className="px-6 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-red-900/20 text-sm font-bold w-full uppercase tracking-wider"
                        >Execute Project Reset</button>
                    </div>
                    {renderTerminal()}
                </div>
            )
        }
    ];

    return (
        <div className="flex flex-col h-screen w-full bg-slate-950 text-slate-200 overflow-hidden font-sans selection:bg-blue-500/30">
            {/* Header/Progress */}
            <div className="h-1.5 bg-slate-800 w-full relative z-50">
                <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-emerald-400 transition-all duration-500 ease-out"
                    style={{ width: `${((currentSlide + 1) / slides.length) * 100}%` }}
                />
            </div>

            <div className="flex-1 flex flex-col md:flex-row max-w-7xl mx-auto w-full p-4 md:p-8 gap-6 h-full">
                
                {/* Sidebar Navigation */}
                <div className="hidden md:flex flex-col w-64 space-y-2 pr-4 border-r border-slate-800/50">
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
                <div className="flex-1 flex flex-col relative bg-slate-950/50 rounded-2xl md:border border-slate-800/50 shadow-2xl overflow-hidden h-full">
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
