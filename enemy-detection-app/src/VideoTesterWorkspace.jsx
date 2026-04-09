import { useEffect, useRef, useState } from 'react';
import { Activity, Film, Pause, Play, Square } from 'lucide-react';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const toFileUrl = (filePath) => {
    const normalized = String(filePath || '').replace(/\\/g, '/');
    return encodeURI(/^[A-Za-z]:\//.test(normalized) ? `file:///${normalized}` : `file://${normalized}`);
};

const formatTime = (seconds) => {
    const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    const minutes = Math.floor(safe / 60);
    const remainder = Math.floor(safe % 60);
    return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
};

const basename = (filePath) => String(filePath || '').split(/[\\/]/).pop() || '';

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
    const summary = { enemy: 0, player: 0, unknown: 0 };
    (Array.isArray(detections) ? detections : []).forEach((detection) => {
        const key = resolveDetectionClassKey(detection);
        if (Object.prototype.hasOwnProperty.call(summary, key)) {
            summary[key] += 1;
        } else {
            summary.unknown += 1;
        }
    });
    return summary;
};

const normalizeDetections = (detections) => (Array.isArray(detections) ? detections : []).map((detection) => {
    const class_key = resolveDetectionClassKey(detection);
    return {
        ...detection,
        class_key,
        class_name: formatDetectionClassName({ ...detection, class_key }),
    };
});

const createEmptyMeta = () => ({
    fps: 0,
    frameCount: 0,
    duration: 0,
    width: 0,
    height: 0,
    modelPath: null,
    device: null,
    deviceName: null,
    sourcePath: null,
});

const createEmptyProgress = () => ({
    processed: 0,
    total: 0,
    percent: 0,
});

const statusLabels = {
    idle: 'Idle',
    starting: 'Starting',
    processing: 'Processing',
    ready: 'Ready',
    complete: 'Complete',
    stopping: 'Stopping',
    stopped: 'Stopped',
    error: 'Error',
};

export default function VideoTesterWorkspace({
    appendLog,
    renderTerminal,
    isAppBusy,
    setAppBusy,
    isVideoPredictionActive,
    setVideoPredictionActive,
    activeTrainingRun,
}) {
    const videoRef = useRef(null);
    const animationRef = useRef(null);
    const sessionIdRef = useRef(null);
    const frameDetectionsRef = useRef(new Map());
    const currentFrameIndexRef = useRef(0);
    const appendLogRef = useRef(appendLog);
    const processingModeRef = useRef('precompute');
    const videoMetaRef = useRef(createEmptyMeta());
    const videoPathRef = useRef('');
    const videoPredictionActiveRef = useRef(isVideoPredictionActive);
    const autoplayPendingRef = useRef(false);

    const [videoPath, setVideoPath] = useState('');
    const [predictionMode, setPredictionMode] = useState('precompute');
    const [processingState, setProcessingState] = useState('idle');
    const [processingProgress, setProcessingProgress] = useState(createEmptyProgress);
    const [videoMeta, setVideoMeta] = useState(createEmptyMeta);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
    const [currentDetections, setCurrentDetections] = useState([]);
    const [isPlaying, setIsPlaying] = useState(false);
    const [lastError, setLastError] = useState('');
    const [lastUpdatedFrame, setLastUpdatedFrame] = useState(null);

    appendLogRef.current = appendLog;
    processingModeRef.current = predictionMode;
    videoMetaRef.current = videoMeta;
    videoPathRef.current = videoPath;
    videoPredictionActiveRef.current = isVideoPredictionActive;

    const currentClassSummary = summarizeDetections(currentDetections);
    const selectedVideoName = basename(videoPath);
    const derivedFrameTotal = videoMeta.frameCount || (videoMeta.duration > 0 && videoMeta.fps > 0 ? Math.max(1, Math.round(videoMeta.duration * videoMeta.fps)) : 0);
    const playbackDisabled = !videoPath || (predictionMode === 'precompute' && processingState !== 'ready' && processingState !== 'complete');
    const busyWithOtherProcess = isAppBusy && !isVideoPredictionActive;
    const modeDescription = predictionMode === 'precompute'
        ? 'Process the whole file first, then play it back with smooth synchronized boxes.'
        : 'Start playback immediately and paint boxes as each processed frame becomes available.';
    const displayedModelPath = videoMeta.modelPath || activeTrainingRun?.stableBestModel || 'models/active_model.json';
    const displayedModelLabel = activeTrainingRun?.modelChoice || activeTrainingRun?.modelLabel || 'Active model';

    const stopSyncLoop = () => {
        if (animationRef.current) {
            window.cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }
    };

    const getEffectiveFps = () => {
        if (videoMetaRef.current.fps > 0) return videoMetaRef.current.fps;
        const video = videoRef.current;
        if (videoMetaRef.current.duration > 0 && videoMetaRef.current.frameCount > 0) {
            return videoMetaRef.current.frameCount / videoMetaRef.current.duration;
        }
        if (video?.duration && video.duration > 0 && videoMetaRef.current.frameCount > 0) {
            return videoMetaRef.current.frameCount / video.duration;
        }
        return 30;
    };

    const syncDetectionsForFrame = (frameIndex) => {
        const safeIndex = Math.max(0, Number.isFinite(frameIndex) ? frameIndex : 0);
        currentFrameIndexRef.current = safeIndex;
        setCurrentFrameIndex(safeIndex);
        const frameEntry = frameDetectionsRef.current.get(safeIndex);
        setCurrentDetections(frameEntry?.detections || []);
    };

    const syncFromVideo = () => {
        const video = videoRef.current;
        if (!video) return;

        const fps = getEffectiveFps();
        const totalFrames = videoMetaRef.current.frameCount
            || (video.duration > 0 && fps > 0 ? Math.max(1, Math.round(video.duration * fps)) : 1);
        const nextFrameIndex = clamp(
            Math.floor((video.currentTime || 0) * fps),
            0,
            Math.max(0, totalFrames - 1),
        );

        if (nextFrameIndex !== currentFrameIndexRef.current) {
            syncDetectionsForFrame(nextFrameIndex);
        }

        if (!video.paused && !video.ended) {
            animationRef.current = window.requestAnimationFrame(syncFromVideo);
        } else {
            animationRef.current = null;
        }
    };

    const startSyncLoop = () => {
        stopSyncLoop();
        animationRef.current = window.requestAnimationFrame(syncFromVideo);
    };

    const resetPredictionState = () => {
        frameDetectionsRef.current = new Map();
        currentFrameIndexRef.current = 0;
        setProcessingState('idle');
        setProcessingProgress(createEmptyProgress());
        setVideoMeta((prev) => ({
            ...createEmptyMeta(),
            duration: prev.duration,
            width: prev.width,
            height: prev.height,
            sourcePath: prev.sourcePath,
        }));
        setCurrentFrameIndex(0);
        setCurrentDetections([]);
        setIsPlaying(false);
        setLastError('');
        setLastUpdatedFrame(null);
    };

    const updateVideoMetaFromElement = () => {
        const video = videoRef.current;
        if (!video) return;
        setVideoMeta((prev) => ({
            ...prev,
            duration: Number.isFinite(video.duration) ? video.duration : prev.duration,
            width: video.videoWidth || prev.width,
            height: video.videoHeight || prev.height,
            sourcePath: videoPathRef.current || prev.sourcePath,
        }));
    };

    const attemptPlay = async () => {
        const video = videoRef.current;
        if (!video) return;
        try {
            await video.play();
            autoplayPendingRef.current = false;
        } catch {
            autoplayPendingRef.current = true;
        }
    };

    const handleSelectVideo = async () => {
        if (busyWithOtherProcess) return;
        if (isVideoPredictionActive && window.electronAPI?.stopVideoPrediction) {
            await window.electronAPI.stopVideoPrediction();
        }

        const selectedPath = await window.electronAPI.selectVideo();
        if (!selectedPath) return;

        stopSyncLoop();
        sessionIdRef.current = null;
        autoplayPendingRef.current = false;
        setVideoPath(selectedPath);
        resetPredictionState();
        appendLogRef.current?.(`[Video] Selected source: ${selectedPath}`);
    };

    const handleStartPrediction = async () => {
        if (!videoPath || busyWithOtherProcess || isVideoPredictionActive) return;

        const video = videoRef.current;
        stopSyncLoop();
        resetPredictionState();
        autoplayPendingRef.current = predictionMode === 'stream';

        if (video) {
            video.pause();
            setIsPlaying(false);
            video.currentTime = 0;
            updateVideoMetaFromElement();
        }

        setProcessingState('starting');
        setAppBusy(true);
        setVideoPredictionActive(true);
        appendLogRef.current?.(
            `[Video] Starting ${predictionMode === 'precompute' ? 'precompute + sync' : 'frame-by-frame live'} inference for ${videoPath}.`,
        );

        const result = await window.electronAPI.startVideoPrediction({
            videoPath,
            mode: predictionMode,
        });

        if (result?.error) {
            setProcessingState('error');
            setLastError(result.error);
            setAppBusy(false);
            setVideoPredictionActive(false);
            appendLogRef.current?.(`[Video][Error] ${result.error}`);
            return;
        }

        sessionIdRef.current = result?.session_id || null;
        if (predictionMode === 'stream') {
            await attemptPlay();
        }
    };

    const handleStopPrediction = async () => {
        if (!isVideoPredictionActive) return;
        setProcessingState('stopping');
        appendLogRef.current?.('[Video] Stopping active prediction session...');
        await window.electronAPI.stopVideoPrediction();
    };

    const handleTogglePlay = async () => {
        const video = videoRef.current;
        if (!video || playbackDisabled) return;

        if (video.paused || video.ended) {
            await attemptPlay();
        } else {
            video.pause();
        }
    };

    const handleSeek = (event) => {
        const video = videoRef.current;
        if (!video) return;
        const nextTime = Number(event.target.value);
        video.currentTime = Number.isFinite(nextTime) ? nextTime : 0;
        syncFromVideo();
    };

    useEffect(() => {
        if (!window.electronAPI?.onVideoPredictionEvent) return undefined;

        window.electronAPI.onVideoPredictionEvent((payload) => {
            if (!payload) return;
            if (!sessionIdRef.current && payload.session_id) {
                sessionIdRef.current = payload.session_id;
            }
            if (sessionIdRef.current && payload.session_id && payload.session_id !== sessionIdRef.current) {
                return;
            }

            if (payload.type === 'started') {
                const nextMeta = {
                    fps: Number(payload.fps || 0),
                    frameCount: Number(payload.frame_count || 0),
                    duration: Number(payload.duration || 0),
                    width: Number(payload.frame_width || 0),
                    height: Number(payload.frame_height || 0),
                    modelPath: payload.model_path || null,
                    device: payload.device || null,
                    deviceName: payload.device_name || null,
                    sourcePath: payload.video_path || videoPathRef.current,
                };
                setVideoMeta((prev) => ({ ...prev, ...nextMeta }));
                setProcessingProgress({
                    processed: 0,
                    total: nextMeta.frameCount,
                    percent: 0,
                });
                setProcessingState('processing');
                setLastError('');
                return;
            }

            if (payload.type === 'frame') {
                const frameIndex = Number(payload.frame_index || 0);
                const detections = normalizeDetections(payload.detections || []);
                frameDetectionsRef.current.set(frameIndex, {
                    detections,
                    classSummary: payload.class_summary || summarizeDetections(detections),
                    timeS: Number(payload.time_s || 0),
                });
                setLastUpdatedFrame(frameIndex);

                if (Math.abs(frameIndex - currentFrameIndexRef.current) <= 1) {
                    setCurrentDetections(detections);
                }
                return;
            }

            if (payload.type === 'progress') {
                const processed = Number(payload.processed_frames || 0);
                const total = Number(payload.total_frames || videoMetaRef.current.frameCount || 0);
                const percent = Number(payload.percent || (total > 0 ? (processed / total) * 100 : 0));
                setProcessingProgress({
                    processed,
                    total,
                    percent,
                });
                setProcessingState('processing');
                return;
            }

            if (payload.type === 'complete') {
                const processed = Number(payload.processed_frames || 0);
                const total = Number(payload.total_frames || processed || videoMetaRef.current.frameCount || 0);
                setProcessingProgress({
                    processed,
                    total,
                    percent: total > 0 ? 100 : 0,
                });
                setProcessingState(processingModeRef.current === 'precompute' ? 'ready' : 'complete');
                setAppBusy(false);
                setVideoPredictionActive(false);
                sessionIdRef.current = null;

                const video = videoRef.current;
                if (processingModeRef.current === 'precompute' && video) {
                    video.pause();
                    video.currentTime = 0;
                    syncDetectionsForFrame(0);
                }
                return;
            }

            if (payload.type === 'stopped') {
                stopSyncLoop();
                const video = videoRef.current;
                if (video) {
                    video.pause();
                }
                setIsPlaying(false);
                setProcessingState('stopped');
                setAppBusy(false);
                setVideoPredictionActive(false);
                sessionIdRef.current = null;
                return;
            }

            if (payload.type === 'error') {
                stopSyncLoop();
                const video = videoRef.current;
                if (video) {
                    video.pause();
                }
                setIsPlaying(false);
                setProcessingState('error');
                setLastError(payload.message || 'Video prediction failed.');
                setAppBusy(false);
                setVideoPredictionActive(false);
                sessionIdRef.current = null;
            }
        });

        return () => {
            stopSyncLoop();
            if (window.electronAPI?.removeVideoPredictionListener) {
                window.electronAPI.removeVideoPredictionListener();
            }
            if (videoPredictionActiveRef.current && window.electronAPI?.stopVideoPrediction) {
                window.electronAPI.stopVideoPrediction();
                setVideoPredictionActive(false);
                setAppBusy(false);
            }
        };
    }, [setAppBusy, setVideoPredictionActive]);

    useEffect(() => {
        if (!videoPath || isVideoPredictionActive) return;
        stopSyncLoop();
        const video = videoRef.current;
        if (video) {
            video.pause();
            video.currentTime = 0;
        }
        resetPredictionState();
    }, [predictionMode]);

    return (
        <div className="flex flex-col h-full space-y-6">
            <div className="bg-slate-800/50 p-6 rounded-xl border border-slate-700 flex flex-col gap-6">
                <div className="grid grid-cols-1 xl:grid-cols-[1.2fr_0.8fr_auto_auto] gap-4">
                    <div className="flex flex-col gap-2">
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Video Source</label>
                        <div className="px-3 py-2.5 bg-slate-900 border border-slate-700 rounded-lg text-sm text-slate-300 font-mono break-all">
                            {videoPath || 'No video selected'}
                        </div>
                    </div>
                    <div className="flex flex-col gap-2">
                        <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Inference Mode</label>
                        <select
                            value={predictionMode}
                            onChange={(event) => setPredictionMode(event.target.value)}
                            disabled={isVideoPredictionActive || busyWithOtherProcess}
                            className="bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-sm text-slate-200 focus:outline-none focus:border-rose-500 disabled:opacity-60"
                        >
                            <option value="precompute">Precompute + sync</option>
                            <option value="stream">Frame-by-frame live</option>
                        </select>
                    </div>
                    <button
                        onClick={handleSelectVideo}
                        disabled={busyWithOtherProcess}
                        className="px-6 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded-lg transition text-sm font-bold self-end"
                    >
                        Select Video
                    </button>
                    <div className="flex gap-3 self-end">
                        <button
                            onClick={handleStartPrediction}
                            disabled={!videoPath || busyWithOtherProcess || isVideoPredictionActive}
                            className="px-6 py-2 bg-rose-600 hover:bg-rose-500 disabled:opacity-50 rounded-lg transition text-white shadow-lg shadow-rose-950/30 text-sm font-bold"
                        >
                            Start Prediction
                        </button>
                        <button
                            onClick={handleStopPrediction}
                            disabled={!isVideoPredictionActive}
                            className="px-5 py-2 bg-red-600/20 hover:bg-red-600/35 disabled:opacity-50 rounded-lg transition text-red-200 border border-red-500/30 text-sm font-bold"
                        >
                            Stop
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-4">
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-4 text-sm text-slate-300 leading-relaxed">
                        <div className="text-[11px] font-black uppercase tracking-[0.24em] text-rose-400 mb-2">Mode Behavior</div>
                        <p>{modeDescription}</p>
                    </div>
                    <div className="bg-slate-900/70 border border-slate-700 rounded-xl p-4 text-sm text-slate-300 leading-relaxed">
                        <div className="text-[11px] font-black uppercase tracking-[0.24em] text-orange-300 mb-2">Session Status</div>
                        <p>
                            {statusLabels[processingState] || 'Idle'}
                            {lastError ? `: ${lastError}` : ''}
                        </p>
                    </div>
                </div>
            </div>

            {!videoPath ? (
                <div className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-slate-800 rounded-xl text-slate-500 p-12 min-h-[360px]">
                    <Film size={48} className="mb-4 opacity-50" />
                    <p>Select a local gameplay video to begin.</p>
                </div>
            ) : (
                <div className="grid flex-1 min-h-[360px] grid-cols-1 xl:grid-cols-[minmax(0,1.45fr)_390px] gap-6">
                    <div className="rounded-xl border-2 border-slate-700 bg-black min-h-[360px] flex flex-col overflow-hidden">
                        <div className="flex-1 flex items-center justify-center overflow-hidden p-6 py-8">
                            <div className="relative inline-block max-w-full max-h-full shadow-2xl">
                                <video
                                    ref={videoRef}
                                    src={toFileUrl(videoPath)}
                                    className="max-w-full max-h-[50vh] object-contain block rounded"
                                    onLoadedMetadata={() => {
                                        updateVideoMetaFromElement();
                                        syncDetectionsForFrame(0);
                                        if (autoplayPendingRef.current && processingModeRef.current === 'stream') {
                                            attemptPlay();
                                        }
                                    }}
                                    onPlay={() => {
                                        setIsPlaying(true);
                                        startSyncLoop();
                                    }}
                                    onPause={() => {
                                        setIsPlaying(false);
                                        stopSyncLoop();
                                        syncFromVideo();
                                    }}
                                    onSeeking={() => syncFromVideo()}
                                    onEnded={() => {
                                        setIsPlaying(false);
                                        stopSyncLoop();
                                        syncFromVideo();
                                    }}
                                />

                                {currentDetections.map((detection, index) => {
                                    const chrome = getDetectionChrome(detection.class_key || detection.class_name);
                                    return (
                                        <div
                                            key={`${index}-${detection.class_id}-${detection.confidence}`}
                                            className={`absolute border-2 ${chrome.border} ${chrome.fill} ${chrome.glow} z-10 transition-all duration-150 pointer-events-none`}
                                            style={{
                                                left: `${(detection.x_center - detection.width / 2) * 100}%`,
                                                top: `${(detection.y_center - detection.height / 2) * 100}%`,
                                                width: `${detection.width * 100}%`,
                                                height: `${detection.height * 100}%`,
                                            }}
                                        >
                                            <div className={`absolute -top-6 left-0 px-2 py-0.5 ${chrome.badge} text-[10px] font-bold uppercase tracking-wider text-white rounded`}>
                                                {detection.class_name} {(Number(detection.confidence || 0) * 100).toFixed(0)}%
                                            </div>
                                        </div>
                                    );
                                })}

                                {predictionMode === 'precompute' && processingState === 'processing' && (
                                    <div className="absolute inset-0 bg-slate-950/50 flex items-center justify-center rounded">
                                        <div className="px-4 py-3 bg-slate-900/90 border border-slate-700 rounded-xl text-sm text-slate-100 font-semibold">
                                            Processing video before playback...
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="border-t border-slate-800 bg-slate-950/70 p-4 flex flex-col gap-4">
                            <div className="flex items-center gap-3">
                                <button
                                    onClick={handleTogglePlay}
                                    disabled={playbackDisabled}
                                    className="w-11 h-11 rounded-full bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center text-slate-100 transition"
                                >
                                    {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
                                </button>
                                <button
                                    onClick={() => {
                                        const video = videoRef.current;
                                        if (!video) return;
                                        video.pause();
                                        video.currentTime = 0;
                                        syncDetectionsForFrame(0);
                                    }}
                                    className="w-11 h-11 rounded-full bg-slate-800 hover:bg-slate-700 flex items-center justify-center text-slate-100 transition"
                                >
                                    <Square size={16} />
                                </button>
                                <div className="text-xs font-mono text-slate-400 whitespace-nowrap">
                                    {formatTime(videoRef.current?.currentTime || 0)} / {formatTime(videoMeta.duration || 0)}
                                </div>
                            </div>

                            <input
                                type="range"
                                min="0"
                                max={Math.max(videoMeta.duration, 0)}
                                step="0.01"
                                value={Math.min(videoRef.current?.currentTime || 0, Math.max(videoMeta.duration, 0))}
                                onChange={handleSeek}
                                className="w-full accent-rose-500"
                            />
                        </div>
                    </div>

                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-5 flex flex-col gap-5 min-h-[360px]">
                        <div className="grid grid-cols-2 gap-3">
                            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4">
                                <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-rose-300">Enemies</div>
                                <div className="mt-2 text-3xl font-black text-white">{currentClassSummary.enemy}</div>
                            </div>
                            <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4">
                                <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-amber-200">Players</div>
                                <div className="mt-2 text-3xl font-black text-white">{currentClassSummary.player}</div>
                            </div>
                        </div>

                        <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4">
                            <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400 mb-3">
                                <Activity size={14} />
                                Processing Progress
                            </div>
                            <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-700">
                                <div
                                    className="bg-gradient-to-r from-rose-500 via-red-500 to-orange-400 h-full rounded-full transition-all duration-200"
                                    style={{ width: `${clamp(processingProgress.percent || 0, 0, 100)}%` }}
                                />
                            </div>
                            <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                                <div>
                                    <div className="text-slate-500">Frames</div>
                                    <div className="font-bold text-white">{processingProgress.processed} / {processingProgress.total || derivedFrameTotal || 0}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">Last Updated</div>
                                    <div className="font-bold text-white">{lastUpdatedFrame ?? 'n/a'}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">Current Frame</div>
                                    <div className="font-bold text-white">{currentFrameIndex}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">FPS</div>
                                    <div className="font-bold text-white">{videoMeta.fps ? videoMeta.fps.toFixed(2) : 'n/a'}</div>
                                </div>
                            </div>
                        </div>

                        <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 text-sm text-slate-200">
                            <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400 mb-3">Session</div>
                            <div className="grid grid-cols-2 gap-3">
                                <div>
                                    <div className="text-slate-500">Mode</div>
                                    <div className="font-bold text-white">{predictionMode === 'precompute' ? 'Precompute + sync' : 'Frame-by-frame live'}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">Status</div>
                                    <div className="font-bold text-white">{statusLabels[processingState] || 'Idle'}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">Video</div>
                                    <div className="font-bold text-white break-all">{selectedVideoName || 'n/a'}</div>
                                </div>
                                <div>
                                    <div className="text-slate-500">Device</div>
                                    <div className="font-bold text-white break-all">{videoMeta.deviceName || videoMeta.device || 'n/a'}</div>
                                </div>
                            </div>
                        </div>

                        <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-4 flex flex-col gap-3 min-h-0 flex-1">
                            <div className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.2em] text-slate-400">
                                <Film size={14} />
                                Current Frame Detections
                            </div>
                            {currentDetections.length ? (
                                <div className="flex flex-col gap-2 overflow-y-auto pr-1">
                                    {currentDetections.map((detection, index) => {
                                        const chrome = getDetectionChrome(detection.class_key || detection.class_name);
                                        return (
                                            <div
                                                key={`video-detection-${index}-${detection.class_id}-${detection.confidence}`}
                                                className="rounded-lg border border-slate-700 bg-slate-950/80 p-3"
                                            >
                                                <div className="flex items-start justify-between gap-3">
                                                    <div>
                                                        <div className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-white ${chrome.badge}`}>
                                                            {detection.class_name}
                                                        </div>
                                                        <div className="mt-2 text-xs text-slate-400">
                                                            Center: [{Number(detection.x_center || 0).toFixed(3)}, {Number(detection.y_center || 0).toFixed(3)}]
                                                        </div>
                                                        <div className="text-xs text-slate-500">
                                                            Size: {Number(detection.width || 0).toFixed(3)} x {Number(detection.height || 0).toFixed(3)}
                                                        </div>
                                                    </div>
                                                    <div className="text-right">
                                                        <div className="text-lg font-black text-white">
                                                            {(Number(detection.confidence || 0) * 100).toFixed(0)}%
                                                        </div>
                                                        <div className="text-[11px] uppercase tracking-widest text-slate-500">confidence</div>
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : (
                                <div className="flex-1 flex flex-col items-center justify-center rounded-lg border border-dashed border-slate-700 text-center text-slate-500 p-6">
                                    <Film size={24} className="mb-3 opacity-60" />
                                    <p className="text-sm">
                                        {processingState === 'processing' && predictionMode === 'stream'
                                            ? 'Waiting for detections on the current playback frame.'
                                            : 'No detections are currently available for this frame.'}
                                    </p>
                                </div>
                            )}
                        </div>

                        <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-3 text-xs text-slate-400 leading-relaxed">
                            <div className="font-bold uppercase tracking-[0.2em] text-slate-500 mb-2">Model</div>
                            <div className="text-sm font-semibold text-white mb-2">{displayedModelLabel}</div>
                            <div className="break-all">{displayedModelPath}</div>
                        </div>
                    </div>
                </div>
            )}

            {renderTerminal()}
        </div>
    );
}
