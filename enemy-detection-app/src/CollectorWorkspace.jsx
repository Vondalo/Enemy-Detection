import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, CheckCircle2, CircleHelp, Copy, Eraser, Flag, Keyboard, Pause, Play, Save, SkipForward } from 'lucide-react';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const MIN_BOX = 8;
const CLASS_OPTIONS = [
    { id: 0, name: 'enemy', color: '#ef4444', accent: 'border-rose-500/40 bg-rose-500/10 text-rose-100' },
    { id: 1, name: 'player', color: '#38bdf8', accent: 'border-sky-500/40 bg-sky-500/10 text-sky-100' },
];
const DEFAULT_CLASS = CLASS_OPTIONS[0];

const toFileUrl = (filePath) => {
    const normalized = filePath.replace(/\\/g, '/');
    return encodeURI(/^[A-Za-z]:\//.test(normalized) ? `file:///${normalized}` : `file://${normalized}`);
};

const formatTime = (seconds) => {
    const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(Math.floor(safe % 60)).padStart(2, '0')}`;
};

const clampBox = (box, width, height) => {
    const safeWidth = clamp(box.width, MIN_BOX, width);
    const safeHeight = clamp(box.height, MIN_BOX, height);
    return {
        ...box,
        x: clamp(box.x, 0, Math.max(0, width - safeWidth)),
        y: clamp(box.y, 0, Math.max(0, height - safeHeight)),
        width: safeWidth,
        height: safeHeight,
    };
};

const pixelsToYolo = (box, width, height) => ({
    x_center: (box.x + box.width / 2) / width,
    y_center: (box.y + box.height / 2) / height,
    width: box.width / width,
    height: box.height / height,
});

const yoloToPixels = (bbox, width, height) => clampBox({
    x: (bbox.x_center - bbox.width / 2) * width,
    y: (bbox.y_center - bbox.height / 2) * height,
    width: bbox.width * width,
    height: bbox.height * height,
}, width, height);

const getClassOption = (classId, className) =>
    CLASS_OPTIONS.find((option) => option.id === Number(classId))
    || CLASS_OPTIONS.find((option) => option.name === String(className || '').toLowerCase())
    || DEFAULT_CLASS;

const drawBox = (ctx, box, color, label) => {
    if (!box) return;
    ctx.save();
    ctx.lineWidth = 3;
    ctx.strokeStyle = color;
    ctx.fillStyle = `${color}22`;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = color;
    ctx.fillRect(box.x, Math.max(0, box.y - 24), Math.max(92, label.length * 8 + 18), 24);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(label, box.x + 8, Math.max(16, box.y - 8));
    ctx.restore();
};

export default function CollectorWorkspace({ session, appendLog, onClose, onSessionUpdate }) {
    const canvasRef = useRef(null);
    const videoRef = useRef(null);
    const exportCanvasRef = useRef(null);
    const dragRef = useRef(null);
    const frameRef = useRef(0);
    const savedFramesRef = useRef({});
    const playTimerRef = useRef(null);

    const [videoMeta, setVideoMeta] = useState({ ready: false, width: 0, height: 0, duration: 0 });
    const [fps, setFps] = useState(30);
    const [frameStep, setFrameStep] = useState(1);
    const [jumpStep, setJumpStep] = useState(15);
    const [clickBox, setClickBox] = useState(96);
    const [selectedClass, setSelectedClass] = useState(DEFAULT_CLASS);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [currentBox, setCurrentBox] = useState(null);
    const [savedFrames, setSavedFrames] = useState({});
    const [savedCount, setSavedCount] = useState(session.savedCount || 0);
    const [lastSavedBox, setLastSavedBox] = useState(null);
    const [status, setStatus] = useState('Load the video, choose player or enemy, draw a box, then save the frame.');
    const [isPlaying, setIsPlaying] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isFinishing, setIsFinishing] = useState(false);
    const [toast, setToast] = useState(null);

    const totalFrames = videoMeta.ready ? Math.max(1, Math.round(videoMeta.duration * fps)) : 1;

    useEffect(() => { exportCanvasRef.current = document.createElement('canvas'); }, []);
    useEffect(() => { frameRef.current = currentFrame; }, [currentFrame]);
    useEffect(() => { savedFramesRef.current = savedFrames; }, [savedFrames]);
    useEffect(() => {
        if (!toast) return undefined;
        const timeout = window.setTimeout(() => setToast(null), 1800);
        return () => window.clearTimeout(timeout);
    }, [toast]);

    useEffect(() => {
        if (!currentBox) return;
        setCurrentBox((prev) => prev ? { ...prev, class_id: selectedClass.id, class_name: selectedClass.name } : prev);
    }, [selectedClass]);

    const getFrameBoxes = (frame = frameRef.current) => savedFramesRef.current[frame] || [];

    const renderCanvas = (box = currentBox, frame = frameRef.current) => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video || !videoMeta.ready || video.readyState < 2) return;

        if (canvas.width !== videoMeta.width || canvas.height !== videoMeta.height) {
            canvas.width = videoMeta.width;
            canvas.height = videoMeta.height;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const savedBoxes = getFrameBoxes(frame).map((savedBox, index) => ({
            pixelBox: yoloToPixels(savedBox, canvas.width, canvas.height),
            classOption: getClassOption(savedBox.class_id, savedBox.class_name),
            index,
        }));

        savedBoxes.forEach(({ pixelBox, classOption, index }) => {
            drawBox(ctx, pixelBox, classOption.color, `${classOption.name} ${index + 1}`);
        });
        if (box) {
            const activeClass = getClassOption(box.class_id, box.class_name);
            drawBox(ctx, box, activeClass.color, `${activeClass.name} draft`);
        }
    };

    useEffect(() => {
        if (!videoMeta.ready) return;
        renderCanvas(currentBox, currentFrame);
    }, [currentBox, currentFrame, savedFrames, videoMeta.ready]);

    const seekToFrame = (frame) => {
        const video = videoRef.current;
        if (!video || !videoMeta.ready) return;

        const safeFrame = clamp(Math.round(frame), 0, Math.max(0, totalFrames - 1));
        const targetTime = clamp(safeFrame / fps, 0, Math.max(0, video.duration));

        if (Math.abs(video.currentTime - targetTime) < 0.0005) {
            setCurrentFrame(safeFrame);
            setCurrentBox(null);
            renderCanvas(null, safeFrame);
            return;
        }

        video.currentTime = targetTime;
    };

    const moveFrames = (delta) => {
        setIsPlaying(false);
        seekToFrame(frameRef.current + delta);
    };

    const clearBox = () => {
        setCurrentBox(null);
        setStatus('Current box cleared.');
        renderCanvas(null);
    };

    const skipFrame = () => {
        clearBox();
        moveFrames(frameStep);
    };

    const duplicateLastBox = () => {
        if (!lastSavedBox || !videoMeta.ready) {
            setStatus('No previous box is available yet.');
            return;
        }

        const nextBox = clampBox(lastSavedBox, videoMeta.width, videoMeta.height);
        setCurrentBox(nextBox);
        setSelectedClass(getClassOption(nextBox.class_id, nextBox.class_name));
        setStatus(`Copied the previous saved ${getClassOption(nextBox.class_id, nextBox.class_name).name} box onto this frame.`);
        renderCanvas(nextBox);
    };

    const persistFrameBoxes = async (nextBoxes, nextStatus, nextToastMessage) => {
        const video = videoRef.current;
        const exportCanvas = exportCanvasRef.current;
        if (!video || !exportCanvas) return false;

        exportCanvas.width = videoMeta.width;
        exportCanvas.height = videoMeta.height;
        const exportCtx = exportCanvas.getContext('2d');
        exportCtx.clearRect(0, 0, exportCanvas.width, exportCanvas.height);
        exportCtx.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height);

        const result = await window.electronAPI.saveManualAnnotation({
            datasetPath: session.datasetPath,
            datasetName: session.datasetName,
            videoName: session.videoName,
            frameIndex: frameRef.current,
            timestamp: video.currentTime || frameRef.current / fps,
            boxes: nextBoxes,
            imageDataUrl: exportCanvas.toDataURL('image/png'),
            bboxSource: 'manual_box',
        });

        if (result.success) {
            setSavedFrames((prev) => ({ ...prev, [frameRef.current]: nextBoxes }));
            setSavedCount(result.savedCount);
            onSessionUpdate?.({ savedCount: result.savedCount });
            setStatus(nextStatus);
            setToast({ tone: 'success', message: nextToastMessage });
            return true;
        }

        appendLog(`[Collector Error] ${result.error}`);
        setStatus(result.error || 'Save failed.');
        setToast({ tone: 'error', message: result.error || 'Save failed.' });
        return false;
    };

    const saveFrame = async (advanceToNext = true) => {
        if (!currentBox || !videoMeta.ready || isSaving) return;

        setIsSaving(true);
        setIsPlaying(false);

        const bbox = {
            ...pixelsToYolo(currentBox, videoMeta.width, videoMeta.height),
            class_id: currentBox.class_id ?? selectedClass.id,
            class_name: currentBox.class_name ?? selectedClass.name,
        };
        const frameBoxes = [...getFrameBoxes(frameRef.current), bbox];
        const saved = await persistFrameBoxes(
            frameBoxes,
            `Saved ${frameBoxes.length} box(es) on frame ${frameRef.current}.`,
            advanceToNext && frameRef.current < totalFrames - 1
                ? `Saved ${bbox.class_name} ${frameBoxes.length} on frame ${frameRef.current}. Moving to the next frame.`
                : `Saved ${bbox.class_name} ${frameBoxes.length} on frame ${frameRef.current}.`
        );

        if (saved) {
            setLastSavedBox(currentBox);
            appendLog(`[Collector] Saved ${frameBoxes.length} box(es) on frame ${frameRef.current}.`);
            if (advanceToNext && frameRef.current < totalFrames - 1) {
                setCurrentBox(null);
                seekToFrame(frameRef.current + frameStep);
            } else {
                setCurrentBox(null);
                renderCanvas(null, frameRef.current);
            }
        }

        setIsSaving(false);
    };

    const undoLastSavedBox = async () => {
        if (!videoMeta.ready || isSaving) return;

        const existingBoxes = getFrameBoxes(frameRef.current);
        if (existingBoxes.length === 0) {
            setStatus('There are no saved boxes on this frame to remove.');
            return;
        }

        setIsSaving(true);
        const nextBoxes = existingBoxes.slice(0, -1);
        if (nextBoxes.length === 0) {
            setSavedFrames((prev) => ({ ...prev, [frameRef.current]: [] }));
            const video = videoRef.current;
            const exportCanvas = exportCanvasRef.current;
            if (video && exportCanvas) {
                exportCanvas.width = videoMeta.width;
                exportCanvas.height = videoMeta.height;
                const exportCtx = exportCanvas.getContext('2d');
                exportCtx.clearRect(0, 0, exportCanvas.width, exportCanvas.height);
                exportCtx.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height);
                const result = await window.electronAPI.saveManualAnnotation({
                    datasetPath: session.datasetPath,
                    datasetName: session.datasetName,
                    videoName: session.videoName,
                    frameIndex: frameRef.current,
                    timestamp: video.currentTime || frameRef.current / fps,
                    boxes: [],
                    imageDataUrl: exportCanvas.toDataURL('image/png'),
                    bboxSource: 'manual_box',
                });
                if (!result.success) {
                    appendLog(`[Collector Error] ${result.error}`);
                    setStatus(result.error || 'Could not update this frame.');
                    setToast({ tone: 'error', message: result.error || 'Could not update this frame.' });
                    setIsSaving(false);
                    return;
                }
                setSavedCount(result.savedCount);
                onSessionUpdate?.({ savedCount: result.savedCount });
            }
            setStatus(`Removed the last saved box from frame ${frameRef.current}.`);
            setToast({ tone: 'success', message: `Removed the last saved box from frame ${frameRef.current}.` });
            renderCanvas(null, frameRef.current);
            setIsSaving(false);
            return;
        }

        const saved = await persistFrameBoxes(
            nextBoxes,
            `Removed the last saved box. ${nextBoxes.length} box(es) remain on frame ${frameRef.current}.`,
            `Removed the last saved box from frame ${frameRef.current}.`
        );
        if (saved) {
            renderCanvas(null, frameRef.current);
        }
        setIsSaving(false);
    };

    const finishSession = async () => {
        if (isFinishing) return;

        setIsPlaying(false);
        setIsFinishing(true);
        const result = await window.electronAPI.finishManualCollection(session.datasetPath);

        if (result.success) {
            appendLog(`[Collector] Finished ${session.datasetName} with ${result.savedCount} saved annotation(s).`);
            onClose?.(result);
        } else {
            appendLog(`[Collector Error] ${result.error}`);
            setStatus(result.error || 'Could not finish the collection session.');
        }

        setIsFinishing(false);
    };

    const pointFromEvent = (event) => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        return {
            x: clamp((event.clientX - rect.left) * (canvas.width / rect.width), 0, canvas.width),
            y: clamp((event.clientY - rect.top) * (canvas.height / rect.height), 0, canvas.height),
        };
    };

    const onPointerDown = (event) => {
        if (!videoMeta.ready || event.button !== 0) return;
        setIsPlaying(false);

        const point = pointFromEvent(event);
        if (!point) return;

        dragRef.current = point;
        const seed = { x: point.x, y: point.y, width: 0, height: 0, class_id: selectedClass.id, class_name: selectedClass.name };
        setCurrentBox(seed);
        setStatus(`Drawing a ${selectedClass.name} box...`);
        renderCanvas(seed);
    };

    const onPointerMove = (event) => {
        if (!dragRef.current || !videoMeta.ready) return;

        const point = pointFromEvent(event);
        if (!point) return;

        const box = {
            x: Math.min(dragRef.current.x, point.x),
            y: Math.min(dragRef.current.y, point.y),
            width: Math.abs(point.x - dragRef.current.x),
            height: Math.abs(point.y - dragRef.current.y),
            class_id: selectedClass.id,
            class_name: selectedClass.name,
        };
        setCurrentBox(box);
        renderCanvas(box);
    };

    const onPointerUp = (event) => {
        if (!dragRef.current || !videoMeta.ready) return;

        const point = pointFromEvent(event) || dragRef.current;
        const rawBox = {
            x: Math.min(dragRef.current.x, point.x),
            y: Math.min(dragRef.current.y, point.y),
            width: Math.abs(point.x - dragRef.current.x),
            height: Math.abs(point.y - dragRef.current.y),
        };
        dragRef.current = null;

        const nextBox = rawBox.width < MIN_BOX || rawBox.height < MIN_BOX
            ? clampBox({ x: point.x - clickBox / 2, y: point.y - clickBox / 2, width: clickBox, height: clickBox, class_id: selectedClass.id, class_name: selectedClass.name }, videoMeta.width, videoMeta.height)
            : clampBox({ ...rawBox, class_id: selectedClass.id, class_name: selectedClass.name }, videoMeta.width, videoMeta.height);

        setCurrentBox(nextBox);
        setStatus(`${selectedClass.name} box ready. Press Enter, Add Box, or Save And Next.`);
        renderCanvas(nextBox);
    };

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return undefined;

        const onLoadedMetadata = () => {
            setVideoMeta({
                ready: true,
                width: video.videoWidth || 1280,
                height: video.videoHeight || 720,
                duration: video.duration || 0,
            });
            setCurrentFrame(0);
            setCurrentBox(null);
            setStatus('Video loaded. Choose player or enemy, then drag for a custom box or click once for a quick square box.');
            appendLog(`[Collector] Loaded ${session.videoName} inside the Electron annotator.`);
        };

        const onLoadedData = () => window.requestAnimationFrame(() => renderCanvas(null, 0));
        const onSeeked = () => {
            const frame = clamp(Math.round(video.currentTime * fps), 0, Math.max(0, totalFrames - 1));
            const savedCountForFrame = getFrameBoxes(frame).length;
            setCurrentFrame(frame);
            setCurrentBox(null);
            setStatus(savedCountForFrame > 0 ? `Frame ${frame} has ${savedCountForFrame} saved box(es). Draw another one or move on.` : `Frame ${frame} ready.`);
            renderCanvas(null, frame);
        };

        video.addEventListener('loadedmetadata', onLoadedMetadata);
        video.addEventListener('loadeddata', onLoadedData);
        video.addEventListener('seeked', onSeeked);

        return () => {
            video.removeEventListener('loadedmetadata', onLoadedMetadata);
            video.removeEventListener('loadeddata', onLoadedData);
            video.removeEventListener('seeked', onSeeked);
        };
    }, [appendLog, fps, session.videoName, totalFrames, videoMeta.height, videoMeta.width]);

    useEffect(() => {
        if (!isPlaying || !videoMeta.ready) {
            if (playTimerRef.current) window.clearInterval(playTimerRef.current);
            playTimerRef.current = null;
            return undefined;
        }

        const interval = window.setInterval(() => {
            if (frameRef.current >= totalFrames - 1) {
                setIsPlaying(false);
                return;
            }
            seekToFrame(frameRef.current + frameStep);
        }, Math.max(50, Math.round((1000 / fps) * frameStep)));

        playTimerRef.current = interval;
        return () => window.clearInterval(interval);
    }, [fps, frameStep, isPlaying, totalFrames, videoMeta.ready]);

    useEffect(() => {
        const onKeyDown = (event) => {
            const tag = event.target?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || !videoMeta.ready) return;

            if (event.key === ' ') {
                event.preventDefault();
                setIsPlaying((prev) => !prev);
                return;
            }

            if (event.key === 'ArrowRight') {
                event.preventDefault();
                moveFrames(event.shiftKey ? jumpStep : frameStep);
                return;
            }

            if (event.key === 'ArrowLeft') {
                event.preventDefault();
                moveFrames(event.shiftKey ? -jumpStep : -frameStep);
                return;
            }

            if (event.key === 'Enter') {
                event.preventDefault();
                saveFrame(!event.shiftKey);
                return;
            }

            if (event.key.toLowerCase() === 'd') {
                event.preventDefault();
                duplicateLastBox();
                return;
            }

            if (event.key.toLowerCase() === 'c' || event.key === 'Backspace') {
                event.preventDefault();
                clearBox();
                return;
            }

            if (event.key.toLowerCase() === 's') {
                event.preventDefault();
                skipFrame();
                return;
            }

            if (event.key.toLowerCase() === 'u') {
                event.preventDefault();
                undoLastSavedBox();
                return;
            }

            if (event.key === '1') {
                event.preventDefault();
                setSelectedClass(CLASS_OPTIONS[0]);
                return;
            }

            if (event.key === '2') {
                event.preventDefault();
                setSelectedClass(CLASS_OPTIONS[1]);
            }
        };

        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [frameStep, jumpStep, videoMeta.ready, currentBox, lastSavedBox, isSaving, isFinishing, currentFrame, fps]);

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.75fr)_320px] 2xl:grid-cols-[minmax(0,1.9fr)_360px] gap-4 lg:gap-5">
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5 shadow-2xl">
                <div className="flex flex-col gap-4">
                    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
                        <div>
                            <div className="text-xs font-black uppercase tracking-[0.25em] text-sky-400">In-App Annotator</div>
                            <h3 className="text-xl font-bold text-white mt-1">{session.datasetName}</h3>
                            <p className="text-sm text-slate-400 mt-1">{session.videoName} - frame {currentFrame} / {Math.max(0, totalFrames - 1)} - {formatTime(videoRef.current?.currentTime || 0)}</p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            <button onClick={() => moveFrames(-frameStep)} className="px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-semibold flex items-center gap-2 transition"><ArrowLeft size={16} />Prev</button>
                            <button onClick={() => setIsPlaying((prev) => !prev)} className="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-sm font-semibold text-white flex items-center gap-2 transition">{isPlaying ? <Pause size={16} /> : <Play size={16} />}{isPlaying ? 'Pause' : 'Play'}</button>
                            <button onClick={() => moveFrames(frameStep)} className="px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-semibold flex items-center gap-2 transition">Next<ArrowRight size={16} /></button>
                            <button onClick={finishSession} disabled={isFinishing} className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 rounded-lg text-sm font-semibold text-white flex items-center gap-2 transition"><CheckCircle2 size={16} />{isFinishing ? 'Saving...' : 'Finish Session'}</button>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-black/80 overflow-hidden p-2 sm:p-3">
                        <div className="w-full aspect-video max-h-[65vh] min-h-[280px] flex items-center justify-center overflow-hidden rounded-xl bg-black">
                            <canvas
                                ref={canvasRef}
                                onMouseDown={onPointerDown}
                                onMouseMove={onPointerMove}
                                onMouseUp={onPointerUp}
                                onMouseLeave={onPointerUp}
                                onContextMenu={(event) => {
                                    event.preventDefault();
                                    clearBox();
                                }}
                                className="block w-full h-full cursor-crosshair"
                            />
                        </div>
                        <video ref={videoRef} src={toFileUrl(session.videoPath)} preload="auto" className="hidden" />
                    </div>

                    <div className="flex flex-col gap-3">
                        <input type="range" min={0} max={Math.max(0, totalFrames - 1)} value={currentFrame} onChange={(event) => seekToFrame(Number(event.target.value))} className="w-full accent-sky-500" />
                        <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                            <span>00:00</span>
                            <span>{currentFrame} / {Math.max(0, totalFrames - 1)}</span>
                            <span>{formatTime(videoMeta.duration)}</span>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
                        <div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold mb-3">Label Class</div>
                        <div className="flex flex-wrap gap-3">
                            {CLASS_OPTIONS.map((option, index) => (
                                <button
                                    key={option.id}
                                    onClick={() => setSelectedClass(option)}
                                    className={`px-4 py-2.5 rounded-xl border text-sm font-semibold transition ${selectedClass.id === option.id ? option.accent : 'border-slate-800 bg-slate-900 text-slate-300 hover:bg-slate-800'}`}
                                >
                                    {option.name} ({index + 1})
                                </button>
                            ))}
                        </div>
                        <div className="text-sm text-slate-300 mt-3">
                            Current draft class: <span className="font-semibold text-white">{selectedClass.name}</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
                        <label className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1"><span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">FPS Guess</span><input type="number" min="1" max="240" value={fps} onChange={(event) => setFps(clamp(Number(event.target.value) || 30, 1, 240))} className="bg-transparent text-sm text-slate-100 focus:outline-none" /></label>
                        <label className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1"><span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Step Frames</span><input type="number" min="1" max="120" value={frameStep} onChange={(event) => setFrameStep(clamp(Number(event.target.value) || 1, 1, 120))} className="bg-transparent text-sm text-slate-100 focus:outline-none" /></label>
                        <label className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1"><span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Jump Frames</span><input type="number" min="1" max="1000" value={jumpStep} onChange={(event) => setJumpStep(clamp(Number(event.target.value) || 15, 1, 1000))} className="bg-transparent text-sm text-slate-100 focus:outline-none" /></label>
                        <label className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1"><span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Click Box (px)</span><input type="number" min="24" max="400" value={clickBox} onChange={(event) => setClickBox(clamp(Number(event.target.value) || 96, 24, 400))} className="bg-transparent text-sm text-slate-100 focus:outline-none" /></label>
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1 justify-center">
                            <span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Save Flow</span>
                            <span className="text-sm text-slate-100">Add Box stays on this frame. Save And Next finishes the frame and advances.</span>
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-3">
                        <button onClick={() => saveFrame(false)} disabled={!currentBox || isSaving} className="px-4 py-2.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-60 rounded-xl text-sm font-bold text-white flex items-center gap-2 transition"><Save size={16} />{isSaving ? 'Saving...' : 'Add Box'}</button>
                        <button onClick={() => saveFrame(true)} disabled={!currentBox || isSaving} className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 rounded-xl text-sm font-bold text-white flex items-center gap-2 transition"><CheckCircle2 size={16} />Save And Next</button>
                        <button onClick={undoLastSavedBox} disabled={isSaving || getFrameBoxes(currentFrame).length === 0} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-60 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><Eraser size={16} />Undo Last Saved</button>
                        <button onClick={duplicateLastBox} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><Copy size={16} />Duplicate Last Box</button>
                        <button onClick={clearBox} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><Eraser size={16} />Clear Draft Box</button>
                        <button onClick={skipFrame} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><SkipForward size={16} />Skip Frame</button>
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
                        <div className="flex flex-wrap gap-6 text-sm">
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Status</div><div className="text-slate-200 mt-1">{status}</div></div>
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Saved Annotations</div><div className="text-slate-200 mt-1">{savedCount}</div></div>
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Frame Enemies</div><div className="text-slate-200 mt-1">{getFrameBoxes(currentFrame).length}</div></div>
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Current Box</div><div className="text-slate-200 mt-1">{currentBox ? `${Math.round(currentBox.width)} x ${Math.round(currentBox.height)} px` : 'No active box'}</div></div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex flex-col gap-4 xl:max-w-[360px]">
                <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                    <div className="flex items-center gap-2 text-sky-400 mb-3"><CircleHelp size={16} /><span className="text-xs font-black uppercase tracking-[0.25em]">How To Label</span></div>
                    <div className="space-y-2 text-sm text-slate-300">
                        <div>Pick <span className="text-white font-semibold">enemy</span> or <span className="text-white font-semibold">player</span> before drawing each box.</div>
                        <div>Drag over the selected target for a precise box.</div>
                        <div>Single-click to place a quick square box centered on the target.</div>
                        <div>Use Add Box to keep stacking more saved boxes on the same frame, including mixed player and enemy labels.</div>
                        <div>Use Duplicate Last Box when the target barely moved.</div>
                        <div>Use Skip Frame when the shot is blocked or not useful.</div>
                    </div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                    <div className="flex items-center gap-2 text-indigo-400 mb-3"><Keyboard size={16} /><span className="text-xs font-black uppercase tracking-[0.25em]">Keyboard Controls</span></div>
                    <div className="space-y-2 text-sm text-slate-300">
                        <div><span className="text-white font-semibold">1</span> switches the draft label to enemy.</div>
                        <div><span className="text-white font-semibold">2</span> switches the draft label to player.</div>
                        <div><span className="text-white font-semibold">Enter</span> saves the active box and moves to the next frame.</div>
                        <div><span className="text-white font-semibold">Shift + Enter</span> saves the active box and stays on the same frame so you can add another box.</div>
                        <div><span className="text-white font-semibold">Space</span> plays or pauses.</div>
                        <div><span className="text-white font-semibold">Arrow Left / Right</span> moves by Step Frames.</div>
                        <div><span className="text-white font-semibold">Shift + Arrows</span> moves by Jump Frames.</div>
                        <div><span className="text-white font-semibold">D</span> duplicates the last box.</div>
                        <div><span className="text-white font-semibold">C</span> or <span className="text-white font-semibold">Backspace</span> clears the box.</div>
                        <div><span className="text-white font-semibold">S</span> skips to the next frame.</div>
                        <div><span className="text-white font-semibold">U</span> removes the last saved box from the current frame.</div>
                        <div><span className="text-white font-semibold">Right-click</span> clears the current box.</div>
                    </div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                    <div className="flex items-center gap-2 text-emerald-400 mb-3"><Flag size={16} /><span className="text-xs font-black uppercase tracking-[0.25em]">Control Tips</span></div>
                    <div className="space-y-2 text-sm text-slate-300">
                        <div>Keep Step Frames at 1 when movement is fast.</div>
                        <div>Switch to player before boxing your own character, then switch back to enemy for opponents.</div>
                        <div>Use Shift + Enter or the Add Box button until every visible character on the frame is saved.</div>
                        <div>Use Save And Next only after the current frame has all of its player and enemy boxes labeled.</div>
                        <div>The timeline slider is the fastest way to jump across long clips.</div>
                    </div>
                </div>
            </div>

            {toast && (
                <div className={`pointer-events-none fixed right-5 top-16 z-50 rounded-xl border px-4 py-3 text-sm font-semibold shadow-2xl backdrop-blur ${toast.tone === 'error' ? 'border-rose-500/40 bg-rose-500/15 text-rose-100' : 'border-emerald-500/40 bg-emerald-500/15 text-emerald-50'}`}>
                    {toast.message}
                </div>
            )}
        </div>
    );
}
