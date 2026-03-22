import { useEffect, useRef, useState } from 'react';
import { ArrowLeft, ArrowRight, CircleHelp, Image as ImageIcon, Keyboard, Plus, RotateCcw, Save, Trash2 } from 'lucide-react';

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const MIN_BOX = 8;
const RESIZE_HANDLE = 14;
const CLASS_OPTIONS = [
    { id: 0, name: 'enemy', color: '#ef4444', accent: 'border-rose-500/40 bg-rose-500/10 text-rose-100' },
    { id: 1, name: 'player', color: '#38bdf8', accent: 'border-sky-500/40 bg-sky-500/10 text-sky-100' },
];
const DEFAULT_CLASS = CLASS_OPTIONS[0];

const toFileUrl = (filePath) => {
    const normalized = filePath.replace(/\\/g, '/');
    return encodeURI(/^[A-Za-z]:\//.test(normalized) ? `file:///${normalized}` : `file://${normalized}`);
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

const pointInRect = (point, rect) =>
    point.x >= rect.x
    && point.x <= rect.x + rect.width
    && point.y >= rect.y
    && point.y <= rect.y + rect.height;

const getResizeHandleRect = (box) => ({
    x: box.x + box.width - RESIZE_HANDLE,
    y: box.y + box.height - RESIZE_HANDLE,
    width: RESIZE_HANDLE,
    height: RESIZE_HANDLE,
});

const cloneBoxes = (boxes) => boxes.map((box) => ({ ...box }));

const drawBox = (ctx, box, color, label, { selected = false } = {}) => {
    if (!box) return;

    ctx.save();
    ctx.lineWidth = selected ? 4 : 3;
    ctx.strokeStyle = color;
    ctx.fillStyle = `${color}22`;
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillRect(box.x, box.y, box.width, box.height);

    if (selected) {
        ctx.setLineDash([8, 6]);
        ctx.strokeRect(box.x - 2, box.y - 2, box.width + 4, box.height + 4);
        ctx.setLineDash([]);
        const handle = getResizeHandleRect(box);
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(handle.x, handle.y, handle.width, handle.height);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(handle.x, handle.y, handle.width, handle.height);
    }

    ctx.fillStyle = color;
    ctx.fillRect(box.x, Math.max(0, box.y - 24), Math.max(104, label.length * 8 + 18), 24);
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText(label, box.x + 8, Math.max(16, box.y - 8));
    ctx.restore();
};

const findBoxHit = (point, pixelBoxes) => {
    for (let index = pixelBoxes.length - 1; index >= 0; index -= 1) {
        const pixelBox = pixelBoxes[index];
        if (pointInRect(point, getResizeHandleRect(pixelBox))) {
            return { type: 'resize', index };
        }
        if (pointInRect(point, pixelBox)) {
            return { type: 'move', index };
        }
    }
    return null;
};

export default function DatasetViewerWorkspace({ dataset, csvName, appendLog, onDatasetChanged }) {
    const canvasRef = useRef(null);
    const imageRef = useRef(null);
    const dragRef = useRef(null);
    const appendLogRef = useRef(appendLog);
    const entriesRef = useRef([]);
    const currentIndexRef = useRef(0);
    const originalBoxesRef = useRef(new Map());
    const currentBoxesRef = useRef([]);

    const [entries, setEntries] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isSaving, setIsSaving] = useState(false);
    const [isDeleting, setIsDeleting] = useState(false);
    const [imageMeta, setImageMeta] = useState({ ready: false, width: 0, height: 0 });
    const [currentIndex, setCurrentIndex] = useState(0);
    const [currentBox, setCurrentBox] = useState(null);
    const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);
    const [selectedClass, setSelectedClass] = useState(DEFAULT_CLASS);
    const [clickBox, setClickBox] = useState(96);
    const [status, setStatus] = useState('Choose a dataset to start reviewing images.');
    const [toast, setToast] = useState(null);

    const currentEntry = entries[currentIndex] || null;
    const currentBoxes = currentEntry?.boxes || [];

    useEffect(() => { entriesRef.current = entries; }, [entries]);
    useEffect(() => { currentIndexRef.current = currentIndex; }, [currentIndex]);
    useEffect(() => { currentBoxesRef.current = currentBoxes; }, [currentBoxes]);
    useEffect(() => { appendLogRef.current = appendLog; }, [appendLog]);
    useEffect(() => {
        if (!toast) return undefined;
        const timeout = window.setTimeout(() => setToast(null), 1800);
        return () => window.clearTimeout(timeout);
    }, [toast]);

    const setEntryBoxes = (entryIndex, updater, dirty = true) => {
        setEntries((prev) => prev.map((entry, index) => {
            if (index !== entryIndex) return entry;
            const nextBoxes = typeof updater === 'function' ? updater(entry.boxes || []) : updater;
            return {
                ...entry,
                boxes: nextBoxes,
                boxCount: nextBoxes.length,
                dirty: dirty ? true : false,
            };
        }));
    };

    const renderCanvas = (draftBox = currentBox) => {
        const canvas = canvasRef.current;
        const image = imageRef.current;
        if (!canvas || !image || !imageMeta.ready || !currentEntry?.imagePath) return;

        if (canvas.width !== imageMeta.width || canvas.height !== imageMeta.height) {
            canvas.width = imageMeta.width;
            canvas.height = imageMeta.height;
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

        currentBoxesRef.current.forEach((savedBox, index) => {
            const pixelBox = yoloToPixels(savedBox, canvas.width, canvas.height);
            const classOption = getClassOption(savedBox.class_id, savedBox.class_name);
            drawBox(ctx, pixelBox, classOption.color, `${classOption.name} ${index + 1}`, { selected: index === selectedBoxIndex });
        });

        if (draftBox) {
            const classOption = getClassOption(draftBox.class_id, draftBox.class_name);
            drawBox(ctx, draftBox, classOption.color, `${classOption.name} draft`);
        }
    };

    useEffect(() => {
        if (!dataset?.path || !csvName) {
            setEntries([]);
            setCurrentIndex(0);
            setCurrentBox(null);
            setSelectedBoxIndex(null);
            setImageMeta({ ready: false, width: 0, height: 0 });
            setStatus('Choose a dataset to start reviewing images.');
            return undefined;
        }

        let cancelled = false;
        const loadEntries = async () => {
            setIsLoading(true);
            const result = await window.electronAPI.listDatasetImages(dataset.path, csvName);
            if (cancelled) return;

            if (result.error) {
                setEntries([]);
                setStatus(result.error);
                appendLogRef.current?.(`[Data Viewer Error] ${result.error}`);
                setIsLoading(false);
                return;
            }

            const nextEntries = (result.entries || []).map((entry) => ({
                ...entry,
                boxes: cloneBoxes(entry.boxes || []),
                dirty: false,
            }));
            originalBoxesRef.current = new Map(nextEntries.map((entry) => [entry.filename, cloneBoxes(entry.boxes || [])]));
            setEntries(nextEntries);
            setCurrentIndex(0);
            setCurrentBox(null);
            setSelectedBoxIndex(null);
            setStatus(nextEntries.length > 0 ? 'Dataset loaded. Select an image, then edit, save, or delete it.' : 'No images were found in this dataset.');
            appendLogRef.current?.(`[Data Viewer] Loaded ${nextEntries.length} image(s) from ${dataset.name}.`);
            setIsLoading(false);
        };

        loadEntries();
        return () => { cancelled = true; };
    }, [csvName, dataset?.name, dataset?.path]);

    useEffect(() => {
        setCurrentBox(null);
        setSelectedBoxIndex(null);
    }, [currentIndex]);

    useEffect(() => {
        if (currentBox) {
            setCurrentBox((prev) => (prev ? { ...prev, class_id: selectedClass.id, class_name: selectedClass.name } : prev));
        }
    }, [selectedClass]);

    useEffect(() => {
        if (!currentEntry?.imagePath) {
            setImageMeta({ ready: false, width: 0, height: 0 });
            return;
        }

        const image = imageRef.current;
        if (!image) return;

        const handleLoad = () => {
            setImageMeta({
                ready: true,
                width: image.naturalWidth || image.width || 1280,
                height: image.naturalHeight || image.height || 720,
            });
            setStatus(currentEntry.missingImage ? `${currentEntry.filename} is missing from the images folder.` : `Reviewing ${currentEntry.filename}.`);
            window.requestAnimationFrame(() => renderCanvas(null));
        };

        if (image.complete) {
            handleLoad();
            return undefined;
        }

        image.addEventListener('load', handleLoad);
        return () => image.removeEventListener('load', handleLoad);
    }, [currentEntry?.filename, currentEntry?.imagePath]);

    useEffect(() => {
        if (imageMeta.ready) {
            renderCanvas(currentBox);
        }
    }, [currentBox, currentBoxes, imageMeta.ready, selectedBoxIndex]);

    const pointFromEvent = (event) => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        const rect = canvas.getBoundingClientRect();
        return {
            x: clamp((event.clientX - rect.left) * (canvas.width / rect.width), 0, canvas.width),
            y: clamp((event.clientY - rect.top) * (canvas.height / rect.height), 0, canvas.height),
        };
    };

    const updateSelectedBoxPixels = (boxIndex, pixelBox) => {
        const clamped = clampBox(pixelBox, imageMeta.width, imageMeta.height);
        const current = currentBoxesRef.current[boxIndex];
        if (!current) return;

        setEntryBoxes(currentIndexRef.current, (boxes) => boxes.map((box, index) => (
            index === boxIndex
                ? {
                    ...box,
                    ...pixelsToYolo(clamped, imageMeta.width, imageMeta.height),
                }
                : box
        )));
    };

    const onPointerDown = (event) => {
        if (!imageMeta.ready || !currentEntry?.imagePath || event.button !== 0) return;

        const point = pointFromEvent(event);
        if (!point) return;

        const pixelBoxes = currentBoxesRef.current.map((box) => yoloToPixels(box, imageMeta.width, imageMeta.height));
        const hit = findBoxHit(point, pixelBoxes);
        if (hit) {
            const hitBox = currentBoxesRef.current[hit.index];
            setSelectedBoxIndex(hit.index);
            setSelectedClass(getClassOption(hitBox.class_id, hitBox.class_name));
            dragRef.current = {
                mode: hit.type,
                startPoint: point,
                boxIndex: hit.index,
                originBox: pixelBoxes[hit.index],
            };
            setCurrentBox(null);
            setStatus(hit.type === 'resize' ? 'Resizing the selected box...' : 'Moving the selected box...');
            return;
        }

        dragRef.current = { mode: 'draw', startPoint: point };
        const seed = {
            x: point.x,
            y: point.y,
            width: 0,
            height: 0,
            class_id: selectedClass.id,
            class_name: selectedClass.name,
        };
        setSelectedBoxIndex(null);
        setCurrentBox(seed);
        setStatus(`Drawing a ${selectedClass.name} box...`);
        renderCanvas(seed);
    };

    const onPointerMove = (event) => {
        if (!dragRef.current || !imageMeta.ready) return;

        const point = pointFromEvent(event);
        if (!point) return;

        const interaction = dragRef.current;
        if (interaction.mode === 'draw') {
            const box = {
                x: Math.min(interaction.startPoint.x, point.x),
                y: Math.min(interaction.startPoint.y, point.y),
                width: Math.abs(point.x - interaction.startPoint.x),
                height: Math.abs(point.y - interaction.startPoint.y),
                class_id: selectedClass.id,
                class_name: selectedClass.name,
            };
            setCurrentBox(box);
            renderCanvas(box);
            return;
        }

        if (interaction.mode === 'move') {
            const deltaX = point.x - interaction.startPoint.x;
            const deltaY = point.y - interaction.startPoint.y;
            updateSelectedBoxPixels(interaction.boxIndex, {
                ...interaction.originBox,
                x: interaction.originBox.x + deltaX,
                y: interaction.originBox.y + deltaY,
            });
            return;
        }

        if (interaction.mode === 'resize') {
            updateSelectedBoxPixels(interaction.boxIndex, {
                ...interaction.originBox,
                width: Math.max(MIN_BOX, point.x - interaction.originBox.x),
                height: Math.max(MIN_BOX, point.y - interaction.originBox.y),
            });
        }
    };

    const onPointerUp = (event) => {
        if (!dragRef.current || !imageMeta.ready) return;

        const interaction = dragRef.current;
        dragRef.current = null;

        if (interaction.mode !== 'draw') {
            setStatus('Box updated locally. Save changes when this image looks correct.');
            renderCanvas(null);
            return;
        }

        const point = pointFromEvent(event) || interaction.startPoint;
        const rawBox = {
            x: Math.min(interaction.startPoint.x, point.x),
            y: Math.min(interaction.startPoint.y, point.y),
            width: Math.abs(point.x - interaction.startPoint.x),
            height: Math.abs(point.y - interaction.startPoint.y),
        };

        const nextBox = rawBox.width < MIN_BOX || rawBox.height < MIN_BOX
            ? clampBox({
                x: point.x - clickBox / 2,
                y: point.y - clickBox / 2,
                width: clickBox,
                height: clickBox,
                class_id: selectedClass.id,
                class_name: selectedClass.name,
            }, imageMeta.width, imageMeta.height)
            : clampBox({
                ...rawBox,
                class_id: selectedClass.id,
                class_name: selectedClass.name,
            }, imageMeta.width, imageMeta.height);

        setCurrentBox(nextBox);
        setStatus(`${selectedClass.name} draft ready. Add it, save this image, or keep editing.`);
        renderCanvas(nextBox);
    };

    const addDraftBox = () => {
        if (!currentBox || !imageMeta.ready) return;
        const nextBox = {
            ...pixelsToYolo(currentBox, imageMeta.width, imageMeta.height),
            class_id: currentBox.class_id ?? selectedClass.id,
            class_name: currentBox.class_name ?? selectedClass.name,
            confidence: 1,
        };
        setEntryBoxes(currentIndexRef.current, (boxes) => [...boxes, nextBox]);
        setSelectedBoxIndex(currentBoxesRef.current.length);
        setCurrentBox(null);
        setStatus(`Added a ${nextBox.class_name} box locally. Save this image to write it to disk.`);
        setToast({ tone: 'success', message: `Added ${nextBox.class_name} box to ${currentEntry.filename}.` });
    };

    const deleteSelectedBox = () => {
        if (selectedBoxIndex === null) {
            setStatus('Select a saved box first.');
            return;
        }

        setEntryBoxes(currentIndexRef.current, (boxes) => boxes.filter((_, index) => index !== selectedBoxIndex));
        setSelectedBoxIndex(null);
        setStatus('Removed the selected box locally. Save this image to persist the change.');
        setToast({ tone: 'success', message: `Removed a box from ${currentEntry.filename}.` });
    };

    const clearDraft = () => {
        setCurrentBox(null);
        setStatus('Draft box cleared.');
    };

    const revertCurrentImage = () => {
        if (!currentEntry) return;
        const original = cloneBoxes(originalBoxesRef.current.get(currentEntry.filename) || []);
        setEntries((prev) => prev.map((entry, index) => (
            index === currentIndexRef.current
                ? { ...entry, boxes: original, boxCount: original.length, dirty: false }
                : entry
        )));
        setCurrentBox(null);
        setSelectedBoxIndex(null);
        setStatus(`Reverted unsaved edits for ${currentEntry.filename}.`);
        setToast({ tone: 'success', message: `Reverted ${currentEntry.filename}.` });
    };

    const saveCurrentImage = async (advance = false) => {
        if (!currentEntry || isSaving || !currentEntry.imagePath) return;

        setIsSaving(true);
        const draftBoxes = currentBox ? [{
            ...pixelsToYolo(currentBox, imageMeta.width, imageMeta.height),
            class_id: currentBox.class_id ?? selectedClass.id,
            class_name: currentBox.class_name ?? selectedClass.name,
            confidence: 1,
        }] : [];
        const boxesToSave = [...currentBoxesRef.current, ...draftBoxes];

        const result = await window.electronAPI.saveDatasetImageAnnotations({
            datasetPath: dataset.path,
            csvName,
            filename: currentEntry.filename,
            boxes: boxesToSave,
            bboxSource: 'editor_box',
        });

        if (result.success) {
            const persistedBoxes = cloneBoxes(result.boxes || boxesToSave);
            originalBoxesRef.current.set(currentEntry.filename, cloneBoxes(persistedBoxes));
            setEntries((prev) => prev.map((entry, index) => (
                index === currentIndexRef.current
                    ? { ...entry, boxes: persistedBoxes, boxCount: persistedBoxes.length, dirty: false }
                    : entry
            )));
            setCurrentBox(null);
            setSelectedBoxIndex(persistedBoxes.length > 0 ? clamp(selectedBoxIndex ?? 0, 0, persistedBoxes.length - 1) : null);
            setStatus(`Saved ${currentEntry.filename} with ${persistedBoxes.length} box(es).`);
            setToast({ tone: 'success', message: `Saved ${currentEntry.filename}.` });
            appendLogRef.current?.(`[Data Viewer] Saved ${currentEntry.filename} (${persistedBoxes.length} box(es)).`);
            onDatasetChanged?.();
            if (advance && currentIndexRef.current < entriesRef.current.length - 1) {
                setCurrentIndex((prev) => prev + 1);
            }
        } else {
            setStatus(result.error || 'Save failed.');
            setToast({ tone: 'error', message: result.error || 'Save failed.' });
            appendLogRef.current?.(`[Data Viewer Error] ${result.error}`);
        }

        setIsSaving(false);
    };

    const deleteCurrentImage = async () => {
        if (!currentEntry || isDeleting) return;
        if (!window.confirm(`Delete ${currentEntry.filename} from this dataset?`)) return;

        setIsDeleting(true);
        const result = await window.electronAPI.deleteDatasetImage(dataset.path, csvName, currentEntry.filename);
        if (result.success) {
            originalBoxesRef.current.delete(currentEntry.filename);
            setEntries((prev) => {
                const nextEntries = prev.filter((entry) => entry.filename !== currentEntry.filename);
                const nextIndex = clamp(currentIndexRef.current, 0, Math.max(0, nextEntries.length - 1));
                window.setTimeout(() => setCurrentIndex(nextIndex), 0);
                return nextEntries;
            });
            setCurrentBox(null);
            setSelectedBoxIndex(null);
            setStatus(`Deleted ${currentEntry.filename} from the dataset.`);
            setToast({ tone: 'success', message: `Deleted ${currentEntry.filename}.` });
            appendLogRef.current?.(`[Data Viewer] Deleted ${currentEntry.filename}.`);
            onDatasetChanged?.();
        } else {
            setStatus(result.error || 'Could not delete the image.');
            setToast({ tone: 'error', message: result.error || 'Could not delete the image.' });
            appendLogRef.current?.(`[Data Viewer Error] ${result.error}`);
        }
        setIsDeleting(false);
    };

    const applyClassSelection = (option) => {
        setSelectedClass(option);
        if (selectedBoxIndex === null || currentBox) return;

        setEntryBoxes(currentIndexRef.current, (boxes) => boxes.map((box, index) => (
            index === selectedBoxIndex
                ? { ...box, class_id: option.id, class_name: option.name }
                : box
        )));
        setStatus(`Changed box ${selectedBoxIndex + 1} to ${option.name}. Save this image to persist the change.`);
    };

    useEffect(() => {
        const onKeyDown = (event) => {
            const tag = event.target?.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
            if (!currentEntry) return;

            if (event.key === 'ArrowRight') {
                event.preventDefault();
                setCurrentIndex((prev) => clamp(prev + 1, 0, Math.max(0, entriesRef.current.length - 1)));
                return;
            }

            if (event.key === 'ArrowLeft') {
                event.preventDefault();
                setCurrentIndex((prev) => clamp(prev - 1, 0, Math.max(0, entriesRef.current.length - 1)));
                return;
            }

            if (event.key === 'Enter') {
                event.preventDefault();
                if (currentBox) {
                    addDraftBox();
                } else {
                    saveCurrentImage(false);
                }
                return;
            }

            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
                event.preventDefault();
                saveCurrentImage(false);
                return;
            }

            if (event.key.toLowerCase() === 'c') {
                event.preventDefault();
                clearDraft();
                return;
            }

            if (event.key.toLowerCase() === 'r') {
                event.preventDefault();
                revertCurrentImage();
                return;
            }

            if (event.key === 'Delete' || event.key === 'Backspace') {
                event.preventDefault();
                if (currentBox) {
                    clearDraft();
                } else {
                    deleteSelectedBox();
                }
                return;
            }

            if (event.key === '1') {
                event.preventDefault();
                applyClassSelection(CLASS_OPTIONS[0]);
                return;
            }

            if (event.key === '2') {
                event.preventDefault();
                applyClassSelection(CLASS_OPTIONS[1]);
            }
        };

        window.addEventListener('keydown', onKeyDown);
        return () => window.removeEventListener('keydown', onKeyDown);
    }, [currentBox, currentEntry, selectedBoxIndex]);

    if (!dataset?.path) {
        return (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 text-sm text-slate-400">
                Choose a dataset and CSV above to open the in-app image editor.
            </div>
        );
    }

    if (isLoading) {
        return (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 text-sm text-slate-300">
                Loading dataset images...
            </div>
        );
    }

    if (!currentEntry) {
        return (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 text-sm text-slate-400">
                No images were found for this dataset and CSV.
            </div>
        );
    }

    return (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.65fr)_360px] 2xl:grid-cols-[minmax(0,1.85fr)_400px] gap-4 lg:gap-5">
            <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5 shadow-2xl">
                <div className="flex flex-col gap-4">
                    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
                        <div>
                            <div className="text-xs font-black uppercase tracking-[0.25em] text-sky-400">Dataset Viewer</div>
                            <h3 className="text-xl font-bold text-white mt-1">{dataset.name}</h3>
                            <p className="text-sm text-slate-400 mt-1">{currentEntry.filename} - image {currentIndex + 1} / {entries.length}</p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                            <button onClick={() => setCurrentIndex((prev) => clamp(prev - 1, 0, Math.max(0, entries.length - 1)))} className="px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-semibold flex items-center gap-2 transition"><ArrowLeft size={16} />Prev</button>
                            <button onClick={() => saveCurrentImage(true)} disabled={isSaving || !currentEntry.imagePath} className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 rounded-lg text-sm font-semibold text-white flex items-center gap-2 transition"><Save size={16} />{isSaving ? 'Saving...' : 'Save & Next'}</button>
                            <button onClick={() => setCurrentIndex((prev) => clamp(prev + 1, 0, Math.max(0, entries.length - 1)))} className="px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm font-semibold flex items-center gap-2 transition">Next<ArrowRight size={16} /></button>
                        </div>
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-black/80 overflow-hidden p-2 sm:p-3">
                        <div className="w-full min-h-[300px] max-h-[68vh] flex items-center justify-center overflow-hidden rounded-xl bg-black">
                            {currentEntry.imagePath ? (
                                <canvas
                                    ref={canvasRef}
                                    onMouseDown={onPointerDown}
                                    onMouseMove={onPointerMove}
                                    onMouseUp={onPointerUp}
                                    onMouseLeave={onPointerUp}
                                    onContextMenu={(event) => {
                                        event.preventDefault();
                                        setSelectedBoxIndex(null);
                                        clearDraft();
                                    }}
                                    className="block w-full h-full object-contain cursor-crosshair"
                                />
                            ) : (
                                <div className="flex flex-col items-center justify-center gap-3 text-slate-500">
                                    <ImageIcon size={42} className="opacity-40" />
                                    <span>{currentEntry.filename} is missing from the images folder.</span>
                                </div>
                            )}
                        </div>
                        {currentEntry.imagePath && (
                            <img
                                key={currentEntry.filename}
                                ref={imageRef}
                                src={toFileUrl(currentEntry.imagePath)}
                                alt={currentEntry.filename}
                                className="hidden"
                            />
                        )}
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
                        <div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold mb-3">Label Class</div>
                        <div className="flex flex-wrap gap-3">
                            {CLASS_OPTIONS.map((option, index) => (
                                <button
                                    key={option.id}
                                    onClick={() => applyClassSelection(option)}
                                    className={`px-4 py-2.5 rounded-xl border text-sm font-semibold transition ${selectedClass.id === option.id ? option.accent : 'border-slate-800 bg-slate-900 text-slate-300 hover:bg-slate-800'}`}
                                >
                                    {option.name} ({index + 1})
                                </button>
                            ))}
                        </div>
                        <div className="text-sm text-slate-300 mt-3">
                            Current class target: <span className="font-semibold text-white">{selectedClass.name}</span>
                            {selectedBoxIndex !== null && !currentBox && (
                                <span className="text-slate-400"> - applied to selected saved box</span>
                            )}
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-3">
                        <label className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1">
                            <span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Click Box (px)</span>
                            <input type="number" min="24" max="400" value={clickBox} onChange={(event) => setClickBox(clamp(Number(event.target.value) || 96, 24, 400))} className="bg-transparent text-sm text-slate-100 focus:outline-none" />
                        </label>
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1 justify-center">
                            <span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Image Boxes</span>
                            <span className="text-sm text-slate-100">{currentBoxes.length}{currentBox ? ' + 1 draft' : ''}</span>
                        </div>
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1 justify-center">
                            <span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Image State</span>
                            <span className="text-sm text-slate-100">{currentEntry.dirty ? 'Unsaved edits' : 'Saved'}</span>
                        </div>
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1 justify-center">
                            <span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Selected Box</span>
                            <span className="text-sm text-slate-100">{selectedBoxIndex !== null ? `#${selectedBoxIndex + 1}` : 'None'}</span>
                        </div>
                        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 flex flex-col gap-1 justify-center">
                            <span className="text-[11px] uppercase tracking-[0.22em] text-slate-500 font-bold">Draft Box</span>
                            <span className="text-sm text-slate-100">{currentBox ? `${Math.round(currentBox.width)} x ${Math.round(currentBox.height)} px` : 'None'}</span>
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-3">
                        <button onClick={addDraftBox} disabled={!currentBox} className="px-4 py-2.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-60 rounded-xl text-sm font-bold text-white flex items-center gap-2 transition"><Plus size={16} />Add Draft Box</button>
                        <button onClick={() => saveCurrentImage(false)} disabled={isSaving || !currentEntry.imagePath || (!currentEntry.dirty && !currentBox)} className="px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 rounded-xl text-sm font-bold text-white flex items-center gap-2 transition"><Save size={16} />Save Image</button>
                        <button onClick={deleteSelectedBox} disabled={selectedBoxIndex === null} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-60 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><Trash2 size={16} />Delete Selected Box</button>
                        <button onClick={clearDraft} disabled={!currentBox} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-60 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><RotateCcw size={16} />Clear Draft</button>
                        <button onClick={revertCurrentImage} disabled={!currentEntry.dirty && !currentBox} className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-60 rounded-xl text-sm font-semibold flex items-center gap-2 transition"><RotateCcw size={16} />Revert Unsaved</button>
                        <button onClick={deleteCurrentImage} disabled={isDeleting} className="px-4 py-2.5 bg-rose-600 hover:bg-rose-500 disabled:opacity-60 rounded-xl text-sm font-bold text-white flex items-center gap-2 transition"><Trash2 size={16} />{isDeleting ? 'Deleting...' : 'Delete Image'}</button>
                    </div>

                    <div className="rounded-2xl border border-slate-800 bg-slate-950/80 p-4">
                        <div className="flex flex-wrap gap-6 text-sm">
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Status</div><div className="text-slate-200 mt-1">{status}</div></div>
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Dataset Images</div><div className="text-slate-200 mt-1">{entries.length}</div></div>
                            <div><div className="text-[11px] uppercase tracking-[0.24em] text-slate-500 font-bold">Saved Boxes</div><div className="text-slate-200 mt-1">{currentBoxes.length}</div></div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex flex-col gap-4 xl:max-w-[400px]">
                <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                    <div className="flex items-center gap-2 text-sky-400 mb-3"><CircleHelp size={16} /><span className="text-xs font-black uppercase tracking-[0.25em]">How To Edit</span></div>
                    <div className="space-y-2 text-sm text-slate-300">
                        <div>Click an existing box to select it.</div>
                        <div>Drag inside a selected box to move it.</div>
                        <div>Drag the small bottom-right handle to resize it.</div>
                        <div>Drag on empty space to draw a new draft box, or single-click for a quick square box.</div>
                        <div>Add Draft Box to stage a new box locally, then Save Image to write the full label file back to disk.</div>
                        <div>Delete Image removes the image, its YOLO label file, and its CSV rows from the dataset.</div>
                    </div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                    <div className="flex items-center gap-2 text-indigo-400 mb-3"><Keyboard size={16} /><span className="text-xs font-black uppercase tracking-[0.25em]">Keyboard Controls</span></div>
                    <div className="space-y-2 text-sm text-slate-300">
                        <div><span className="text-white font-semibold">1</span> sets the class to enemy.</div>
                        <div><span className="text-white font-semibold">2</span> sets the class to player.</div>
                        <div><span className="text-white font-semibold">Enter</span> adds the draft box, or saves the image if there is no draft.</div>
                        <div><span className="text-white font-semibold">Ctrl/Cmd + S</span> saves the current image.</div>
                        <div><span className="text-white font-semibold">Delete / Backspace</span> removes the selected box, or clears the draft.</div>
                        <div><span className="text-white font-semibold">C</span> clears the draft box.</div>
                        <div><span className="text-white font-semibold">R</span> reverts unsaved changes for the current image.</div>
                        <div><span className="text-white font-semibold">Arrow Left / Right</span> moves through the image list.</div>
                    </div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                    <div className="flex items-center gap-2 text-emerald-400 mb-3"><ImageIcon size={16} /><span className="text-xs font-black uppercase tracking-[0.25em]">Image List</span></div>
                    <div className="space-y-2 max-h-[360px] overflow-y-auto pr-1">
                        {entries.map((entry, index) => (
                            <button
                                key={entry.filename}
                                onClick={() => setCurrentIndex(index)}
                                className={`w-full text-left rounded-xl border px-3 py-3 transition ${index === currentIndex ? 'border-sky-500/40 bg-sky-500/10' : 'border-slate-800 bg-slate-950/70 hover:bg-slate-900'}`}
                            >
                                <div className="flex items-center justify-between gap-3">
                                    <div className="min-w-0">
                                        <div className="truncate text-sm font-semibold text-slate-100">{entry.filename}</div>
                                        <div className="text-xs text-slate-500 mt-1">{entry.boxCount} box(es){entry.missingImage ? ' - missing image' : ''}</div>
                                    </div>
                                    {entry.dirty && <span className="text-[10px] uppercase tracking-[0.18em] text-amber-300">Unsaved</span>}
                                </div>
                            </button>
                        ))}
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
