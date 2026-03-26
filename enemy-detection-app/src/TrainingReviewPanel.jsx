import { useState } from 'react';
import { ArrowLeft, ArrowRight, BarChart3, CheckCircle2, Image as ImageIcon, Search } from 'lucide-react';

const toFileUrl = (filePath) => {
    if (!filePath) return '';
    const normalized = filePath.replace(/\\/g, '/');
    return encodeURI(/^[A-Za-z]:\//.test(normalized) ? `file:///${normalized}` : `file://${normalized}`);
};

const formatMetric = (value) => {
    if (typeof value !== 'number' || Number.isNaN(value)) return 'n/a';
    return `${(value * 100).toFixed(1)}%`;
};

const formatBox = (item) => (
    `[${item.x_center.toFixed(3)}, ${item.y_center.toFixed(3)}] ${item.width.toFixed(3)} x ${item.height.toFixed(3)}`
);

const STATUS_STYLE = {
    match: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100',
    missed_ground_truth: 'border-amber-400/30 bg-amber-500/10 text-amber-100',
    extra_prediction: 'border-sky-400/30 bg-sky-500/10 text-sky-100',
    false_positive: 'border-rose-500/30 bg-rose-500/10 text-rose-100',
    mixed: 'border-orange-500/30 bg-orange-500/10 text-orange-100',
    negative_correct: 'border-slate-500/30 bg-slate-500/10 text-slate-200',
};

const formatStatus = (status) => {
    if (!status) return 'Unknown';
    return status.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
};

const getStatusChrome = (status) => STATUS_STYLE[status] || 'border-slate-700 bg-slate-800 text-slate-200';

export default function TrainingReviewPanel({ summary, reviewManifest }) {
    const [currentIndex, setCurrentIndex] = useState(0);
    const entries = Array.isArray(reviewManifest?.entries) ? reviewManifest.entries : [];
    const safeIndex = entries.length ? Math.min(currentIndex, entries.length - 1) : 0;
    const currentEntry = entries[safeIndex] || null;

    if (!summary) {
        return (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-400">
                Train a model to populate held-out metrics and review images here.
            </div>
        );
    }

    const evaluation = summary.evaluation || {};
    const metrics = evaluation.metrics || reviewManifest?.aggregate_metrics || {};
    const reviewSummary = evaluation.review_summary || reviewManifest?.summary || {};
    const imageSrc = currentEntry?.review_image_path || currentEntry?.image_path || null;

    return (
        <div className="flex flex-col gap-6">
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5">
                <div className="flex items-center gap-2 text-rose-400 mb-4">
                    <BarChart3 size={16} />
                    <span className="text-xs font-black uppercase tracking-[0.24em]">Held-Out Evaluation</span>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                    <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-rose-200">Precision</div>
                        <div className="mt-2 text-2xl font-black text-white">{formatMetric(metrics.precision)}</div>
                    </div>
                    <div className="rounded-xl border border-orange-400/20 bg-orange-500/10 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-orange-100">Recall</div>
                        <div className="mt-2 text-2xl font-black text-white">{formatMetric(metrics.recall)}</div>
                    </div>
                    <div className="rounded-xl border border-sky-400/20 bg-sky-500/10 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-sky-100">mAP50</div>
                        <div className="mt-2 text-2xl font-black text-white">{formatMetric(metrics.map50)}</div>
                    </div>
                    <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/10 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-emerald-100">mAP50-95</div>
                        <div className="mt-2 text-2xl font-black text-white">{formatMetric(metrics.map50_95)}</div>
                    </div>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mt-3">
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Train Images</div>
                        <div className="mt-2 text-xl font-black text-white">{summary.dataset?.train_images ?? 'n/a'}</div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Test Images</div>
                        <div className="mt-2 text-xl font-black text-white">{summary.dataset?.test_images ?? summary.dataset?.val_images ?? 'n/a'}</div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Matches</div>
                        <div className="mt-2 text-xl font-black text-white">{reviewSummary.matched_boxes ?? 'n/a'}</div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Missed GT</div>
                        <div className="mt-2 text-xl font-black text-white">{reviewSummary.missed_ground_truth_boxes ?? 'n/a'}</div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">False Positives</div>
                        <div className="mt-2 text-xl font-black text-white">{reviewSummary.false_positive_boxes ?? 'n/a'}</div>
                    </div>
                </div>

                <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-sm text-slate-300">
                    <div>
                        Requested test split: <span className="font-semibold text-white">
                            {typeof summary.dataset?.test_split === 'number' ? `${(summary.dataset.test_split * 100).toFixed(0)}%` : 'external holdout'}
                        </span>
                    </div>
                    <div className="mt-1">
                        Review overlay legend: <span className="text-emerald-300 font-semibold">GT</span> boxes are green and <span className="text-rose-300 font-semibold">Pred</span> boxes are red.
                    </div>
                </div>
            </div>

            {!entries.length ? (
                <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-400">
                    No per-image held-out review manifest was returned for this run.
                </div>
            ) : (
                <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.5fr)_360px] gap-5">
                    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5">
                        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                                    <div>
                                        <div className="text-xs font-black uppercase tracking-[0.24em] text-rose-400">Held-Out Review</div>
                                        <h3 className="mt-1 text-xl font-bold text-white">{currentEntry?.filename || 'No image selected'}</h3>
                                        <p className="text-sm text-slate-400 mt-1">
                                            Image {safeIndex + 1} / {entries.length}
                                        </p>
                                    </div>
                            <div className="flex gap-2">
                                <button
                                    onClick={() => setCurrentIndex((prev) => Math.max(0, prev - 1))}
                                    disabled={safeIndex === 0}
                                    className="px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-lg text-sm font-semibold flex items-center gap-2 transition"
                                >
                                    <ArrowLeft size={16} />Prev
                                </button>
                                <button
                                    onClick={() => setCurrentIndex((prev) => Math.min(entries.length - 1, prev + 1))}
                                    disabled={safeIndex >= entries.length - 1}
                                    className="px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-lg text-sm font-semibold flex items-center gap-2 transition"
                                >
                                    Next<ArrowRight size={16} />
                                </button>
                            </div>
                        </div>

                        <div className="rounded-2xl border border-slate-800 bg-black/80 overflow-hidden p-3">
                            <div className="w-full min-h-[340px] max-h-[68vh] flex items-center justify-center overflow-hidden rounded-xl bg-black">
                                {imageSrc ? (
                                    <img
                                        src={toFileUrl(imageSrc)}
                                        alt={currentEntry.filename}
                                        className="max-w-full max-h-[65vh] object-contain rounded"
                                    />
                                ) : (
                                    <div className="flex flex-col items-center gap-3 text-slate-500">
                                        <ImageIcon size={42} className="opacity-40" />
                                        <span>No review image was generated for this entry.</span>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
                            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">Status</div>
                                <div className={`mt-2 inline-flex rounded-full px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] border ${getStatusChrome(currentEntry?.status)}`}>
                                    {formatStatus(currentEntry?.status)}
                                </div>
                            </div>
                            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">GT Boxes</div>
                                <div className="mt-2 text-xl font-black text-white">{currentEntry?.counts?.ground_truth ?? 0}</div>
                            </div>
                            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">Predictions</div>
                                <div className="mt-2 text-xl font-black text-white">{currentEntry?.counts?.predictions ?? 0}</div>
                            </div>
                            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">Matched</div>
                                <div className="mt-2 text-xl font-black text-white">{currentEntry?.counts?.matched ?? 0}</div>
                            </div>
                        </div>
                    </div>

                    <div className="flex flex-col gap-4">
                        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                            <div className="flex items-center gap-2 text-orange-300 mb-3">
                                <Search size={16} />
                                <span className="text-xs font-black uppercase tracking-[0.24em]">Current Image</span>
                            </div>
                            <div className="space-y-2 text-sm text-slate-300">
                                <div>Missed GT: <span className="font-semibold text-white">{currentEntry?.counts?.missed_ground_truth ?? 0}</span></div>
                                <div>False Positives: <span className="font-semibold text-white">{currentEntry?.counts?.false_positives ?? 0}</span></div>
                                <div>Image Size: <span className="font-semibold text-white">{currentEntry?.image_size?.width ?? 0} x {currentEntry?.image_size?.height ?? 0}</span></div>
                            </div>
                        </div>

                        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                            <div className="flex items-center gap-2 text-emerald-300 mb-3">
                                <CheckCircle2 size={16} />
                                <span className="text-xs font-black uppercase tracking-[0.24em]">Ground Truth</span>
                            </div>
                            <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                                {currentEntry?.ground_truth?.length ? currentEntry.ground_truth.map((item, index) => (
                                    <div key={`gt-${index}`} className="rounded-xl border border-slate-800 bg-slate-950/80 p-3">
                                        <div className="text-sm font-semibold text-white">{item.class_name}</div>
                                        <div className="text-xs text-slate-400 mt-1">{formatBox(item)}</div>
                                    </div>
                                )) : (
                                    <div className="text-sm text-slate-500">No ground-truth boxes.</div>
                                )}
                            </div>
                        </div>

                        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                            <div className="flex items-center gap-2 text-rose-300 mb-3">
                                <ImageIcon size={16} />
                                <span className="text-xs font-black uppercase tracking-[0.24em]">Predictions</span>
                            </div>
                            <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                                {currentEntry?.predictions?.length ? currentEntry.predictions.map((item, index) => (
                                    <div key={`pred-${index}`} className="rounded-xl border border-slate-800 bg-slate-950/80 p-3">
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="text-sm font-semibold text-white">{item.class_name}</div>
                                            <div className="text-xs font-bold text-rose-200">{(item.confidence * 100).toFixed(1)}%</div>
                                        </div>
                                        <div className="text-xs text-slate-400 mt-1">{formatBox(item)}</div>
                                    </div>
                                )) : (
                                    <div className="text-sm text-slate-500">No predictions above threshold.</div>
                                )}
                            </div>
                        </div>

                        <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-4 lg:p-5">
                            <div className="flex items-center gap-2 text-slate-300 mb-3">
                                <ImageIcon size={16} />
                                <span className="text-xs font-black uppercase tracking-[0.24em]">Image List</span>
                            </div>
                            <div className="space-y-2 max-h-[260px] overflow-y-auto pr-1">
                                {entries.map((entry, index) => (
                                    <button
                                        key={`${entry.filename}-${index}`}
                                        onClick={() => setCurrentIndex(index)}
                                        className={`w-full text-left rounded-xl border px-3 py-3 transition ${
                                            index === safeIndex
                                                ? 'border-rose-500/40 bg-rose-500/10'
                                                : 'border-slate-800 bg-slate-950/70 hover:bg-slate-900'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0">
                                                <div className="truncate text-sm font-semibold text-slate-100">{entry.filename}</div>
                                                <div className="text-xs text-slate-500 mt-1">
                                                    GT {entry.counts?.ground_truth ?? 0} | Pred {entry.counts?.predictions ?? 0}
                                                </div>
                                            </div>
                                            <span className={`shrink-0 rounded-full border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.2em] ${getStatusChrome(entry.status)}`}>
                                                {formatStatus(entry.status)}
                                            </span>
                                        </div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
