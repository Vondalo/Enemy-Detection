import { useEffect, useState } from 'react';
import { ArrowLeft, ArrowRight, BarChart3, Image as ImageIcon, LineChart, Search } from 'lucide-react';

const toFileUrl = (filePath) => {
    if (!filePath) return '';
    const normalized = filePath.replace(/\\/g, '/');
    return encodeURI(/^[A-Za-z]:\//.test(normalized) ? `file:///${normalized}` : `file://${normalized}`);
};

const formatMetric = (value) => (typeof value === 'number' && !Number.isNaN(value) ? `${(value * 100).toFixed(1)}%` : 'n/a');
const formatDateTime = (value) => {
    if (!value) return 'n/a';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? 'n/a' : parsed.toLocaleString();
};
const formatStatus = (value) => String(value || 'unknown').replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
const formatMetricKey = (value) => String(value || 'Metric').replace(/[/_]+/g, ' ').replace(/\(b\)/gi, ' Box').replace(/\bmap\b/gi, 'mAP').replace(/\b\w/g, (char) => char.toUpperCase());
const STATUS_STYLE = {
    match: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-100',
    missed_ground_truth: 'border-amber-400/30 bg-amber-500/10 text-amber-100',
    extra_prediction: 'border-sky-400/30 bg-sky-500/10 text-sky-100',
    false_positive: 'border-rose-500/30 bg-rose-500/10 text-rose-100',
    mixed: 'border-orange-500/30 bg-orange-500/10 text-orange-100',
    negative_correct: 'border-slate-500/30 bg-slate-500/10 text-slate-200',
};
const getStatusChrome = (value) => STATUS_STYLE[value] || 'border-slate-700 bg-slate-800 text-slate-200';

const StatCard = ({ label, value, chrome }) => (
    <div className={`rounded-xl border p-4 ${chrome}`}>
        <div className="text-[11px] font-bold uppercase tracking-[0.2em]">{label}</div>
        <div className="mt-2 text-2xl font-black text-white">{value}</div>
    </div>
);

const Section = ({ title, icon, children }) => (
    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5">
        <div className="flex items-center gap-2 text-rose-300 mb-4">
            {icon}
            <span className="text-xs font-black uppercase tracking-[0.24em]">{title}</span>
        </div>
        {children}
    </div>
);

export default function TrainingReviewPanel({ summary, reviewManifest, artifactManifest, trainingResultsRows }) {
    const [reviewIndex, setReviewIndex] = useState(0);
    const [artifactIndex, setArtifactIndex] = useState(0);

    const metrics = summary?.evaluation?.metrics || reviewManifest?.aggregate_metrics || {};
    const reviewSummary = summary?.evaluation?.review_summary || reviewManifest?.summary || {};
    const reviewEntries = Array.isArray(reviewManifest?.entries) ? reviewManifest.entries : [];
    const safeReviewIndex = reviewEntries.length ? Math.min(reviewIndex, reviewEntries.length - 1) : 0;
    const currentReview = reviewEntries[safeReviewIndex] || null;
    const reviewImageSrc = currentReview?.review_image_path || currentReview?.image_path || null;

    const artifacts = Array.isArray(artifactManifest?.images) ? artifactManifest.images : [];
    const safeArtifactIndex = artifacts.length ? Math.min(artifactIndex, artifacts.length - 1) : 0;
    const currentArtifact = artifacts[safeArtifactIndex] || null;
    const perClass = Array.isArray(metrics?.per_class) ? metrics.per_class : [];
    const rawMetricRows = Object.entries(metrics?.results_dict || {});
    const epochRows = Array.isArray(trainingResultsRows) ? trainingResultsRows : [];
    const epochColumns = epochRows.length ? Object.keys(epochRows[0]) : [];
    const savedTables = Array.isArray(artifactManifest?.tables) ? artifactManifest.tables : [];
    const savedWeights = Array.isArray(artifactManifest?.weights) ? artifactManifest.weights : [];
    const modelLabel = summary?.model_choice || summary?.chosen_model || 'n/a';

    useEffect(() => setReviewIndex(0), [summary?.run_id, reviewEntries.length]);
    useEffect(() => setArtifactIndex(0), [summary?.run_id, artifactManifest?.generated_at, artifacts.length]);

    if (!summary) {
        return (
            <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-6 text-sm text-slate-400">
                Train a model or load one from the saved model history to populate saved plots, metrics, and held-out reviews here.
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-6">
            <Section title="Model Summary" icon={<BarChart3 size={16} />}>
                <div className="grid grid-cols-2 xl:grid-cols-6 gap-3">
                    <StatCard label="Precision" value={formatMetric(metrics?.precision)} chrome="border-rose-500/20 bg-rose-500/10 text-rose-200" />
                    <StatCard label="Recall" value={formatMetric(metrics?.recall)} chrome="border-orange-400/20 bg-orange-500/10 text-orange-100" />
                    <StatCard label="mAP50" value={formatMetric(metrics?.map50)} chrome="border-sky-400/20 bg-sky-500/10 text-sky-100" />
                    <StatCard label="mAP75" value={formatMetric(metrics?.map75)} chrome="border-indigo-400/20 bg-indigo-500/10 text-indigo-100" />
                    <StatCard label="mAP50-95" value={formatMetric(metrics?.map50_95)} chrome="border-emerald-400/20 bg-emerald-500/10 text-emerald-100" />
                    <StatCard label="Fitness" value={formatMetric(metrics?.fitness)} chrome="border-fuchsia-400/20 bg-fuchsia-500/10 text-fuchsia-100" />
                </div>

                <div className="grid grid-cols-1 xl:grid-cols-3 gap-3 mt-3 text-sm text-slate-300">
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Run</div>
                        <div className="mt-2 text-white font-semibold">{formatDateTime(summary?.created_at)}</div>
                        <div className="mt-1 text-slate-400 break-all">{modelLabel}</div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Training Source</div>
                        <div className="mt-2 text-white font-semibold">{summary?.training_source?.dataset_name || 'n/a'}</div>
                        <div className="mt-1 text-slate-400">{summary?.training_source?.csv_name || 'n/a'}</div>
                    </div>
                    <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                        <div className="text-[11px] font-bold uppercase tracking-[0.2em] text-slate-500">Artifacts</div>
                        <div className="mt-2 text-white font-semibold">{artifactManifest?.counts?.plotCount ?? 0} plots</div>
                        <div className="mt-1 text-slate-400">{savedTables.length} tables, {savedWeights.length} weights, {artifactManifest?.counts?.previewCount ?? 0} previews</div>
                    </div>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-6 gap-3 mt-3">
                    <StatCard label="Train Images" value={summary?.dataset?.train_images ?? 'n/a'} chrome="border-slate-700 bg-slate-950/70 text-slate-500" />
                    <StatCard label="Test Images" value={summary?.dataset?.test_images ?? summary?.dataset?.val_images ?? 'n/a'} chrome="border-slate-700 bg-slate-950/70 text-slate-500" />
                    <StatCard label="Matches" value={reviewSummary?.matched_boxes ?? 'n/a'} chrome="border-slate-700 bg-slate-950/70 text-slate-500" />
                    <StatCard label="Missed GT" value={reviewSummary?.missed_ground_truth_boxes ?? 'n/a'} chrome="border-slate-700 bg-slate-950/70 text-slate-500" />
                    <StatCard label="False Positives" value={reviewSummary?.false_positive_boxes ?? 'n/a'} chrome="border-slate-700 bg-slate-950/70 text-slate-500" />
                    <StatCard label="Review Conf" value={typeof summary?.evaluation?.confidence_threshold === 'number' ? `${(summary.evaluation.confidence_threshold * 100).toFixed(0)}%` : 'n/a'} chrome="border-slate-700 bg-slate-950/70 text-slate-500" />
                </div>
            </Section>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                <Section title="Per-Class Metrics" icon={<LineChart size={16} />}>
                    {!perClass.length ? <div className="text-sm text-slate-500">No per-class metrics were saved for this run.</div> : (
                        <div className="overflow-x-auto">
                            <table className="min-w-full text-sm text-slate-300">
                                <thead className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
                                    <tr>{Object.keys(perClass[0]).map((column) => <th key={column} className="px-3 py-2 text-left whitespace-nowrap">{column}</th>)}</tr>
                                </thead>
                                <tbody>{perClass.map((row, index) => <tr key={index} className="border-t border-slate-800">{Object.entries(row).map(([column, value]) => <td key={`${column}-${index}`} className="px-3 py-2 whitespace-nowrap">{typeof value === 'number' ? value.toFixed(6) : String(value)}</td>)}</tr>)}</tbody>
                            </table>
                        </div>
                    )}
                </Section>

                <Section title="Raw Metrics" icon={<BarChart3 size={16} />}>
                    {!rawMetricRows.length ? <div className="text-sm text-slate-500">No raw metric dictionary was saved for this run.</div> : (
                        <div className="space-y-2">
                            {rawMetricRows.map(([key, value]) => (
                                <div key={key} className="rounded-xl border border-slate-800 bg-slate-950/80 p-3 flex items-center justify-between gap-3">
                                    <div className="text-sm text-slate-300">{formatMetricKey(key)}</div>
                                    <div className="text-sm font-semibold text-white">{typeof value === 'number' ? value.toFixed(6) : String(value)}</div>
                                </div>
                            ))}
                        </div>
                    )}
                </Section>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.3fr)_420px] gap-5">
                <Section title="Saved Plots" icon={<ImageIcon size={16} />}>
                    {!artifacts.length ? <div className="text-sm text-slate-500">No saved training plots were found for this run.</div> : (
                        <div className="space-y-4">
                            <div className="rounded-2xl border border-slate-800 bg-black/80 overflow-hidden p-3">
                                <div className="w-full min-h-[320px] max-h-[60vh] flex items-center justify-center overflow-hidden rounded-xl bg-black">
                                    {currentArtifact?.path ? <img src={toFileUrl(currentArtifact.path)} alt={currentArtifact.label} className="max-w-full max-h-[56vh] object-contain rounded" /> : <div className="text-slate-500">No plot selected.</div>}
                                </div>
                            </div>
                            {currentArtifact && <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 text-xs text-slate-400"><div className="text-sm font-semibold text-white">{currentArtifact.label}</div><div className="mt-1 uppercase tracking-[0.2em] text-slate-500">{currentArtifact.category || 'image'}</div><div className="mt-2 break-all">{currentArtifact.path}</div></div>}
                        </div>
                    )}
                </Section>

                <div className="flex flex-col gap-4">
                    <Section title="Plot List" icon={<Search size={16} />}>
                        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                            {artifacts.length ? artifacts.map((item, index) => (
                                <button key={`${item.key}-${index}`} onClick={() => setArtifactIndex(index)} className={`w-full text-left rounded-xl border px-3 py-3 transition ${index === safeArtifactIndex ? 'border-amber-500/40 bg-amber-500/10' : 'border-slate-800 bg-slate-950/70 hover:bg-slate-900'}`}>
                                    <div className="truncate text-sm font-semibold text-slate-100">{item.label}</div>
                                    <div className="mt-1 text-xs text-slate-500 truncate">{item.relative_path}</div>
                                </button>
                            )) : <div className="text-sm text-slate-500">No plot images were archived.</div>}
                        </div>
                    </Section>

                    <Section title="Saved Files" icon={<BarChart3 size={16} />}>
                        <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                            {[...savedTables, ...savedWeights].length ? [...savedTables, ...savedWeights].map((item, index) => (
                                <div key={`${item.key}-${index}`} className="rounded-xl border border-slate-800 bg-slate-950/80 p-3">
                                    <div className="text-sm font-semibold text-white">{item.label}</div>
                                    <div className="mt-1 text-xs text-slate-500">{item.relative_path}</div>
                                    <div className="mt-2 text-[11px] text-slate-400 break-all">{item.path}</div>
                                </div>
                            )) : <div className="text-sm text-slate-500">No extra saved files were recorded for this run.</div>}
                        </div>
                    </Section>
                </div>
            </div>

            <Section title="Epoch History" icon={<LineChart size={16} />}>
                {!epochRows.length ? <div className="text-sm text-slate-500">No epoch-by-epoch training table was found for this run.</div> : (
                    <div className="overflow-auto max-h-[360px] rounded-xl border border-slate-800">
                        <table className="min-w-full text-sm text-slate-300">
                            <thead className="sticky top-0 bg-slate-950 text-[11px] uppercase tracking-[0.2em] text-slate-500">
                                <tr>{epochColumns.map((column) => <th key={column} className="px-3 py-2 text-left whitespace-nowrap">{column}</th>)}</tr>
                            </thead>
                            <tbody>{epochRows.map((row, index) => <tr key={index} className="border-t border-slate-800">{epochColumns.map((column) => <td key={`${column}-${index}`} className="px-3 py-2 whitespace-nowrap">{row[column] ?? ''}</td>)}</tr>)}</tbody>
                        </table>
                    </div>
                )}
            </Section>

            <Section title="Held-Out Review" icon={<ImageIcon size={16} />}>
                {!reviewEntries.length ? <div className="text-sm text-slate-500">No per-image held-out review manifest was returned for this run.</div> : (
                    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.4fr)_360px] gap-5">
                        <div>
                            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
                                <div>
                                    <div className="text-xl font-bold text-white">{currentReview?.filename || 'No image selected'}</div>
                                    <div className="text-sm text-slate-400 mt-1">Image {safeReviewIndex + 1} / {reviewEntries.length}</div>
                                </div>
                                <div className="flex gap-2">
                                    <button onClick={() => setReviewIndex((value) => Math.max(0, value - 1))} disabled={safeReviewIndex === 0} className="px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-lg text-sm font-semibold flex items-center gap-2 transition"><ArrowLeft size={16} />Prev</button>
                                    <button onClick={() => setReviewIndex((value) => Math.min(reviewEntries.length - 1, value + 1))} disabled={safeReviewIndex >= reviewEntries.length - 1} className="px-3 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-lg text-sm font-semibold flex items-center gap-2 transition">Next<ArrowRight size={16} /></button>
                                </div>
                            </div>
                            <div className="rounded-2xl border border-slate-800 bg-black/80 overflow-hidden p-3">
                                <div className="w-full min-h-[320px] max-h-[64vh] flex items-center justify-center overflow-hidden rounded-xl bg-black">
                                    {reviewImageSrc ? <img src={toFileUrl(reviewImageSrc)} alt={currentReview?.filename} className="max-w-full max-h-[60vh] object-contain rounded" /> : <div className="text-slate-500">No review image was generated for this entry.</div>}
                                </div>
                            </div>
                        </div>

                        <div className="space-y-4">
                            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">Current Image</div>
                                <div className={`mt-3 inline-flex rounded-full px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] border ${getStatusChrome(currentReview?.status)}`}>{formatStatus(currentReview?.status)}</div>
                                <div className="mt-3 text-sm text-slate-300">GT {currentReview?.counts?.ground_truth ?? 0} · Pred {currentReview?.counts?.predictions ?? 0} · Matched {currentReview?.counts?.matched ?? 0}</div>
                                <div className="mt-1 text-sm text-slate-400">Missed GT {currentReview?.counts?.missed_ground_truth ?? 0} · False Positives {currentReview?.counts?.false_positives ?? 0}</div>
                            </div>
                            <div className="rounded-xl border border-slate-700 bg-slate-950/70 p-4">
                                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500 font-bold">Image List</div>
                                <div className="space-y-2 max-h-[320px] overflow-y-auto pr-1 mt-3">
                                    {reviewEntries.map((entry, index) => (
                                        <button key={`${entry.filename}-${index}`} onClick={() => setReviewIndex(index)} className={`w-full text-left rounded-xl border px-3 py-3 transition ${index === safeReviewIndex ? 'border-rose-500/40 bg-rose-500/10' : 'border-slate-800 bg-slate-950/70 hover:bg-slate-900'}`}>
                                            <div className="truncate text-sm font-semibold text-slate-100">{entry.filename}</div>
                                            <div className="mt-1 text-xs text-slate-500">GT {entry.counts?.ground_truth ?? 0} · Pred {entry.counts?.predictions ?? 0}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </Section>
        </div>
    );
}
