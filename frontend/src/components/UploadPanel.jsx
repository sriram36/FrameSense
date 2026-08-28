import { useCallback, useEffect, useRef, useState } from "react";

const PIPELINE_LOGS = ["Uploading image data...", "Extracting image features (sharpness, contrast, noise)...", "Running RandomForest degradation classifier...", "Computing PCA reconstruction error for anomalies...", "Generating defect localization heatmap...", "Finalizing quality score..."];

export default function UploadPanel({ onAnalyze, loading }) {
  const [file, setFile] = useState(null); const [previewUrl, setPreviewUrl] = useState(null); const [dragActive, setDragActive] = useState(false); const [logIndex, setLogIndex] = useState(0); const inputRef = useRef(null);
  const selectFile = useCallback((selected) => { if (!selected || !selected.type.startsWith("image/")) return; setFile(selected); setPreviewUrl(URL.createObjectURL(selected)); }, []);
  const handleDrop = (e) => { e.preventDefault(); setDragActive(false); selectFile(e.dataTransfer.files?.[0]); };
  useEffect(() => { if (!loading) { setLogIndex(0); return; } const interval = setInterval(() => setLogIndex((prev) => Math.min(prev + 1, PIPELINE_LOGS.length - 1)), 2000); return () => clearInterval(interval); }, [loading]);
  return <section className="upload-panel" aria-label="Upload image for analysis">
    <div className="upload-panel-header"><div><p className="section-kicker">START AN ANALYSIS</p><h2>Inspect a new frame</h2></div><span className="format-chip">JPG · PNG · WEBP</span></div>
    <div className={`dropzone ${dragActive ? "dropzone-active" : ""} ${previewUrl ? "has-preview" : ""}`} onDragOver={(e) => { e.preventDefault(); setDragActive(true); }} onDragLeave={() => setDragActive(false)} onDrop={handleDrop} onClick={() => !loading && inputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => { if (!loading && (e.key === "Enter" || e.key === " ")) inputRef.current?.click(); }}>
      {previewUrl ? <><img src={previewUrl} alt="Selected upload preview" className="dropzone-preview" /><div className="preview-overlay">Click to replace image</div></> : <><div className="upload-glyph" aria-hidden="true"><span>↑</span></div><p className="dropzone-title">Drop an image to begin</p><p className="dropzone-hint">or click to browse your files</p></>}
      <input ref={inputRef} type="file" accept="image/*" hidden onChange={(e) => selectFile(e.target.files?.[0])} />
    </div>
    {file && <div className="file-summary"><span className="file-type">IMG</span><div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB · Ready for inspection</span></div></div>}
    {loading && <div className="analysis-logs"><div className="log-spinner" /><div className="log-text">{PIPELINE_LOGS.slice(0, logIndex + 1).map((log, i) => <div key={i} className={`log-line ${i === logIndex ? "log-active" : "log-muted"}`}><span className="log-prefix">›</span>{log}</div>)}</div></div>}
    <div className="upload-actions"><button className="btn btn-primary" disabled={!file || loading} onClick={() => onAnalyze(file)}>{loading ? "Processing frame..." : "Analyze image  →"}</button>{file && !loading && <button className="btn btn-ghost" onClick={() => { setFile(null); setPreviewUrl(null); }}>Clear</button>}</div>
  </section>;
}
