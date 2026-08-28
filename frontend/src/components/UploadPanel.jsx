import { useCallback, useEffect, useRef, useState } from "react";

const PIPELINE_LOGS = [
  "Uploading image data...",
  "Extracting image features (sharpness, contrast, noise)...",
  "Running RandomForest degradation classifier...",
  "Computing PCA reconstruction error for anomalies...",
  "Generating defect localization heatmap...",
  "Finalizing quality score...",
];

export default function UploadPanel({ onAnalyze, loading }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [logIndex, setLogIndex] = useState(0);
  const inputRef = useRef(null);

  const selectFile = useCallback((selected) => {
    if (!selected) return;
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files?.[0];
    selectFile(dropped);
  };

  useEffect(() => {
    if (!loading) {
      setLogIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setLogIndex((prev) => Math.min(prev + 1, PIPELINE_LOGS.length - 1));
    }, 600); // Progress through logs every 600ms
    return () => clearInterval(interval);
  }, [loading]);

  return (
    <div className="upload-panel">
      <div
        className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => !loading && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (!loading && (e.key === "Enter" || e.key === " ")) inputRef.current?.click(); }}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected upload preview" className="dropzone-preview" />
        ) : (
          <>
            <p className="dropzone-title">Drop an image here, or click to choose one</p>
            <p className="dropzone-hint">JPG or PNG</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => selectFile(e.target.files?.[0])}
        />
      </div>

      {loading && (
        <div className="analysis-logs">
          <div className="log-spinner"></div>
          <div className="log-text">
            {PIPELINE_LOGS.slice(0, logIndex + 1).map((log, i) => (
              <div key={i} className={`log-line ${i === logIndex ? "log-active" : "log-muted"}`}>
                <span className="log-prefix">&gt;</span> {log}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="upload-actions">
        <button
          className="btn btn-primary"
          disabled={!file || loading}
          onClick={() => onAnalyze(file)}
        >
          {loading ? "Analyzing..." : "Analyze image"}
        </button>
        {file && !loading && (
          <button className="btn btn-ghost" onClick={() => { setFile(null); setPreviewUrl(null); }}>
            Clear
          </button>
        )}
      </div>
    </div>
  );
}
