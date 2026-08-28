import { useCallback, useRef, useState } from "react";

export default function UploadPanel({ onAnalyze, loading }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
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

  return (
    <div className="upload-panel">
      <div
        className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") inputRef.current?.click(); }}
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
