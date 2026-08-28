import { useEffect, useState } from "react";
import { API, analyzeImage, getResult, listResults } from "./api.js";
import HistoryList from "./components/HistoryList.jsx";
import ResultCard from "./components/ResultCard.jsx";
import UploadPanel from "./components/UploadPanel.jsx";

export default function App() {
  const [tab, setTab] = useState("upload");

  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [historyItems, setHistoryItems] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState(null);
  const [historyStale, setHistoryStale] = useState(true);

  const runAnalysis = async (file) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));
    setHeatmapUrl(null);
    try {
      const data = await analyzeImage(file);
      setResult(data);
      if (data.heatmap_url) setHeatmapUrl(API + data.heatmap_url);
      setHistoryStale(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const data = await listResults(50, 0);
      setHistoryItems(data.items);
      setHistoryStale(false);
    } catch (err) {
      setHistoryError(err.message);
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    if (tab === "history" && historyStale) loadHistory();
  }, [tab, historyStale]);

  const openHistoryItem = async (id) => {
    setError(null);
    setLoading(true);
    try {
      const data = await getResult(id);
      setResult(data);
      setPreviewUrl(data.image_url ? API + data.image_url : null);
      setHeatmapUrl(data.heatmap_url ? API + data.heatmap_url : null);
      setTab("upload");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Image quality &amp; defect detector</h1>
        <nav className="tabs">
          <button className={tab === "upload" ? "tab tab-active" : "tab"} onClick={() => setTab("upload")}>
            Analyze
          </button>
          <button className={tab === "history" ? "tab tab-active" : "tab"} onClick={() => setTab("history")}>
            History
          </button>
        </nav>
      </header>

      <main className="app-main">
        {tab === "upload" && (
          <>
            <UploadPanel onAnalyze={runAnalysis} loading={loading} />
            {error && <p className="error-text">Analysis failed: {error}</p>}
            {result && <ResultCard result={result} imageUrl={previewUrl} heatmapUrl={heatmapUrl} />}
          </>
        )}

        {tab === "history" && (
          <HistoryList
            items={historyItems}
            loading={historyLoading}
            error={historyError}
            onSelect={openHistoryItem}
          />
        )}
      </main>
    </div>
  );
}
