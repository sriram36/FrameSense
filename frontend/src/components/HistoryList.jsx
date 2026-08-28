import { API } from "../api.js";

const LABEL_COLOR = {
  ACCEPTABLE: "#0F6E56",
  DEGRADED: "#BA7517",
  DEFECTIVE: "#A32D2D",
};

export default function HistoryList({ items, onSelect, loading, error }) {
  if (loading) return <p className="muted">Loading history...</p>;
  if (error) return <p className="error-text">Couldn't load history: {error}</p>;
  if (items.length === 0) return <p className="muted">No analyses yet. Upload an image to get started.</p>;

  return (
    <div className="history-list">
      {items.map((item) => (
        <button key={item.id} className="history-row" onClick={() => onSelect(item.id)}>
          <div className="history-thumb-wrapper">
            <img src={API + item.image_url} alt="" className="history-thumb" />
          </div>
          <span className="history-filename">{item.filename}</span>
          <span className="history-score" style={{ color: LABEL_COLOR[item.quality_label] }}>
            {item.quality_score}
          </span>
          <span className="history-label">{item.quality_label}</span>
          <span className="history-date">{new Date(item.created_at).toLocaleString()}</span>
        </button>
      ))}
    </div>
  );
}
