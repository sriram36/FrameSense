import IssueBadge from "./IssueBadge.jsx";
import ScoreGauge from "./ScoreGauge.jsx";

const FEATURE_LABELS = {
  sharpness: "Sharpness",
  brightness: "Brightness",
  contrast: "Contrast",
  noise_estimate: "Noise estimate",
  saturation: "Saturation",
  blockiness: "Blockiness",
  edge_density: "Edge density",
};

export default function ResultCard({ result, imageUrl }) {
  const features = Object.entries(result.features || {});

  return (
    <div className="result-card">
      <div className="result-top">
        {imageUrl && (
          <img src={imageUrl} alt={result.filename} className="result-image" />
        )}
        <ScoreGauge score={result.quality_score} label={result.quality_label} />
      </div>

      <div className="result-section">
        <h3>Detected issues</h3>
        {result.issues.length === 0 ? (
          <p className="muted">No issues detected.</p>
        ) : (
          <div className="issue-list">
            {result.issues.map((issue, i) => (
              <IssueBadge key={i} issue={issue} />
            ))}
          </div>
        )}
      </div>

      <div className="result-section">
        <h3>Image statistics</h3>
        <div className="feature-grid">
          {features.map(([key, value]) => (
            <div key={key} className="feature-item">
              <span className="feature-label">{FEATURE_LABELS[key] || key}</span>
              <span className="feature-value">{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
