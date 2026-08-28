const LABEL_COLOR = {
  ACCEPTABLE: "#0F6E56",
  DEGRADED: "#BA7517",
  DEFECTIVE: "#A32D2D",
};

export default function ScoreGauge({ score, label }) {
  const color = LABEL_COLOR[label] || "#5F5E5A";
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - score / 100);

  return (
    <div className="score-gauge">
      <svg viewBox="0 0 140 140" width="140" height="140">
        <circle cx="70" cy="70" r={radius} fill="none" stroke="#E4E2D8" strokeWidth="10" />
        <circle
          cx="70"
          cy="70"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 70 70)"
        />
        <text x="70" y="66" textAnchor="middle" className="gauge-score">{score}</text>
        <text x="70" y="86" textAnchor="middle" className="gauge-unit">/ 100</text>
      </svg>
      <div className="gauge-label" style={{ color }}>{label}</div>
    </div>
  );
}
