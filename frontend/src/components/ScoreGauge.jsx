const LABEL_COLOR = {
  ACCEPTABLE: "#10B981", // Emerald
  DEGRADED: "#F59E0B",   // Amber
  DEFECTIVE: "#EF4444",  // Red
};

export default function ScoreGauge({ score, label }) {
  const color = LABEL_COLOR[label] || "#94A3B8";
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - score / 100);

  return (
    <div className="score-gauge">
      <svg 
        viewBox="0 0 140 140" 
        width="140" 
        height="140" 
        style={{ filter: `drop-shadow(0 0 16px ${color}80) drop-shadow(0 0 32px ${color}40)` }}
        className={label === "DEFECTIVE" ? "gauge-pulse" : ""}
      >
        <circle cx="70" cy="70" r={radius} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="10" />
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
          style={{ transition: "stroke-dashoffset 1s ease-in-out" }}
        />
        <text x="70" y="66" textAnchor="middle" className="gauge-score">{score}</text>
        <text x="70" y="86" textAnchor="middle" className="gauge-unit">/ 100</text>
      </svg>
      <div className="gauge-label" style={{ color }}>{label}</div>
    </div>
  );
}
