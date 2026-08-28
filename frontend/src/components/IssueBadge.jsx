const SEVERITY_COLOR = {
  low: { bg: "rgba(16, 185, 129, 0.15)", border: "rgba(16, 185, 129, 0.4)", text: "#34D399" },
  medium: { bg: "rgba(245, 158, 11, 0.15)", border: "rgba(245, 158, 11, 0.4)", text: "#FBBF24" },
  high: { bg: "rgba(239, 68, 68, 0.15)", border: "rgba(239, 68, 68, 0.4)", text: "#F87171" },
};

export default function IssueBadge({ issue }) {
  const colors = SEVERITY_COLOR[issue.severity] || { bg: "rgba(255,255,255,0.05)", border: "rgba(255,255,255,0.1)", text: "#CBD5E1" };
  const label = issue.type.replace(/_/g, " ");

  return (
    <div 
      className="issue-badge" 
      style={{ 
        background: colors.bg, 
        borderColor: colors.border,
        color: colors.text,
        boxShadow: `0 0 12px ${colors.bg}`
      }}
    >
      <span className="issue-type">{label}</span>
      <span className="issue-severity">{issue.severity}</span>
      <span className="issue-confidence">{Math.round(issue.confidence * 100)}% conf.</span>
    </div>
  );
}
