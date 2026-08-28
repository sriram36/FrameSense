const SEVERITY_COLOR = {
  low: { bg: "#EAF3DE", text: "#27500A" },
  medium: { bg: "#FAEEDA", text: "#633806" },
  high: { bg: "#FCEBEB", text: "#791F1F" },
};

export default function IssueBadge({ issue }) {
  const colors = SEVERITY_COLOR[issue.severity] || { bg: "#F1EFE8", text: "#444441" };
  const label = issue.type.replace(/_/g, " ");

  return (
    <div className="issue-badge" style={{ background: colors.bg, color: colors.text }}>
      <span className="issue-type">{label}</span>
      <span className="issue-severity">{issue.severity}</span>
      <span className="issue-confidence">{Math.round(issue.confidence * 100)}% conf.</span>
    </div>
  );
}
