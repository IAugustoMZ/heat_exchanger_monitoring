import type { DataPoint } from '../types';

interface Props {
  currentPoint: DataPoint | null;
}

export default function AlertPanel({ currentPoint }: Props) {
  if (!currentPoint) return null;

  const { alarm_predicted, alarm_actual, dP_total_predicted, alarm_threshold_pa } = currentPoint;

  if (alarm_predicted) {
    return (
      <div className="alert-banner danger">
        <span className="status-dot alarm" />
        <span>
          <strong>FREEZING ALARM — </strong>
          Predicted ΔP ({dP_total_predicted?.toFixed(0)} Pa) exceeds threshold ({alarm_threshold_pa} Pa).
          Initiate defrost procedure.
        </span>
      </div>
    );
  }

  if (dP_total_predicted && dP_total_predicted > alarm_threshold_pa * 0.85) {
    return (
      <div className="alert-banner warning">
        <span>⚡</span>
        <span>
          <strong>WARNING — </strong>
          ΔP trending toward alarm threshold ({((dP_total_predicted / alarm_threshold_pa) * 100).toFixed(0)}% of limit).
          Monitor closely.
        </span>
      </div>
    );
  }

  if (alarm_actual && !alarm_predicted) {
    return (
      <div className="alert-banner warning">
        <span>📊</span>
        <span>
          Actual freezing alarm active but model has not predicted threshold crossing yet.
        </span>
      </div>
    );
  }

  return (
    <div className="alert-banner success">
      <span className="status-dot online" />
      <span>System operating normally. No frost alarm detected.</span>
    </div>
  );
}
