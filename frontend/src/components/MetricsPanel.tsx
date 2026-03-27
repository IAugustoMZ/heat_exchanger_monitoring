import type { DataPoint, ScenarioMeta } from '../types';

interface Props {
  currentPoint: DataPoint | null;
  scenarioMeta: ScenarioMeta | null;
}

export default function MetricsPanel({ currentPoint, scenarioMeta }: Props) {
  return (
    <div className="card">
      <div className="card-header">Live Process Metrics</div>

      {!currentPoint ? (
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', textAlign: 'center', padding: '1rem' }}>
          No data yet. Select a scenario and click Begin.
        </p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div className="metric">
            <div className="metric-value">
              {currentPoint.t_h.toFixed(2)}
              <span className="metric-unit"> h</span>
            </div>
            <div className="metric-label">Elapsed Time</div>
          </div>

          <div className="metric">
            <div className={`metric-value ${currentPoint.alarm_predicted ? 'danger' : ''}`}>
              {currentPoint.delta_P_Pa.toFixed(0)}
              <span className="metric-unit"> Pa</span>
            </div>
            <div className="metric-label">ΔP Actual</div>
          </div>

          <div className="metric">
            <div className="metric-value">
              {currentPoint.dP_error_predicted !== null
                ? currentPoint.dP_error_predicted.toFixed(1)
                : '--'}
              <span className="metric-unit"> Pa</span>
            </div>
            <div className="metric-label">dP Error (ML)</div>
          </div>

          <div className="metric">
            <div className="metric-value">
              {currentPoint.U_mean_W_m2K.toFixed(1)}
              <span className="metric-unit"> W/m²K</span>
            </div>
            <div className="metric-label">U Overall</div>
          </div>

          <div className="metric">
            <div className="metric-value" style={{ color: 'var(--color-accent)' }}>
              {(currentPoint.delta_f_max_m * 1000).toFixed(2)}
              <span className="metric-unit"> mm</span>
            </div>
            <div className="metric-label">Frost Thickness</div>
          </div>

          <div className="metric">
            <div className="metric-value">
              {currentPoint.T_h_in_K.toFixed(1)}
              <span className="metric-unit"> K</span>
            </div>
            <div className="metric-label">Gas Inlet T</div>
          </div>
        </div>
      )}

      {scenarioMeta && (
        <div style={{
          marginTop: '1rem',
          padding: '0.75rem',
          background: 'var(--color-surface-alt)',
          borderRadius: 'var(--radius)',
          fontSize: '0.8rem',
          color: 'var(--color-text-secondary)',
        }}>
          <strong>{scenarioMeta.label}</strong> — {scenarioMeta.avg_duration_h}h simulation,
          ~{scenarioMeta.avg_points} data points
        </div>
      )}
    </div>
  );
}
