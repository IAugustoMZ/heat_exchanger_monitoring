import type { DataPoint } from '../types';

interface Props {
  currentPoint: DataPoint | null;
}

export default function RuntimeForecast({ currentPoint }: Props) {
  const remaining   = currentPoint?.remaining_runtime_h ?? null;
  const rulLower    = currentPoint?.rul_lower ?? null;
  const rulUpper    = currentPoint?.rul_upper ?? null;
  const slope       = currentPoint?.dp_error_slope_pa_per_h ?? null;
  const defrostDate = currentPoint?.defrost_date ?? null;
  const threshold   = currentPoint?.alarm_threshold_pa ?? 943.3;
  const modelAvail  = currentPoint?.model_available ?? false;

  const urgencyColor =
    remaining === null ? 'var(--color-text-muted)'
    : remaining < 2   ? 'var(--color-danger)'
    : remaining < 6   ? 'var(--color-warning)'
    : 'var(--color-success)';

  const defrostLocal = defrostDate
    ? new Date(defrostDate + 'Z').toLocaleString(undefined, {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      })
    : null;

  return (
    <div className="card">
      <div className="card-header">Forecast Summary</div>

      {!modelAvail || (remaining === null && slope === null) ? (
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem', textAlign: 'center', padding: '1rem 0' }}>
          {!modelAvail ? 'Awaiting model predictions…' : 'Trend stable — no threshold crossing projected'}
        </p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', padding: '0.5rem 0' }}>

          {/* Remaining Runtime */}
          <div className="metric">
            <div className="metric-value" style={{ color: urgencyColor }}>
              {remaining !== null
                ? (remaining > 99 ? '>99' : remaining.toFixed(1))
                : '∞'}
              <span className="metric-unit"> h</span>
            </div>
            <div className="metric-label">Remaining Runtime</div>
            {rulLower !== null && rulUpper !== null && remaining !== null && (
              <div style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', marginTop: '0.2rem' }}>
                ±1σ: {rulLower.toFixed(1)} – {rulUpper.toFixed(1)} h
              </div>
            )}
          </div>

          {/* Projected Defrost Date */}
          <div className="metric">
            <div className="metric-value" style={{ fontSize: defrostLocal ? '0.9rem' : '1.75rem', color: urgencyColor }}>
              {defrostLocal ?? '—'}
            </div>
            <div className="metric-label">Projected Defrost Date</div>
          </div>

          {/* dP Error Slope */}
          <div className="metric">
            <div className="metric-value" style={{ fontSize: '1.1rem' }}>
              {slope !== null ? slope.toFixed(2) : '—'}
            </div>
            <div className="metric-label">ΔP Error Slope (Pa/h)</div>
          </div>

          {/* Alarm threshold reminder */}
          <div className="metric">
            <div className="metric-value" style={{ fontSize: '1.1rem', color: 'var(--color-danger)' }}>
              {threshold.toFixed(0)}
              <span className="metric-unit"> Pa</span>
            </div>
            <div className="metric-label">Alarm Threshold</div>
          </div>
        </div>
      )}

      {/* Urgency banner */}
      {remaining !== null && remaining < 4 && (
        <div style={{
          marginTop: '0.75rem', padding: '0.6rem 0.75rem',
          background: remaining < 2 ? 'var(--color-danger-light)' : 'var(--color-warning-light)',
          borderRadius: 'var(--radius)',
          fontSize: '0.8rem',
          color: remaining < 2 ? '#991b1b' : '#92400e',
          fontWeight: 600,
        }}>
          ⚠ Schedule defrost within the next {remaining.toFixed(0)} hour{remaining < 1.5 ? '' : 's'}.
        </div>
      )}
    </div>
  );
}
