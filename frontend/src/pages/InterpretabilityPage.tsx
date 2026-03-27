import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts';
import { fetchInterpretability, fetchAllModels, getFigureUrl } from '../api';
import type { InterpretabilityData, ModelInfo } from '../types';

export default function InterpretabilityPage() {
  const [data, setData] = useState<InterpretabilityData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchInterpretability(), fetchAllModels()])
      .then(([interpData, modelsData]) => {
        setData(interpData);
        setModels(modelsData.models || []);
      })
      .catch((err) => setError(`Failed to load interpretability data: ${err.message}`))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-text-muted)' }}>
        Loading interpretability data from MLflow...
      </div>
    );
  }

  if (error) {
    return <div className="alert-banner danger">⚠ {error}</div>;
  }

  // Prepare feature importance chart data
  const importanceData = data?.feature_importance
    ? Object.entries(data.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .map(([name, value]) => ({
          name,
          label: getOperatorLabel(name),
          value: Number(value),
          isObservable: isObservableFeature(name),
        }))
    : [];

  const forecast = data?.forecast || {};

  return (
    <div>
      <h1 style={{
        fontSize: '1.5rem', fontWeight: 700,
        color: 'var(--color-primary-700)', marginBottom: '0.5rem'
      }}>
        Model Interpretability
      </h1>
      <p style={{ color: 'var(--color-text-secondary)', marginBottom: '2rem', fontSize: '0.9rem' }}>
        Understanding how the Lasso model (R²=0.970) makes frost predictions.
        Answering the three critical business questions.
      </p>

      {/* ─── Question A: Feature Importance ─── */}
      <div className="interp-section">
        <h2>A. What physical processes drive ΔP and U degradation?</h2>
        <p>
          Signal importance shows which process variables the Lasso model relies on most.
          <span style={{ color: '#f97316' }}>■ Orange</span> = computed process deviations (error signals),{' '}
          <span style={{ color: '#2563a0' }}>■ Blue</span> = direct DCS/SCADA readings.
        </p>

        {importanceData.length > 0 ? (
          <div className="card">
            <ResponsiveContainer width="100%" height={Math.max(200, importanceData.length * 48)}>
              <BarChart data={importanceData} layout="vertical" margin={{ left: 210, right: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" fontSize={11} />
                <YAxis type="category" dataKey="label" fontSize={11} width={200} />
                <Tooltip
                  contentStyle={{ fontSize: '0.8rem', borderRadius: '8px' }}
                  formatter={(value: number, _name: string, props: { payload?: { name?: string } }) => [
                    value.toFixed(4),
                    props.payload?.name ?? 'Importance',
                  ]}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {importanceData.map((entry, index) => (
                    <Cell key={index} fill={entry.isObservable ? '#2563a0' : '#f97316'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--color-text-muted)' }}>
            No feature importance data available for this model.
          </div>
        )}

        {/* Physical root cause chain */}
        <div className="card" style={{ marginTop: '1rem' }}>
          <div style={{ fontWeight: 700, fontSize: '0.85rem', color: 'var(--color-primary-700)', marginBottom: '0.75rem' }}>
            Physical Root Cause Chain
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', fontSize: '0.83rem' }}>
            <div>
              <div style={{ fontWeight: 600, color: '#ea580c', marginBottom: '0.4rem' }}>
                What drives ΔP increase?
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', color: 'var(--color-text-secondary)' }}>
                <div>Heavy C6+ components in feed gas</div>
                <div style={{ paddingLeft: '1rem' }}>↓ Contact with sub-zero LNG side</div>
                <div style={{ paddingLeft: '1rem' }}>↓ Crystallise on tube inner walls</div>
                <div style={{ paddingLeft: '1rem' }}>↓ Frost layer reduces hydraulic diameter D<sub>h</sub></div>
                <div style={{ paddingLeft: '1rem', color: '#ef4444', fontWeight: 600 }}>
                  ↑ ΔP ∝ D<sub>h</sub><sup>−5</sup> → exponential rise → ALARM at 943 Pa
                </div>
              </div>
            </div>
            <div>
              <div style={{ fontWeight: 600, color: '#0891b2', marginBottom: '0.4rem' }}>
                What drives U degradation?
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem', color: 'var(--color-text-secondary)' }}>
                <div>Frost layer forms on tube surface</div>
                <div style={{ paddingLeft: '1rem' }}>↓ R<sub>frost</sub> = δ<sub>f</sub> / k<sub>frost</sub> adds thermal resistance</div>
                <div style={{ paddingLeft: '1rem' }}>↓ 1/U = 1/h<sub>h</sub> + R<sub>frost</sub> + 1/h<sub>c</sub></div>
                <div style={{ paddingLeft: '1rem', color: '#f97316', fontWeight: 600 }}>
                  ↓ U drops → feed gas exits warmer → less LNG vaporised
                </div>
              </div>
            </div>
          </div>
          <div style={{
            marginTop: '1rem', padding: '0.75rem',
            background: 'var(--color-surface-alt)', borderRadius: '8px',
            fontSize: '0.8rem', color: 'var(--color-text-secondary)',
          }}>
            <strong>What operators should watch:</strong>
            <ul style={{ margin: '0.4rem 0 0 1.2rem', lineHeight: 1.7 }}>
              <li><strong>T_h,out declining</strong> — hot gas leaving warmer than expected signals reduced heat exchange</li>
              <li><strong>T_c,out rising</strong> — LNG exits at higher temperature, indicating thermal resistance build-up</li>
              <li><strong>ΔP trending up</strong> — primary KPI; the model tracks the deviation from the clean-tube ideal (628.8 Pa baseline)</li>
              <li><strong>ΔP Anomaly (prev. reading)</strong> — the dominant model input; frost is highly persistent (lag-1 ACF ≈ 0.999)</li>
            </ul>
          </div>
        </div>
      </div>

      {/* ─── Question B: Observable Proxy Features ─── */}
      <div className="interp-section">
        <h2>B. Can existing sensors proxy for heavy component loading?</h2>
        <p>
          Our instruments cannot directly measure heavy hydrocarbon components (C6+) that cause
          tube fouling. However, the model identifies which observable signals act as proxies:
        </p>

        <div className="card">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
            {importanceData.filter(f => f.isObservable).length > 0 ? (
              importanceData
                .filter(f => f.isObservable)
                .map((feature) => (
                  <div key={feature.name} style={{
                    padding: '1rem',
                    background: 'var(--color-primary-50)',
                    borderRadius: 'var(--radius)',
                    borderLeft: '4px solid var(--color-primary-400)',
                  }}>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--color-primary-700)' }}>
                      {feature.name}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.25rem' }}>
                      Importance: {feature.value.toFixed(4)}
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                      {getFeatureDescription(feature.name)}
                    </div>
                  </div>
                ))
            ) : (
              <p style={{ color: 'var(--color-text-muted)', padding: '1rem' }}>
                The Lasso model selected only derived features. This indicates error signals
                (computed from first-principles) are more predictive than raw DCS readings alone.
              </p>
            )}
          </div>
          <div style={{
            marginTop: '1rem', padding: '0.75rem',
            background: 'var(--color-surface-alt)', borderRadius: 'var(--radius)',
            fontSize: '0.8rem', color: 'var(--color-text-secondary)',
          }}>
            <strong>Insight:</strong> The dominant feature <code>dP_error(t-1)</code> is
            the previous timestep's deviation from clean-tube ΔP. This autoregressive pattern
            means the frost accumulation is highly persistent (lag-1 ACF &gt; 0.999). The error
            signal <code>U_error</code> captures heat transfer degradation — an indirect proxy
            for heavy component deposition on tube surfaces.
          </div>
        </div>
      </div>

      {/* ─── Question C: Runtime Forecast ─── */}
      <div className="interp-section">
        <h2>C. What is the expected runtime / defrost date?</h2>
        <p>
          The live forecast is computed from streaming sensor data on the{' '}
          <strong>Monitor</strong> page — it updates in real-time as each data point arrives.
          The methodology below describes how the projection is made.
        </p>

        <div className="grid-2">
          {/* Static training-time baseline from MLflow artifact */}
          <div className="card">
            <div className="card-header">Training-Time Baseline (from MLflow)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', padding: '0.5rem 0' }}>
              <div className="metric">
                <div className="metric-value">
                  {forecast.t_defrost_h ? Number(forecast.t_defrost_h).toFixed(1) : '—'}
                  <span className="metric-unit"> h</span>
                </div>
                <div className="metric-label">Defrost Time (test set)</div>
              </div>
              <div className="metric">
                <div className="metric-value">
                  {forecast.remaining_runtime_h ? Number(forecast.remaining_runtime_h).toFixed(1) : '∞'}
                  <span className="metric-unit"> h</span>
                </div>
                <div className="metric-label">Remaining Runtime (test set)</div>
              </div>
              <div className="metric">
                <div className="metric-value" style={{ fontSize: '1rem' }}>
                  {forecast.dP_error_slope_pa_per_h
                    ? Number(forecast.dP_error_slope_pa_per_h).toFixed(2)
                    : '—'}
                </div>
                <div className="metric-label">ΔP Error Slope (Pa/h)</div>
              </div>
              <div className="metric">
                <div className="metric-value" style={{ fontSize: '0.9rem' }}>
                  {forecast.defrost_date
                    ? new Date(forecast.defrost_date).toLocaleDateString()
                    : '—'}
                </div>
                <div className="metric-label">Projected Defrost Date</div>
              </div>
            </div>
            <div style={{
              marginTop: '0.75rem', padding: '0.6rem 0.75rem',
              background: 'var(--color-surface-alt)', borderRadius: 'var(--radius)',
              fontSize: '0.78rem', color: 'var(--color-text-muted)',
            }}>
              This is computed once from the full test dataset. For live projections updated each second, go to the <strong>Monitor</strong> page.
            </div>
          </div>

          <div className="card">
            <div className="card-header">Methodology</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
              <ol style={{ paddingLeft: '1.25rem' }}>
                <li>Predict dP_error on test history using the Lasso pipeline</li>
                <li>Fit a linear trend to the last 30% of predictions</li>
                <li>Extrapolate: find <em>t</em> where dP_ideal + dP_error(t) ≥ threshold</li>
                <li>Report remaining runtime and calendar defrost date</li>
              </ol>
              <p style={{ marginTop: '0.75rem' }}>
                <strong>Alarm threshold:</strong> 943.3 Pa (150% of clean-tube baseline 628.8 Pa)
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* ─── MLflow Figures ─── */}
      {data?.figures && data.figures.length > 0 && (
        <div className="interp-section">
          <h2>Model Artifacts from MLflow</h2>
          <p>Training-time plots generated during model evaluation.</p>
          <div className="grid-2">
            {data.figures.map((fig) => (
              <div key={fig.filename} className="figure-container">
                <img
                  src={getFigureUrl(fig.filename)}
                  alt={fig.name}
                  loading="lazy"
                />
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
                  {fig.name.replace(/_/g, ' ')}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ─── Model Comparison ─── */}
      {models.length > 0 && (
        <div className="interp-section">
          <h2>All Trained Models</h2>
          <p>Comparison of all 7 models trained via Optuna hyperparameter optimization.</p>
          <div className="card" style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--color-border)' }}>
                  <th style={thStyle}>Model</th>
                  <th style={thStyle}>Status</th>
                  <th style={thStyle}>Top Feature</th>
                  <th style={thStyle}>Remaining Runtime</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const topFeature = Object.entries(m.importance || {})
                    .sort(([, a], [, b]) => Number(b) - Number(a))[0];
                  return (
                    <tr key={m.run_id} style={{
                      borderBottom: '1px solid var(--color-border)',
                      background: m.is_best ? 'var(--color-primary-50)' : undefined,
                    }}>
                      <td style={tdStyle}>
                        {m.model_name}
                        {m.is_best && (
                          <span style={{
                            marginLeft: '0.5rem', fontSize: '0.7rem', fontWeight: 600,
                            background: 'var(--color-primary-400)', color: 'white',
                            padding: '0.1rem 0.4rem', borderRadius: '4px',
                          }}>
                            BEST
                          </span>
                        )}
                      </td>
                      <td style={tdStyle}>
                        <span className={`status-dot ${m.is_best ? 'online' : ''}`}
                          style={{ marginRight: '0.5rem' }} />
                        {m.is_best ? 'Production' : 'Trained'}
                      </td>
                      <td style={tdStyle}>
                        {topFeature ? `${topFeature[0]} (${Number(topFeature[1]).toFixed(3)})` : '—'}
                      </td>
                      <td style={tdStyle}>
                        {m.forecast?.remaining_runtime_h
                          ? `${Number(m.forecast.remaining_runtime_h).toFixed(1)} h`
                          : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Helpers ───

const OBSERVABLE = new Set([
  'T_h_in_K', 'T_h_out_K', 'T_c_in_K', 'T_c_out_K', 'delta_P_Pa',
]);

function isObservableFeature(name: string): boolean {
  return [...OBSERVABLE].some((obs) => name.includes(obs));
}

function getOperatorLabel(raw: string): string {
  const MAP: Record<string, string> = {
    'dP_error(t-1)':     'ΔP Anomaly — previous reading',
    'dP_error(t-2)':     'ΔP Anomaly — 2 readings ago',
    'U_error':           'Heat Transfer Degradation (U)',
    'delta_P_Pa(t-2)':   'Tube Pressure Drop — 2 readings ago',
    'delta_P_Pa':        'Tube Pressure Drop',
    'T_h_in_K':          'Hot Gas Inlet Temp (T_h,in)',
    'T_h_out_K':         'Hot Gas Outlet Temp (T_h,out)',
    'T_c_in_K':          'LNG Inlet Temp (T_c,in)',
    'T_c_out_K':         'LNG Outlet Temp (T_c,out)',
  };
  // Exact match first, then prefix match
  if (MAP[raw]) return MAP[raw];
  for (const [key, label] of Object.entries(MAP)) {
    if (raw.startsWith(key)) return label;
  }
  return raw;
}

function getFeatureDescription(name: string): string {
  const descriptions: Record<string, string> = {
    'T_h_in_K': 'Feed gas inlet temperature — reflects upstream conditions',
    'T_h_out_K': 'Feed gas outlet temperature — responds to frost insulation',
    'T_c_in_K': 'LNG inlet temperature — driving force for heat transfer',
    'T_c_out_K': 'LNG outlet temperature — strongest proxy for thermal degradation',
    'delta_P_Pa': 'Tube-side pressure drop — primary KPI (∝ D_h^-5)',
  };
  for (const [key, desc] of Object.entries(descriptions)) {
    if (name.includes(key)) return desc;
  }
  return 'Derived feature from observable sensors';
}

const thStyle: React.CSSProperties = {
  textAlign: 'left', padding: '0.75rem 0.5rem',
  fontWeight: 600, color: 'var(--color-text-secondary)',
  fontSize: '0.8rem', textTransform: 'uppercase',
  letterSpacing: '0.05em',
};

const tdStyle: React.CSSProperties = {
  padding: '0.6rem 0.5rem',
};
