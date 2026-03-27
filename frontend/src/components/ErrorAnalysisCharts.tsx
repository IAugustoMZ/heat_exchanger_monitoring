import { useMemo } from 'react';
import {
  ComposedChart, LineChart, BarChart,
  Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend, Cell,
} from 'recharts';
import type { DataPoint } from '../types';

interface Props {
  dataPoints: DataPoint[];
}

const NUM_BINS = 20;

export default function ErrorAnalysisCharts({ dataPoints }: Props) {
  // Points where model has produced a prediction
  const withPred = useMemo(
    () => dataPoints.filter((p) => p.dP_error_predicted !== null),
    [dataPoints],
  );

  // Time-series data: raw error + pct error per point (downsampled)
  const tsData = useMemo(() => {
    if (withPred.length === 0) return [];
    const maxPts = 500;
    const step = withPred.length <= maxPts ? 1 : Math.ceil(withPred.length / maxPts);
    return withPred
      .filter((_, i) => i % step === 0 || i === withPred.length - 1)
      .map((p) => {
        const rawErr = p.dP_error_actual - p.dP_error_predicted!;
        // % relative to the actual ΔP reading (always ~600–900 Pa) — gives operator-meaningful values
        const pctErr = p.delta_P_Pa > 10
          ? (rawErr / p.delta_P_Pa) * 100
          : null;
        return { t_h: p.t_h, rawErr, pctErr };
      });
  }, [withPred]);

  // Histogram buckets of raw error
  const histData = useMemo(() => {
    if (withPred.length < 2) return [];
    const errors = withPred.map((p) => p.dP_error_actual - p.dP_error_predicted!);
    const min = Math.min(...errors);
    const max = Math.max(...errors);
    if (max === min) return [];
    const binWidth = (max - min) / NUM_BINS;
    const counts = Array.from({ length: NUM_BINS }, () => 0);
    errors.forEach((e) => {
      const idx = Math.min(Math.floor((e - min) / binWidth), NUM_BINS - 1);
      counts[idx]++;
    });
    return counts.map((count, i) => ({
      center: min + (i + 0.5) * binWidth,
      label: (min + (i + 0.5) * binWidth).toFixed(1),
      count,
    }));
  }, [withPred]);

  if (dataPoints.length === 0) return null;

  const hasData = withPred.length >= 5;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* ── Time-series: raw error ── */}
      <div className="card">
        <div className="card-header">Model Residual — Raw Error (Pa)</div>
        {!hasData ? (
          <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '0.85rem', padding: '2rem 0' }}>
            Accumulating predictions…
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={tsData} margin={{ top: 5, right: 24, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="t_h"
                label={{ value: 'Time [h]', position: 'insideBottomRight', offset: -5, fontSize: 10 }}
                fontSize={10}
                tickFormatter={(v: number) => v.toFixed(1)}
              />
              <YAxis
                label={{ value: 'Error [Pa]', angle: -90, position: 'insideLeft', fontSize: 10 }}
                fontSize={10}
                domain={['auto', 'auto']}
              />
              <Tooltip
                contentStyle={{ fontSize: '0.78rem', borderRadius: '8px' }}
                formatter={(v: number) => [`${v.toFixed(2)} Pa`, 'Raw Error']}
                labelFormatter={(t: number) => `t = ${t.toFixed(2)} h`}
              />
              <Legend verticalAlign="top" height={24} />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 3" />
              <Line
                type="monotone"
                dataKey="rawErr"
                stroke="#8b5cf6"
                strokeWidth={1.8}
                dot={false}
                name="Residual (Pa)"
                connectNulls={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Time-series: % error ── */}
      <div className="card">
        <div className="card-header">Model Residual — Relative Error (% of actual ΔP)</div>
        {!hasData ? (
          <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '0.85rem', padding: '2rem 0' }}>
            Accumulating predictions…
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={tsData} margin={{ top: 5, right: 24, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="t_h"
                label={{ value: 'Time [h]', position: 'insideBottomRight', offset: -5, fontSize: 10 }}
                fontSize={10}
                tickFormatter={(v: number) => v.toFixed(1)}
              />
              <YAxis
                label={{ value: 'Error [%]', angle: -90, position: 'insideLeft', fontSize: 10 }}
                fontSize={10}
                domain={['auto', 'auto']}
                tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{ fontSize: '0.78rem', borderRadius: '8px' }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(v: any) =>
                  v === null || v === undefined
                    ? ['N/A', 'Relative Error']
                    : [`${(v as number).toFixed(1)}%`, 'Relative Error']
                }
                labelFormatter={(t: number) => `t = ${t.toFixed(2)} h`}
              />
              <Legend verticalAlign="top" height={24} />
              <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 3" />
              <Line
                type="monotone"
                dataKey="pctErr"
                stroke="#f59e0b"
                strokeWidth={1.8}
                dot={false}
                name="Relative Error (%)"
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Error distribution histogram ── */}
      <div className="card">
        <div className="card-header">
          Error Distribution — Raw Residual (Pa)
          {withPred.length > 0 && (
            <span style={{ fontSize: '0.78rem', fontWeight: 400, marginLeft: '0.75rem', color: 'var(--color-text-muted)' }}>
              n = {withPred.length}
            </span>
          )}
        </div>
        {histData.length < 3 ? (
          <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '0.85rem', padding: '2rem 0' }}>
            Accumulating predictions…
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={histData} margin={{ top: 5, right: 24, bottom: 24, left: 10 }} barCategoryGap="2%">
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
              <XAxis
                dataKey="label"
                label={{ value: 'Residual [Pa]', position: 'insideBottom', offset: -14, fontSize: 10 }}
                fontSize={9}
                interval={Math.max(0, Math.floor(histData.length / 8) - 1)}
                tickFormatter={(v: string) => parseFloat(v).toFixed(0)}
              />
              <YAxis
                label={{ value: 'Count', angle: -90, position: 'insideLeft', fontSize: 10 }}
                fontSize={10}
                allowDecimals={false}
              />
              <Tooltip
                contentStyle={{ fontSize: '0.78rem', borderRadius: '8px' }}
                formatter={(count: number | string, _: string, item?: { payload?: { center?: number; count?: number } }) => [
                  count,
                  `bin ≈ ${(item?.payload?.center ?? 0).toFixed(1)} Pa`,
                ]}
                cursor={{ fill: 'rgba(139,92,246,0.08)' }}
              />
              {/* Zero-error reference line */}
              <ReferenceLine x="0.0" stroke="#94a3b8" strokeDasharray="4 3" />
              <Bar dataKey="count" name="Count" radius={[3, 3, 0, 0]}>
                {histData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.center < 0 ? '#f97316' : '#8b5cf6'}
                    fillOpacity={0.78}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
