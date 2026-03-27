import { useMemo } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import type { DataPoint } from '../types';

interface Props {
  dataPoints: DataPoint[];
}

const WARN_THRESHOLD_H = 6;
const DANGER_THRESHOLD_H = 2;

export default function RULChart({ dataPoints }: Props) {
  const chartData = useMemo(() => {
    // Only include points where model has produced a remaining_runtime estimate
    const filtered = dataPoints.filter((p) => p.remaining_runtime_h !== null);
    if (filtered.length === 0) return [];

    const maxPoints = 400;
    const decimated =
      filtered.length <= maxPoints
        ? filtered
        : filtered.filter((_, i) => i % Math.ceil(filtered.length / maxPoints) === 0 || i === filtered.length - 1);

    // Add 2-point moving average and clamp uncertainty band to ≥0
    return decimated.map((p, i, arr) => {
      const ma2 =
        i === 0
          ? p.remaining_runtime_h
          : ((arr[i - 1].remaining_runtime_h ?? p.remaining_runtime_h!) + p.remaining_runtime_h!) / 2;
      return {
        ...p,
        ma2,
        // Recharts Area (range) needs [lower, upper] as a tuple via a custom key
        rul_band: p.rul_lower !== null && p.rul_upper !== null
          ? [Math.max(0, p.rul_lower), p.rul_upper] as [number, number]
          : null,
      };
    });
  }, [dataPoints]);

  const latestRUL = useMemo(() => {
    for (let i = dataPoints.length - 1; i >= 0; i--) {
      if (dataPoints[i].remaining_runtime_h !== null) return dataPoints[i].remaining_runtime_h!;
    }
    return null;
  }, [dataPoints]);

  const rulColor =
    latestRUL === null ? 'var(--color-text-muted)'
    : latestRUL < DANGER_THRESHOLD_H ? 'var(--color-danger)'
    : latestRUL < WARN_THRESHOLD_H ? 'var(--color-warning)'
    : 'var(--color-success)';

  if (dataPoints.length === 0) return null;

  const hasBand = chartData.some((d) => d.rul_band !== null);

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Remaining Useful Life (RUL)</span>
        {latestRUL !== null && (
          <span style={{ fontSize: '1.25rem', fontWeight: 800, color: rulColor }}>
            {latestRUL > 99 ? '>99' : latestRUL.toFixed(1)} h
          </span>
        )}
        {latestRUL === null && (
          <span style={{ fontSize: '0.8rem', color: 'var(--color-text-muted)' }}>
            Trend stable — no threshold crossing projected
          </span>
        )}
      </div>

      {chartData.length < 3 ? (
        <p style={{ textAlign: 'center', color: 'var(--color-text-muted)', fontSize: '0.85rem', padding: '2rem 0' }}>
          Accumulating prediction history…
        </p>
      ) : (
        <ResponsiveContainer width="100%" height={230}>
          <ComposedChart data={chartData} margin={{ top: 8, right: 30, bottom: 5, left: 10 }}>
            <defs>
              <linearGradient id="rulGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="bandGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"  stopColor="#64748b" stopOpacity={0.45} />
                <stop offset="100%" stopColor="#64748b" stopOpacity={0.18} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t_h"
              label={{ value: 'Time elapsed [h]', position: 'insideBottomRight', offset: -5, fontSize: 10 }}
              fontSize={10}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <YAxis
              label={{ value: 'RUL [h]', angle: -90, position: 'insideLeft', fontSize: 10 }}
              fontSize={10}
              domain={[0, 'auto']}
              allowDataOverflow={false}
            />
            <Tooltip
              contentStyle={{ fontSize: '0.78rem', borderRadius: '8px' }}
              formatter={(value: number | [number, number], name: string) => {
                if (name === 'rul_band') {
                  if (Array.isArray(value)) return [`${value[0].toFixed(1)} – ${value[1].toFixed(1)} h`, '±1σ band'];
                  return [String(value), name];
                }
                if (name === 'ma2') return [`${(value as number).toFixed(1)} h`, '2-pt MA'];
                return [`${(value as number) > 99 ? '>99' : (value as number).toFixed(1)} h`, 'Est. RUL'];
              }}
              labelFormatter={(t: number) => `t = ${t.toFixed(2)} h`}
            />
            <Legend verticalAlign="top" height={26} />

            <ReferenceLine
              y={WARN_THRESHOLD_H}
              stroke="#f59e0b"
              strokeDasharray="6 3"
              label={{ value: `Warn ${WARN_THRESHOLD_H}h`, fill: '#f59e0b', fontSize: 9, position: 'right' }}
            />
            <ReferenceLine
              y={DANGER_THRESHOLD_H}
              stroke="#ef4444"
              strokeDasharray="6 3"
              label={{ value: `Critical ${DANGER_THRESHOLD_H}h`, fill: '#ef4444', fontSize: 9, position: 'right' }}
            />

            {/* ±1σ uncertainty band */}
            {hasBand && (
              <Area
                type="monotone"
                dataKey="rul_band"
                fill="url(#bandGradient)"
                stroke="none"
                dot={false}
                legendType="none"
                name="rul_band"
                connectNulls={false}
                activeDot={false}
              />
            )}

            {/* RUL estimate */}
            <Area
              type="monotone"
              dataKey="remaining_runtime_h"
              stroke="#3b82f6"
              strokeWidth={2}
              fill="url(#rulGradient)"
              dot={false}
              name="remaining_runtime_h"
              connectNulls={false}
            />

            {/* 2-point moving average (dashed) */}
            <Line
              type="monotone"
              dataKey="ma2"
              stroke="#64748b"
              strokeWidth={1.5}
              strokeDasharray="5 3"
              dot={false}
              name="ma2"
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
