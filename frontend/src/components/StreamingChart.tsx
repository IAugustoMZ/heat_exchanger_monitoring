import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';
import type { DataPoint } from '../types';

interface Props {
  dataPoints: DataPoint[];
}

const ALARM_THRESHOLD = 943.3;

export default function StreamingChart({ dataPoints }: Props) {
  // Downsample if too many points for performance
  const chartData = useMemo(() => {
    const maxPoints = 500;
    if (dataPoints.length <= maxPoints) return dataPoints;
    const step = Math.ceil(dataPoints.length / maxPoints);
    return dataPoints.filter((_, i) => i % step === 0 || i === dataPoints.length - 1);
  }, [dataPoints]);

  if (dataPoints.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <p style={{ color: 'var(--color-text-muted)', fontSize: '0.875rem' }}>
          Select a scenario and click <strong>Begin</strong> to start streaming data
        </p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* ΔP Chart */}
      <div className="card">
        <div className="card-header">Pressure Drop (ΔP) — Actual vs ML Predicted</div>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="t_h"
              label={{ value: 'Time [h]', position: 'insideBottomRight', offset: -5 }}
              fontSize={11}
            />
            <YAxis
              label={{ value: 'ΔP [Pa]', angle: -90, position: 'insideLeft' }}
              fontSize={11}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{ fontSize: '0.8rem', borderRadius: '8px' }}
              formatter={(value: number) => [`${value.toFixed(1)} Pa`]}
            />
            <Legend verticalAlign="top" height={30} />
            <ReferenceLine
              y={ALARM_THRESHOLD}
              stroke="#ef4444"
              strokeDasharray="8 4"
              label={{ value: `Alarm ${ALARM_THRESHOLD} Pa`, fill: '#ef4444', fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="delta_P_Pa"
              stroke="#1a3a5c"
              strokeWidth={2}
              dot={false}
              name="ΔP Actual"
            />
            <Line
              type="monotone"
              dataKey="dP_total_predicted"
              stroke="#3b82f6"
              strokeWidth={2}
              strokeDasharray="5 3"
              dot={false}
              name="ΔP Predicted (ML)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Temperature Chart */}
      <div className="card">
        <div className="card-header">Temperature Readings</div>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="t_h" fontSize={11} />
            <YAxis
              label={{ value: 'T [K]', angle: -90, position: 'insideLeft' }}
              fontSize={11}
              domain={['auto', 'auto']}
            />
            <Tooltip
              contentStyle={{ fontSize: '0.8rem', borderRadius: '8px' }}
              formatter={(value: number) => [`${value.toFixed(1)} K`]}
            />
            <Legend verticalAlign="top" height={30} />
            <Line type="monotone" dataKey="T_h_in_K" stroke="#f97316" strokeWidth={1.5} dot={false} name="T_h,in" />
            <Line type="monotone" dataKey="T_h_out_K" stroke="#ef4444" strokeWidth={1.5} dot={false} name="T_h,out" />
            <Line type="monotone" dataKey="T_c_in_K" stroke="#06b6d4" strokeWidth={1.5} dot={false} name="T_c,in" />
            <Line type="monotone" dataKey="T_c_out_K" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="T_c,out" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
