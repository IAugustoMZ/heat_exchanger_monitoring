import { useState, useEffect, useCallback, useRef } from 'react';
import { fetchScenarios, createStreamUrl } from '../api';
import type { ScenarioMeta, DataPoint } from '../types';
import HeatExchangerDiagram from '../components/HeatExchangerDiagram';
import StreamingChart from '../components/StreamingChart';
import AlertPanel from '../components/AlertPanel';
import RuntimeForecast from '../components/RuntimeForecast';
import MetricsPanel from '../components/MetricsPanel';
import RULChart from '../components/RULChart';
import ErrorAnalysisCharts from '../components/ErrorAnalysisCharts';

export default function MainPage() {
  const [scenarios, setScenarios] = useState<ScenarioMeta[]>([]);
  const [selectedScenario, setSelectedScenario] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const [currentPoint, setCurrentPoint] = useState<DataPoint | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState<number>(1);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    fetchScenarios()
      .then((data) => {
        setScenarios(data.scenarios);
        if (data.scenarios.length > 0) {
          setSelectedScenario(data.scenarios[0].scenario);
        }
      })
      .catch((err) => setError(`Failed to load scenarios: ${err.message}`));
  }, []);

  const startStream = useCallback(() => {
    if (!selectedScenario) return;

    // Reset state
    setDataPoints([]);
    setCurrentPoint(null);
    setError(null);
    setIsStreaming(true);

    const url = createStreamUrl(selectedScenario, 1, speed);
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener('datapoint', (event) => {
      const point: DataPoint = JSON.parse(event.data);
      setCurrentPoint(point);
      setDataPoints((prev) => [...prev, point]);
    });

    es.addEventListener('complete', () => {
      setIsStreaming(false);
      es.close();
    });

    es.onerror = () => {
      setIsStreaming(false);
      setError('Stream connection lost. The backend may be unavailable.');
      es.close();
    };
  }, [selectedScenario, speed]);

  const stopStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const scenarioMeta = scenarios.find((s) => s.scenario === selectedScenario);

  return (
    <>
      {/* Control bar */}
      <div className="control-bar">
        <select
          className="select-field"
          value={selectedScenario}
          onChange={(e) => {
            setSelectedScenario(e.target.value);
            if (isStreaming) stopStream();
            setDataPoints([]);
            setCurrentPoint(null);
          }}
          disabled={isStreaming}
        >
          {scenarios.map((s) => (
            <option key={s.scenario} value={s.scenario}>
              {s.label} ({s.avg_duration_h}h — {s.avg_points} pts)
            </option>
          ))}
        </select>

        {/* Sampling speed slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '200px' }}>
          <label style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)', whiteSpace: 'nowrap' }}>
            Speed
          </label>
          <input
            type="range"
            min={0.25}
            max={8}
            step={0.25}
            value={speed}
            disabled={isStreaming}
            onChange={(e) => setSpeed(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <span style={{
            fontSize: '0.78rem',
            fontWeight: 700,
            color: 'var(--color-primary)',
            minWidth: '44px',
            textAlign: 'right',
          }}>
            {speed}×
          </span>
        </div>

        {!isStreaming ? (
          <button className="btn btn-primary" onClick={startStream} disabled={!selectedScenario}>
            ▶ Begin
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopStream}>
            ■ Stop
          </button>
        )}

        {isStreaming && currentPoint && (
          <div style={{ flex: 1 }}>
            <div className="progress-bar">
              <div className="progress-bar-fill" style={{ width: `${currentPoint.progress}%` }} />
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)' }}>
              {currentPoint.progress}% — Point {currentPoint.index + 1}/{currentPoint.total_points}
            </span>
          </div>
        )}
      </div>

      {error && <div className="alert-banner danger">⚠ {error}</div>}

      {/* Alert Banner */}
      <AlertPanel currentPoint={currentPoint} />

      {/* Top grid: HX diagram + sidebar metrics */}
      <div className="grid-main">
        <HeatExchangerDiagram currentPoint={currentPoint} />
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <MetricsPanel currentPoint={currentPoint} scenarioMeta={scenarioMeta ?? null} />
          <RuntimeForecast currentPoint={currentPoint} />
        </div>
      </div>

      {/* Full-width streaming charts */}
      <div style={{ marginTop: '1.25rem' }}>
        <StreamingChart dataPoints={dataPoints} />
      </div>
      <div style={{ marginTop: '1.25rem' }}>
        <RULChart dataPoints={dataPoints} />
      </div>
      <div style={{ marginTop: '1.25rem' }}>
        <ErrorAnalysisCharts dataPoints={dataPoints} />
      </div>
    </>
  );
}
