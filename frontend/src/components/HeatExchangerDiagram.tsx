import type { DataPoint } from '../types';

interface Props {
  currentPoint: DataPoint | null;
}

const fmt = (v: number | string, unit: string) =>
  typeof v === 'number' ? `${v.toFixed(1)} ${unit}` : '--';

/** Small P&ID instrument circle bubble */
function InstrumentBubble({
  x, y, tag, value, unit, color = '#1a3a5c',
}: { x: number; y: number; tag: string; value: number | string; unit: string; color?: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={22} fill="white" stroke={color} strokeWidth={1.5} />
      <text x={x} y={y - 5} textAnchor="middle" fontSize={8} fill={color} fontWeight={700}>{tag}</text>
      <line x1={x - 18} y1={y} x2={x + 18} y2={y} stroke={color} strokeWidth={0.8} />
      <text x={x} y={y + 11} textAnchor="middle" fontSize={9} fill={color} fontWeight={600}>
        {fmt(value, unit)}
      </text>
    </g>
  );
}

/** Hatching pattern for engineering diagram walls */
function HatchPattern({ id, color = '#94a3b8' }: { id: string; color?: string }) {
  return (
    <pattern id={id} patternUnits="userSpaceOnUse" width={6} height={6}>
      <path d="M-1,1 l2,-2 M0,6 l6,-6 M5,7 l2,-2" stroke={color} strokeWidth={0.8} />
    </pattern>
  );
}

/** Flange bolt circle */
function FlangeRing({ x, y, h, bolts = 8 }: { x: number; y: number; h: number; bolts?: number }) {
  const boltR = 2.5;
  const cx = x;
  const spacing = h / (bolts + 1);
  return (
    <g>
      {Array.from({ length: bolts }, (_, i) => (
        <circle key={i} cx={cx} cy={y + spacing * (i + 1)} r={boltR}
          fill="#78716c" stroke="#44403c" strokeWidth={0.6} />
      ))}
    </g>
  );
}

export default function HeatExchangerDiagram({ currentPoint }: Props) {
  const T_h_in  = currentPoint?.T_h_in_K  ?? '--';
  const T_h_out = currentPoint?.T_h_out_K ?? '--';
  const T_c_in  = currentPoint?.T_c_in_K  ?? '--';
  const T_c_out = currentPoint?.T_c_out_K ?? '--';
  const dP      = currentPoint?.delta_P_Pa ?? '--';
  const frost   = currentPoint ? (currentPoint.delta_f_max_m * 1000).toFixed(2) : '--';
  const isAlarm = currentPoint?.alarm_predicted ?? false;

  // Frost thickness in pixels for tube overlay (scale: 1 mm → 3 px, max 8 px)
  const frostPx = currentPoint ? Math.min(currentPoint.delta_f_max_m * 3000, 8) : 0;

  // Shell geometry
  const SX = 170, SY = 90, SW = 480, SH = 160;
  const shellMidY = SY + SH / 2;

  // Tube layout: 8 tubes evenly spaced
  const tubeCount = 8;
  const tubeH = 7;
  const tubeSpacing = (SH - 40) / (tubeCount + 1);
  const tubes = Array.from({ length: tubeCount }, (_, i) => SY + 20 + tubeSpacing * (i + 1) - tubeH / 2);

  // Baffle x-positions (5 baffles to create more realistic zigzag)
  const baffleCount = 5;
  const baffles = Array.from({ length: baffleCount }, (_, i) =>
    SX + (SW / (baffleCount + 1)) * (i + 1)
  );

  // Channel head / bonnet dimensions
  const chW = 36;
  const tsW = 14; // tube sheet width

  // Rear head ellipse radius
  const rearRx = 50;

  return (
    <div className="card">
      <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span>E-101 — Shell &amp; Tube Heat Exchanger</span>
        <span style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)', fontWeight: 400 }}>
          TEMA Type BEM · 1 Pass
        </span>
      </div>

      <svg viewBox="-30 -10 920 490" style={{ width: '100%', height: 'auto', display: 'block' }}>
        <defs>
          <HatchPattern id="hatch-steel" color="#64748b" />
          <HatchPattern id="hatch-frost" color="#bae6fd" />

          {/* Shell interior warm yellow fill */}
          <linearGradient id="shellFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#fef9c3" />
            <stop offset="50%" stopColor="#fde68a" />
            <stop offset="100%" stopColor="#fef9c3" />
          </linearGradient>

          {/* Hot tube fill — golden yellow like the reference */}
          <linearGradient id="tubeFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#fbbf24" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="100%" stopColor="#fbbf24" />
          </linearGradient>

          {/* Channel head fill */}
          <linearGradient id="channelFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#e2e8f0" />
            <stop offset="50%" stopColor="#cbd5e1" />
            <stop offset="100%" stopColor="#e2e8f0" />
          </linearGradient>

          {/* Steel metallic fill */}
          <linearGradient id="steelFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#94a3b8" />
            <stop offset="50%" stopColor="#64748b" />
            <stop offset="100%" stopColor="#94a3b8" />
          </linearGradient>

          {/* Rear head dark fill (like the reference image dark red/brown end cap) */}
          <linearGradient id="rearHeadFill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#e2e8f0" />
            <stop offset="60%" stopColor="#7f1d1d" />
            <stop offset="100%" stopColor="#991b1b" />
          </linearGradient>

          {/* Frost fill */}
          <linearGradient id="frostFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(224,242,255,0.95)" />
            <stop offset="100%" stopColor="rgba(186,230,253,0.8)" />
          </linearGradient>

          {/* Alarm glow */}
          <filter id="alarmGlow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>

          {/* Arrowhead markers */}
          <marker id="arrowHot" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#ea580c" />
          </marker>
          <marker id="arrowCold" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#0284c7" />
          </marker>
          <marker id="arrowColdLeft" markerWidth="8" markerHeight="8" refX="2" refY="3" orient="auto">
            <path d="M8,0 L8,6 L0,3 z" fill="#0284c7" />
          </marker>

          {/* Shell-side flow curl marker */}
          <marker id="arrowFlow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" fill="#dc2626" opacity={0.55} />
          </marker>

          {/* Clip for shell interior */}
          <clipPath id="shellClip">
            <rect x={SX} y={SY} width={SW} height={SH} />
          </clipPath>

          {/* Clip for rear head — only show right half of ellipse */}
          <clipPath id="rearHeadClip">
            <rect x={SX + SW} y={SY - 10} width={rearRx + 10} height={SH + 20} />
          </clipPath>
        </defs>

        {/* ══════════════════════════════════════════
            SUPPORT SADDLES (legs)
        ══════════════════════════════════════════ */}
        {[SX + 80, SX + SW - 80].map((lx, i) => (
          <g key={`leg-${i}`}>
            {/* Saddle arc */}
            <path d={`M${lx - 25},${SY + SH + 3} Q${lx},${SY + SH + 14} ${lx + 25},${SY + SH + 3}`}
              fill="none" stroke="#64748b" strokeWidth={3} />
            {/* Left leg */}
            <line x1={lx - 25} y1={SY + SH + 3} x2={lx - 30} y2={SY + SH + 35}
              stroke="#64748b" strokeWidth={3} />
            {/* Right leg */}
            <line x1={lx + 25} y1={SY + SH + 3} x2={lx + 30} y2={SY + SH + 35}
              stroke="#64748b" strokeWidth={3} />
            {/* Base plate */}
            <rect x={lx - 35} y={SY + SH + 33} width={70} height={5}
              fill="#94a3b8" stroke="#475569" strokeWidth={1} />
          </g>
        ))}

        {/* ══════════════════════════════════════════
            SHELL BODY — cylindrical look
        ══════════════════════════════════════════ */}

        {/* Shell main rectangle */}
        <rect x={SX} y={SY} width={SW} height={SH}
          fill="url(#shellFill)" stroke="#475569" strokeWidth={2.5} />

        {/* Shell wall thickness — top */}
        <rect x={SX} y={SY} width={SW} height={8}
          fill="url(#steelFill)" stroke="none" opacity={0.7} />
        <rect x={SX} y={SY} width={SW} height={8}
          fill="url(#hatch-steel)" stroke="none" opacity={0.4} />

        {/* Shell wall thickness — bottom */}
        <rect x={SX} y={SY + SH - 8} width={SW} height={8}
          fill="url(#steelFill)" stroke="none" opacity={0.7} />
        <rect x={SX} y={SY + SH - 8} width={SW} height={8}
          fill="url(#hatch-steel)" stroke="none" opacity={0.4} />

        {/* ══════════════════════════════════════════
            REAR HEAD — Right-half elliptical dished end (like reference)
        ══════════════════════════════════════════ */}
        <g clipPath="url(#rearHeadClip)">
          <ellipse cx={SX + SW} cy={shellMidY} rx={rearRx} ry={SH / 2}
            fill="url(#rearHeadFill)" stroke="#475569" strokeWidth={2.5} />
          {/* Inner U-turn curves visible through the head opening */}
          <ellipse cx={SX + SW} cy={shellMidY} rx={rearRx * 0.6} ry={SH / 2 * 0.6}
            fill="none" stroke="#fbbf24" strokeWidth={1.5} opacity={0.6} />
          <ellipse cx={SX + SW} cy={shellMidY} rx={rearRx * 0.35} ry={SH / 2 * 0.35}
            fill="none" stroke="#fbbf24" strokeWidth={1} opacity={0.4} />
        </g>

        {/* Right tube sheet (at shell-rear head junction) */}
        <rect x={SX + SW - tsW} y={SY} width={tsW} height={SH}
          fill="url(#steelFill)" stroke="#475569" strokeWidth={2} />
        <rect x={SX + SW - tsW} y={SY} width={tsW} height={SH}
          fill="url(#hatch-steel)" stroke="none" opacity={0.5} />
        {/* Flange bolts at right tube sheet */}
        <FlangeRing x={SX + SW - tsW / 2} y={SY} h={SH} bolts={10} />

        {/* ══════════════════════════════════════════
            LEFT CHANNEL HEAD (Front bonnet / water box)
        ══════════════════════════════════════════ */}

        {/* Left tube sheet */}
        <rect x={SX} y={SY} width={tsW} height={SH}
          fill="url(#steelFill)" stroke="#475569" strokeWidth={2} />
        <rect x={SX} y={SY} width={tsW} height={SH}
          fill="url(#hatch-steel)" stroke="none" opacity={0.5} />
        {/* Flange bolts at left tube sheet */}
        <FlangeRing x={SX + tsW / 2} y={SY} h={SH} bolts={10} />

        {/* Channel head box */}
        <rect x={SX - chW} y={SY} width={chW} height={SH}
          fill="url(#channelFill)" stroke="#475569" strokeWidth={2} rx={3} />
        {/* Channel head wall hatch */}
        <rect x={SX - chW} y={SY} width={chW} height={8}
          fill="url(#hatch-steel)" opacity={0.5} />
        <rect x={SX - chW} y={SY + SH - 8} width={chW} height={8}
          fill="url(#hatch-steel)" opacity={0.5} />

        {/* Flange ring between channel head and tube sheet */}
        <rect x={SX - 4} y={SY - 4} width={8} height={SH + 8}
          fill="url(#steelFill)" stroke="#475569" strokeWidth={1.5} rx={1} />

        {/* Left-most flange (channel head end) */}
        <rect x={SX - chW - 4} y={SY - 4} width={8} height={SH + 8}
          fill="url(#steelFill)" stroke="#475569" strokeWidth={1.5} rx={1} />
        <FlangeRing x={SX - chW} y={SY - 4} h={SH + 8} bolts={10} />

        {/* ══════════════════════════════════════════
            BAFFLE PLATES
        ══════════════════════════════════════════ */}
        {baffles.map((bx, i) => (
          <g key={`baffle-${i}`}>
            {i % 2 === 0 ? (
              // Cut at top — baffle covers bottom portion
              <rect x={bx - 3} y={SY + SH * 0.22} width={6} height={SH * 0.78 - 8}
                fill="url(#steelFill)" stroke="#475569" strokeWidth={1.2} />
            ) : (
              // Cut at bottom — baffle covers top portion
              <rect x={bx - 3} y={SY + 8} width={6} height={SH * 0.78 - 8}
                fill="url(#steelFill)" stroke="#475569" strokeWidth={1.2} />
            )}
          </g>
        ))}

        {/* ══════════════════════════════════════════
            SHELL-SIDE FLOW SWIRL ARROWS (like the reference spirals)
        ══════════════════════════════════════════ */}
        <g clipPath="url(#shellClip)" opacity={0.45}>
          {/* Between each pair of baffles, draw curved flow paths */}
          {[SX + 20, ...baffles, SX + SW - tsW - 20].reduce<[number, number][]>((pairs, val, idx, arr) => {
            if (idx > 0) pairs.push([arr[idx - 1], val]);
            return pairs;
          }, []).map(([x1, x2], i) => {
            const mx = (x1 + x2) / 2;
            const dir = i % 2 === 0 ? 1 : -1;
            const cY1 = shellMidY - dir * (SH * 0.28);
            const cY2 = shellMidY + dir * (SH * 0.28);
            return (
              <g key={`curl-${i}`}>
                {/* Curved flowing path with arrowhead */}
                <path
                  d={`M${x1 + 8},${shellMidY}
                      C${x1 + 15},${cY1} ${mx},${cY1} ${mx},${shellMidY}
                      C${mx},${cY2} ${x2 - 15},${cY2} ${x2 - 8},${shellMidY}`}
                  fill="none" stroke="#dc2626" strokeWidth={1.8}
                  strokeDasharray="6,3" markerEnd="url(#arrowFlow)" />
                {/* Second curl offset */}
                <path
                  d={`M${x1 + 12},${shellMidY + dir * 10}
                      C${x1 + 20},${cY1 + dir * 5} ${mx + 5},${cY1 + dir * 5} ${mx + 5},${shellMidY + dir * 10}
                      C${mx + 5},${cY2 - dir * 5} ${x2 - 10},${cY2 - dir * 5} ${x2 - 4},${shellMidY + dir * 10}`}
                  fill="none" stroke="#dc2626" strokeWidth={1.2}
                  strokeDasharray="4,4" opacity={0.6} />
              </g>
            );
          })}
        </g>

        {/* Shell-side flow label */}
        <text x={SX + SW / 2} y={SY + SH - 14} textAnchor="middle"
          fontSize={9} fill="#475569" fontStyle="italic" opacity={0.8}>
          SHELL SIDE — LNG (cold)
        </text>

        {/* ══════════════════════════════════════════
            TUBES — golden yellow horizontal tubes
        ══════════════════════════════════════════ */}
        {tubes.map((ty, i) => (
          <g key={`tube-${i}`}>
            {/* Tube barrel spanning between tube sheets */}
            <rect x={SX + tsW} y={ty} width={SW - tsW * 2} height={tubeH}
              fill="url(#tubeFill)" stroke="#b45309" strokeWidth={0.8} rx={tubeH / 2} />

            {/* Frost overlay */}
            {frostPx > 0.3 && (
              <rect
                x={SX + tsW}
                y={ty - frostPx}
                width={SW - tsW * 2}
                height={tubeH + frostPx * 2}
                rx={tubeH / 2 + frostPx}
                fill="url(#frostFill)"
                stroke="#bae6fd"
                strokeWidth={0.8}
                opacity={Math.min(0.3 + frostPx / 8 * 0.65, 0.95)}
              />
            )}
          </g>
        ))}

        {/* Centre-line (dashed, like reference image) */}
        <line x1={SX - chW - 10} y1={shellMidY} x2={SX + SW + rearRx - 10} y2={shellMidY}
          stroke="#1e40af" strokeWidth={0.8} strokeDasharray="8,4" opacity={0.35} />

        {/* Tube-side flow label */}
        <text x={SX + SW / 2} y={SY + SH + 22} textAnchor="middle"
          fontSize={9} fill="#7c2d12" fontStyle="italic">
          TUBE SIDE — Feed Gas (hot)
        </text>

        {/* ══════════════════════════════════════════
            NOZZLES with flanges
        ══════════════════════════════════════════ */}

        {/* Shell-side INLET nozzle (top, right side) — "Heating/Cooling Fluid" in reference */}
        <g>
          <rect x={SX + SW - 100} y={SY - 38} width={24} height={38}
            fill="url(#channelFill)" stroke="#475569" strokeWidth={1.5} />
          {/* Nozzle flange ring */}
          <rect x={SX + SW - 104} y={SY - 40} width={32} height={6}
            fill="url(#steelFill)" stroke="#475569" strokeWidth={1} rx={1} />
          <text x={SX + SW - 88} y={SY - 44} textAnchor="middle" fontSize={8} fill="#1e293b" fontWeight={600}>N3</text>
        </g>

        {/* Shell-side OUTLET nozzle (bottom, left side) */}
        <g>
          <rect x={SX + 76} y={SY - 38} width={24} height={38}
            fill="url(#channelFill)" stroke="#475569" strokeWidth={1.5} />
          <rect x={SX + 72} y={SY - 40} width={32} height={6}
            fill="url(#steelFill)" stroke="#475569" strokeWidth={1} rx={1} />
          <text x={SX + 88} y={SY - 44} textAnchor="middle" fontSize={8} fill="#1e293b" fontWeight={600}>N4</text>
        </g>

        {/* Tube-side inlet nozzle — bottom of left channel head */}
        <g>
          <rect x={SX - chW / 2 - 12} y={SY + SH} width={24} height={34}
            fill="#fef3c7" stroke="#b45309" strokeWidth={1.5} />
          <rect x={SX - chW / 2 - 16} y={SY + SH + 30} width={32} height={6}
            fill="url(#steelFill)" stroke="#475569" strokeWidth={1} rx={1} />
          <text x={SX - chW / 2} y={SY + SH + 50} textAnchor="middle" fontSize={8} fill="#7c2d12" fontWeight={600}>N1</text>
        </g>

        {/* Tube-side outlet nozzle — bottom of left channel head (other side) */}
        <g>
          <rect x={SX - chW + 2} y={SY + SH} width={24} height={34}
            fill="#fef3c7" stroke="#b45309" strokeWidth={1.5} />
          <rect x={SX - chW - 2} y={SY + SH + 30} width={32} height={6}
            fill="url(#steelFill)" stroke="#475569" strokeWidth={1} rx={1} />
          <text x={SX - chW + 14} y={SY + SH + 50} textAnchor="middle" fontSize={8} fill="#7c2d12" fontWeight={600}>N2</text>
        </g>

        {/* ══════════════════════════════════════════
            PIPING LINES with flow arrows
        ══════════════════════════════════════════ */}

        {/* Hot gas inlet pipe — comes from bottom-left */}
        <polyline points={`30,${SY + SH + 60} ${SX - chW / 2},${SY + SH + 60} ${SX - chW / 2},${SY + SH + 34}`}
          fill="none" stroke="#ea580c" strokeWidth={3} markerEnd="url(#arrowHot)" />
        <text x={10} y={SY + SH + 56} fontSize={9} fill="#64748b" fontWeight={600}>Product</text>

        {/* Hot gas outlet pipe — exits from bottom channel head */}
        <polyline points={`${SX - chW + 14},${SY + SH + 34} ${SX - chW + 14},${SY + SH + 76} 30,${SY + SH + 76}`}
          fill="none" stroke="#ea580c" strokeWidth={3} markerEnd="url(#arrowColdLeft)" />
        <text x={10} y={SY + SH + 72} fontSize={9} fill="#64748b" fontWeight={600}>Product Out</text>

        {/* LNG inlet — Heating/Cooling Fluid from top */}
        <polyline points={`${SX + SW - 88},${SY - 66} ${SX + SW - 88},${SY - 40}`}
          fill="none" stroke="#0284c7" strokeWidth={3} markerEnd="url(#arrowCold)" />
        <text x={SX + SW - 88} y={SY - 72} textAnchor="middle" fontSize={9} fill="#64748b" fontWeight={600}>
          Heating / Cooling
        </text>
        <text x={SX + SW - 88} y={SY - 62} textAnchor="middle" fontSize={9} fill="#64748b" fontWeight={600}>
          Fluid
        </text>

        {/* LNG outlet — from left nozzle, up and out */}
        <polyline points={`${SX + 88},${SY - 40} ${SX + 88},${SY - 56} 30,${SY - 56}`}
          fill="none" stroke="#0284c7" strokeWidth={3} markerEnd="url(#arrowColdLeft)" />
        <text x={10} y={SY - 52} fontSize={9} fill="#64748b" fontWeight={600}>LNG OUT</text>

        {/* ══════════════════════════════════════════
            INSTRUMENT BUBBLES — P&ID style
        ══════════════════════════════════════════ */}

        {/* TI-101 — Hot gas inlet temp (on product inlet pipe) */}
        <line x1={100} y1={SY + SH + 60} x2={100} y2={SY + SH + 96}
          stroke="#94a3b8" strokeWidth={0.8} strokeDasharray="3,2" />
        <InstrumentBubble x={100} y={SY + SH + 118} tag="TI-101"
          value={T_h_in} unit="K" color="#ea580c" />

        {/* TI-102 — Hot gas outlet temp (on product outlet pipe) */}
        <line x1={100} y1={SY + SH + 76} x2={100} y2={SY + SH + 142}
          stroke="#94a3b8" strokeWidth={0.8} strokeDasharray="3,2" />
        <InstrumentBubble x={100} y={SY + SH + 164} tag="TI-102"
          value={T_h_out} unit="K" color="#ef4444" />

        {/* TI-103 — LNG inlet temp (on top-right nozzle pipe) */}
        <line x1={SX + SW - 60} y1={SY - 56} x2={SX + SW - 30} y2={SY - 56}
          stroke="#94a3b8" strokeWidth={0.8} strokeDasharray="3,2" />
        <InstrumentBubble x={SX + SW - 8} y={SY - 56} tag="TI-103"
          value={T_c_in} unit="K" color="#0284c7" />

        {/* TI-104 — LNG outlet temp (on LNG outlet pipe) */}
        <line x1={60} y1={SY - 56} x2={48} y2={SY - 56}
          stroke="#94a3b8" strokeWidth={0.8} strokeDasharray="3,2" />
        <InstrumentBubble x={25} y={SY - 56} tag="TI-104"
          value={T_c_out} unit="K" color="#3b82f6" />

        {/* PDI-101 — differential pressure (on shell mid-bottom) */}
        <line x1={SX + SW / 2} y1={SY + SH} x2={SX + SW / 2} y2={SY + SH + 56}
          stroke="#94a3b8" strokeWidth={0.8} strokeDasharray="3,2" />
        <g>
          <rect
            x={SX + SW / 2 - 42} y={SY + SH + 56}
            width={84} height={46} rx={8}
            fill={isAlarm ? '#fef2f2' : 'white'}
            stroke={isAlarm ? '#ef4444' : '#334155'}
            strokeWidth={isAlarm ? 2 : 1.5}
            filter={isAlarm ? 'url(#alarmGlow)' : undefined}
          />
          <text x={SX + SW / 2} y={SY + SH + 69} textAnchor="middle" fontSize={8}
            fill={isAlarm ? '#ef4444' : '#334155'} fontWeight={700}>PDI-101</text>
          <line x1={SX + SW / 2 - 36} y1={SY + SH + 72} x2={SX + SW / 2 + 36} y2={SY + SH + 72}
            stroke="#94a3b8" strokeWidth={0.8} />
          <text x={SX + SW / 2} y={SY + SH + 88} textAnchor="middle" fontSize={11}
            fill={isAlarm ? '#ef4444' : '#1a3a5c'} fontWeight={800}>
            {fmt(dP, 'Pa')}
          </text>
        </g>

        {/* Frost indicator — bottom right */}
        <g>
          <line x1={SX + SW - 60} y1={SY + SH} x2={SX + SW - 60} y2={SY + SH + 56}
            stroke="#94a3b8" strokeWidth={0.8} strokeDasharray="3,2" />
          <rect x={SX + SW - 108} y={SY + SH + 56} width={98} height={46} rx={8}
            fill="white" stroke="#67e8f9" strokeWidth={1.5} />
          <text x={SX + SW - 59} y={SY + SH + 69} textAnchor="middle"
            fontSize={8} fill="#0891b2" fontWeight={700}>FI-101 Frost</text>
          <line x1={SX + SW - 102} y1={SY + SH + 72} x2={SX + SW - 16} y2={SY + SH + 72}
            stroke="#94a3b8" strokeWidth={0.8} />
          <text x={SX + SW - 59} y={SY + SH + 88} textAnchor="middle"
            fontSize={11} fill="#0891b2" fontWeight={700}>
            {frost} mm
          </text>
        </g>

        {/* ══════════════════════════════════════════
            ALARM & STATUS INDICATORS
        ══════════════════════════════════════════ */}

        {isAlarm && (
          <g>
            <circle cx={SX + SW / 2} cy={SY - 30} r={14} fill="#ef4444" opacity={0.92}>
              <animate attributeName="opacity" dur="0.9s" values="0.92;0.3;0.92" repeatCount="indefinite" />
            </circle>
            <text x={SX + SW / 2} y={SY - 25} textAnchor="middle" fontSize={13}
              fill="white" fontWeight={900}>!</text>
            <text x={SX + SW / 2} y={SY - 48} textAnchor="middle" fontSize={9}
              fill="#ef4444" fontWeight={700}>FREEZING ALARM</text>
          </g>
        )}

        {/* ML model tag (bottom-left status) */}
        <g>
          <circle cx={SX + 20} cy={SY + SH + 12} r={4}
            fill={currentPoint?.model_available ? '#10b981' : '#94a3b8'} />
          <text x={SX + 28} y={SY + SH + 16} fontSize={8.5} fill="#64748b">
            ML · Lasso R²=0.970 · {currentPoint?.model_available ? 'Online' : 'Awaiting…'}
          </text>
        </g>

        {/* Equipment tag */}
        <text x={SX + 20} y={SY + 22} fontSize={9} fill="#334155" fontWeight={700}>E-101</text>
      </svg>
    </div>
  );
}

