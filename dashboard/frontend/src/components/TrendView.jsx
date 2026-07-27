import { useEffect, useState } from 'react'
import {
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts'
import { getTrend } from '../api.js'
import './TrendView.css'

function buildChartData(trend) {
  const points = trend.timestamps.map((t, i) => ({
    timestamp: t,
    basis_weight: trend.basis_weight[i],
    off_spec: trend.off_spec[i],
    projected: null,
  }))
  if (points.length) {
    points[points.length - 1].projected = points[points.length - 1].basis_weight
  }
  trend.extrapolation.basis_weight.forEach((v, i) => {
    points.push({
      timestamp: trend.extrapolation.start_timestamp + i,
      basis_weight: null,
      off_spec: false,
      projected: v,
    })
  })
  return points
}

function OffSpecDot(props) {
  const { cx, cy, payload } = props
  if (!payload.off_spec) return null
  return <circle cx={cx} cy={cy} r={2.5} fill="var(--honeywell-red-muted)" stroke="none" />
}

function Strip({ label, values, timestamps, color, unit }) {
  const data = timestamps.map((t, i) => ({ timestamp: t, value: values[i] }))
  const latest = values[values.length - 1]
  return (
    <div className="strip">
      <div className="strip-label">
        <span>{label}</span>
        <span className="strip-value" style={{ color }}>
          {latest?.toFixed(2)}
          {unit}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={46}>
        <ComposedChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 4 }}>
          <YAxis hide domain={['auto', 'auto']} />
          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function TrendView({ endIndex, onOutOfRange }) {
  const [trend, setTrend] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getTrend(endIndex)
      .then((data) => {
        if (!cancelled) {
          setTrend(data)
          setError(null)
        }
      })
      .catch((err) => {
        if (cancelled) return
        setError(err.message)
        onOutOfRange()
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endIndex])

  return (
    <div className="panel trend-view">
      <div className="trend-header">
        <div>
          <h2 className="panel-title">Basis Weight Trend</h2>
          <p className="panel-subtitle">±2.5% tolerance band · dashed line shows projected trend</p>
        </div>
        {trend && (
          <span className="tag" style={{ color: 'var(--honeywell-gray)', border: 'none', backgroundColor: '#F0F0F0' }}>
            Target: {trend.basis_weight_setpoint.toFixed(1)}
          </span>
        )}
      </div>

      {error && !trend && <div className="error-banner">{error}</div>}

      {trend && (
        <>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={buildChartData(trend)} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
              <XAxis
                dataKey="timestamp"
                tick={{ fontSize: 11, fill: 'var(--muted)' }}
                tickLine={false}
                axisLine={{ stroke: 'var(--hairline)' }}
              />
              <YAxis
                tick={{ fontSize: 11, fill: 'var(--muted)' }}
                tickLine={false}
                axisLine={false}
                domain={['auto', 'auto']}
                width={40}
              />
              <Tooltip
                contentStyle={{
                  background: 'var(--panel-raised)',
                  border: '1px solid var(--hairline)',
                  borderRadius: 8,
                  fontSize: 12,
                }}
                labelStyle={{ color: 'var(--muted)' }}
              />
              <ReferenceArea
                y1={trend.control_band.low}
                y2={trend.control_band.high}
                fill="var(--honeywell-red-muted)"
                fillOpacity={0.08}
                stroke="none"
              />
              <ReferenceLine
                y={trend.basis_weight_setpoint}
                stroke="var(--muted)"
                strokeDasharray="3 3"
              />
              <Line
                type="monotone"
                dataKey="basis_weight"
                stroke="var(--honeywell-red-muted)"
                strokeWidth={2}
                dot={<OffSpecDot />}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="projected"
                stroke="var(--amber)"
                strokeWidth={2}
                strokeDasharray="6 4"
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>

          <div className="strip-group">
            <Strip
              label="Moisture"
              values={trend.moisture}
              timestamps={trend.timestamps}
              color="var(--honeywell-red-muted)"
              unit="%"
            />
            <Strip
              label="Ash Content"
              values={trend.ash_content}
              timestamps={trend.timestamps}
              color="var(--amber)"
              unit="%"
            />
          </div>
        </>
      )}
    </div>
  )
}