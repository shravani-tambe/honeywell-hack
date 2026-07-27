import { useEffect, useState } from 'react'
import { getRisk } from '../api.js'
import './RiskPanel.css'

const LOW_BAND = 0.15
const HIGH_BAND = 0.95

function zoneFor(risk) {
  if (risk >= HIGH_BAND) return { label: 'Critical', color: 'var(--honeywell-red-muted)' }
  if (risk >= LOW_BAND) return { label: 'Elevated', color: 'var(--amber)' }
  return { label: 'Normal', color: 'var(--honeywell-gray)' }
}

export default function RiskPanel({ endIndex, onOutOfRange }) {
  const [risk, setRisk] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    getRisk(endIndex)
      .then((data) => {
        if (!cancelled) {
          setRisk(data)
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

  const zone = risk ? zoneFor(risk.risk) : null
  const pct = risk ? Math.round(risk.risk * 1000) / 10 : 0

  return (
    <div className="panel risk-panel">
      <h2 className="panel-title">Off-Spec Risk</h2>
      <p className="panel-subtitle">Risk prediction for next 5 seconds</p>

      {error && !risk && <div className="error-banner">{error}</div>}

      {risk && (
        <>
          <div className="risk-readout">
            <span className="risk-value" style={{ color: zone.color }}>
              {pct.toFixed(1)}
            </span>
            <span className="risk-unit">%</span>
            <span className="risk-zone" style={{ color: zone.color, backgroundColor: zone.color === 'var(--honeywell-red-muted)' ? '#FDF2F2' : '#F0F0F0', padding: '4px 10px', borderRadius: '4px', fontSize: '11px', fontWeight: '500' }}>
              {zone.label}
            </span>
          </div>

          <div className="risk-bar">
            <div className="risk-bar-tick" style={{ left: `${LOW_BAND * 100}%` }} />
            <div className="risk-bar-tick" style={{ left: `${HIGH_BAND * 100}%` }} />
            <div
              className="risk-bar-fill"
              style={{ width: `${pct}%`, background: zone.color }}
            />
          </div>

          <div className="risk-meta">
            <div>
              <span className="risk-meta-label">Grade</span>
              <span className="risk-meta-value">{risk.grade.replace('_', ' ')}</span>
            </div>
            <div>
              <span className="risk-meta-label">Transition</span>
              <span className="risk-meta-value">#{risk.transition_id}</span>
            </div>
            <div>
              <span className="risk-meta-label">State</span>
              <span className="risk-meta-value">
                {risk.is_transitioning ? 'Transitioning' : 'Steady state'}
              </span>
            </div>
            <div>
              <span className="risk-meta-label">Now</span>
              <span className="risk-meta-value" style={{ color: risk.off_spec_now ? 'var(--honeywell-red-muted)' : 'var(--honeywell-gray)' }}>
                {risk.off_spec_now ? 'Off-spec' : 'In-spec'}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  )
}