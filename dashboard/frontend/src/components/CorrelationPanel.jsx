import { useEffect, useState } from 'react'
import { getCorrelations } from '../api.js'
import { sourceColor, sourceLabel } from '../sourceColors.js'
import './CorrelationPanel.css'

function strengthColor(correlation) {
  const abs = Math.abs(correlation)
  if (abs >= 0.8) return 'var(--honeywell-red-muted)'
  if (abs >= 0.65) return 'var(--amber)'
  return 'var(--honeywell-gray)'
}

export default function CorrelationPanel() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getCorrelations().then(setData).catch((err) => setError(err.message))
  }, [])

  return (
    <div className="panel correlation-panel">
      <h2 className="panel-title">Discovered Correlations</h2>
      <p className="panel-subtitle">Variable relationships and inference sources</p>

      {error && <div className="error-banner">{error}</div>}

      {data && (
        <>
          <div className="corr-table scroll-list">
            {data.cross_correlation.map((c, i) => (
              <div className="corr-row" key={i}>
                <div className="corr-vars">
                  <span>{c.variable_a}</span>
                  <span className="corr-lag">{c.lag}s →</span>
                  <span>{c.variable_b}</span>
                </div>
                <span className="corr-value" style={{ color: strengthColor(c.correlation) }}>
                  r={c.correlation.toFixed(2)}
                </span>
                <span className="tag" style={{ color: sourceColor(c.source), border: 'none', backgroundColor: sourceColor(c.source) === 'var(--honeywell-red-muted)' ? '#FDF2F2' : '#F0F0F0' }}>
                  {sourceLabel(c.source)}
                </span>
              </div>
            ))}
          </div>

          <div className="corr-subsection">
            <h3>Feature Importance</h3>
            {data.model_importance.slice(0, 6).map((f, i) => (
              <div className="importance-row" key={i}>
                <span className="importance-name">{f.variable_a}</span>
                <div className="importance-bar-track">
                  <div
                    className="importance-bar-fill"
                    style={{ width: `${Math.min(100, f.correlation * 400)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="corr-subsection">
            <h3>Deviation by Scenario</h3>
            {[...data.scenario_deviation_summary]
              .sort((a, b) => b.mean_deviation_pct - a.mean_deviation_pct)
              .map((s) => (
                <div className="scenario-row" key={s.transition_scenario}>
                  <span className="scenario-name">{s.transition_scenario}</span>
                  <span className="scenario-value">
                    {s.mean_deviation_pct}% avg · {s.max_deviation_pct}% max
                  </span>
                </div>
              ))}
          </div>
        </>
      )}
    </div>
  )
}