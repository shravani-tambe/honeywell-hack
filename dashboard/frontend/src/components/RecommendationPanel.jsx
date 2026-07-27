import { useEffect, useState } from 'react'
import { getAccuracy, getRecommendations, postDecision } from '../api.js'
import { sourceColor, sourceLabel } from '../sourceColors.js'
import './RecommendationPanel.css'

export default function RecommendationPanel() {
  const [suggestions, setSuggestions] = useState(null)
  const [accuracy, setAccuracy] = useState(null)
  const [pendingId, setPendingId] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getRecommendations('pending').then(setSuggestions).catch((err) => setError(err.message))
    getAccuracy().then(setAccuracy).catch(() => {})
  }, [])

  const decide = async (id, decision) => {
    setPendingId(id)
    try {
      await postDecision(id, decision)
      setSuggestions((prev) => prev.filter((s) => s.suggestion_id !== id))
    } catch (err) {
      setError(err.message)
    } finally {
      setPendingId(null)
    }
  }

  return (
    <div className="panel recommendation-panel">
      <div className="rec-header">
        <div>
          <h2 className="panel-title">Setpoint Recommendations</h2>
          <p className="panel-subtitle">Pending suggestions for process adjustment</p>
        </div>
        {accuracy && accuracy.accuracy !== null && (
          <div className="accuracy-badge">
            <span className="accuracy-value">{(accuracy.accuracy * 100).toFixed(1)}%</span>
            <span className="accuracy-label">In-spec rate · {accuracy.n_evaluated} evaluated</span>
          </div>
        )}
      </div>

      {error && <div className="error-banner">{error}</div>}

      {suggestions && suggestions.length === 0 && (
        <div className="empty-state">No pending suggestions right now.</div>
      )}

      {suggestions && suggestions.length > 0 && (
        <div className="rec-list scroll-list">
          {suggestions.map((s) => (
            <div className="rec-card" key={s.suggestion_id}>
              <div className="rec-card-top">
                <span className="rec-variable">{s.variable.replace('_', ' ')}</span>
                <span className="tag" style={{ color: sourceColor(s.source), border: 'none', backgroundColor: sourceColor(s.source) === 'var(--honeywell-red-muted)' ? '#FDF2F2' : '#F0F0F0' }}>
                  {sourceLabel(s.source)}
                </span>
              </div>

              <div className="rec-values">
                <span>{s.current_value}</span>
                <span className="rec-arrow">
                  {s.recommended_value > s.current_value ? '→' : '→'}
                </span>
                <span className="rec-target">{s.recommended_value}</span>
                <span className="rec-reduction">
                  −{(s.predicted_risk_reduction * 100).toFixed(1)}% risk
                </span>
              </div>

              <p className="rec-rationale">{s.rationale}</p>

              <div className="rec-actions">
                <button
                  className="rec-btn accept"
                  disabled={pendingId === s.suggestion_id}
                  onClick={() => decide(s.suggestion_id, 'accepted')}
                >
                  Accept
                </button>
                <button
                  className="rec-btn reject"
                  disabled={pendingId === s.suggestion_id}
                  onClick={() => decide(s.suggestion_id, 'rejected')}
                >
                  Reject
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}