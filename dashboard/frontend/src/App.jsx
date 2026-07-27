import { useEffect, useRef, useState } from 'react'
import TrendView from './components/TrendView.jsx'
import RiskPanel from './components/RiskPanel.jsx'
import CorrelationPanel from './components/CorrelationPanel.jsx'
import RecommendationPanel from './components/RecommendationPanel.jsx'
import './App.css'

const START_INDEX = 700
const TICK_MS = 500
const SPEEDS = [1, 5, 20]

export default function App() {
  const [endIndex, setEndIndex] = useState(START_INDEX)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(5)
  const [jumpValue, setJumpValue] = useState(String(START_INDEX))
  const [outOfRange, setOutOfRange] = useState(false)

  const handleOutOfRange = () => {
    setIsPlaying(false)
    setOutOfRange(true)
  }

  useEffect(() => {
    if (!isPlaying) return undefined
    const id = setInterval(() => setEndIndex((i) => i + speed), TICK_MS)
    return () => clearInterval(id)
  }, [isPlaying, speed])

  const jump = (e) => {
    e.preventDefault()
    const parsed = parseInt(jumpValue, 10)
    if (Number.isFinite(parsed) && parsed >= 0) {
      setOutOfRange(false)
      setEndIndex(parsed)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-title">
          <h1>Grade Change Intelligence</h1>
          <p>Live basis-weight monitoring &amp; AI setpoint recommendations</p>
        </div>

        <div className="playback">
          <button
            className={`playback-toggle ${isPlaying ? 'is-playing' : ''}`}
            onClick={() => {
              setOutOfRange(false)
              setIsPlaying((p) => !p)
            }}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </button>

          <div className="speed-group">
            {SPEEDS.map((s) => (
              <button
                key={s}
                className={`speed-btn ${speed === s ? 'active' : ''}`}
                onClick={() => setSpeed(s)}
              >
                {s}x
              </button>
            ))}
          </div>

          <form className="jump-form" onSubmit={jump}>
            <label htmlFor="jump-index">Index</label>
            <input
              id="jump-index"
              value={jumpValue}
              onChange={(e) => setJumpValue(e.target.value)}
              inputMode="numeric"
            />
            <button type="submit">Go</button>
          </form>
        </div>
      </header>

      {outOfRange && (
        <div className="range-banner">Reached the end of the dataset at index {endIndex}.</div>
      )}

      <main className="app-grid">
        <section className="app-grid-main">
          <TrendView endIndex={endIndex} onOutOfRange={handleOutOfRange} />
          <RecommendationPanel />
        </section>

        <section className="app-grid-side">
          <RiskPanel endIndex={endIndex} onOutOfRange={handleOutOfRange} />
          <CorrelationPanel />
        </section>
      </main>
    </div>
  )
}