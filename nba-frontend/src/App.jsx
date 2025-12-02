import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const dateKey = (date) => date.toISOString().split('T')[0]

const friendlyDate = (date) =>
  date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })

const fetchMatchups = async (key) => {
  const url = `${API_BASE}/api/matchups?date=${encodeURIComponent(key)}`
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`)
  }
  const body = await res.json()
  const games = Array.isArray(body.games) ? body.games : []

  return games.map((g, idx) => ({
    id: g.id || `${key}-${idx}`,
    homeTeam: g.homeTeam,
    awayTeam: g.awayTeam,
    tipoff: g.tipoff || 'Time TBA',
    predictedWinner: g.predictedWinner,
    confidence: typeof g.confidence === 'number' ? g.confidence : null,
    modelSummary: g.modelSummary || g.summary || 'Model summary unavailable.',
    newsPoints: Array.isArray(g.newsPoints) ? g.newsPoints : [],
    gameDate: g.gameDate || key,
  }))
}

function App() {
  const [selectedDate, setSelectedDate] = useState(() => new Date())
  const [matchups, setMatchups] = useState([])
  const [selectedMatchup, setSelectedMatchup] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const key = useMemo(() => dateKey(selectedDate), [selectedDate])

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const data = await fetchMatchups(key)
        setMatchups(data)
        setSelectedMatchup(data[0] ?? null)
      } catch (err) {
        setError('Could not load matchups. Please retry.')
        setMatchups([])
        setSelectedMatchup(null)
      } finally {
        setLoading(false)
      }
    }

    load()
  }, [key])

  const handleDateChange = (evt) => {
    const next = evt.target.value ? new Date(`${evt.target.value}T00:00:00`) : new Date()
    setSelectedDate(next)
  }

  const shiftDate = (days) => {
    setSelectedDate((prev) => {
      const shifted = new Date(prev)
      shifted.setDate(prev.getDate() + days)
      return shifted
    })
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">NBA Prediction Studio</p>
          <h1>See model picks by matchup</h1>
          <p className="lede">Switch dates, browse games, and open a matchup to view the model winner and supporting news context.</p>
        </div>
        <div className="date-switcher">
          <button onClick={() => shiftDate(-1)} aria-label="Previous day">
            ← Prev
          </button>
          <div className="date-label">
            <span>{friendlyDate(selectedDate)}</span>
            <input
              type="date"
              value={key}
              onChange={handleDateChange}
              aria-label="Select date"
            />
          </div>
          <button onClick={() => shiftDate(1)} aria-label="Next day">
            Next →
          </button>
        </div>
      </header>

      <main className="layout">
        <section className="matchups">
          <div className="section-heading">
            <h2>Matchups</h2>
            <span className="pill">{loading ? 'Loading matchups…' : `${matchups.length} games`}</span>
          </div>

          {error && <div className="banner error">{error}</div>}
          {loading && <div className="banner">Loading matchups…</div>}
          {!loading && !error && !matchups.length && <div className="banner">No games scheduled for this date.</div>}

          <div className="matchup-grid">
            {matchups.map((game) => {
              const isActive = selectedMatchup?.id === game.id
              return (
                <button
                  key={game.id}
                  className={`matchup-card ${isActive ? 'active' : ''}`}
                  onClick={() => setSelectedMatchup(game)}
                >
                  <div className="teams">
                    <div className="team away">{game.awayTeam}</div>
                    <span className="at">@</span>
                    <div className="team home">{game.homeTeam}</div>
                  </div>
                  <div className="meta-row">
                    <span className="tipoff">{game.tipoff}</span>
                    <span className="pill subtle">Model: {game.predictedWinner}</span>
                  </div>
                  <div className="confidence">
                    Confidence: <strong>{Math.round(game.confidence * 100)}%</strong>
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        <section className="detail">
          {!selectedMatchup ? (
            <div className="empty">Select a matchup to see the model output.</div>
          ) : (
            <div className="detail-card">
              <div className="detail-header">
                <div>
                  <p className="eyebrow">Model pick</p>
                  <h3>
                    {selectedMatchup.awayTeam} @ {selectedMatchup.homeTeam}
                  </h3>
                  <p className="subtitle">{selectedMatchup.tipoff}</p>
                </div>
                <div className="pick-pill">
                  {selectedMatchup.predictedWinner}
                  <span className="confidence-chip">{Math.round(selectedMatchup.confidence * 100)}% conf.</span>
                </div>
              </div>

              <div className="model-summary">
                <p>{selectedMatchup.modelSummary}</p>
              </div>

              <div className="news-block">
                <h4>Evidence highlights</h4>
                <ul>
                  {selectedMatchup.newsPoints.map((point, idx) => (
                    <li key={idx}>{point}</li>
                  ))}
                </ul>
              </div>

              <div className="hint">Hook this pane up to your API to show live predictions and LangGraph news explanations.</div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
