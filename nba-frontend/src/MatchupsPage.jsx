import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchMatchups } from './api.js'

const dateKey = (date) => date.toISOString().split('T')[0]

const friendlyDate = (date) =>
  date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  })

const monthLabel = (date) =>
  date.toLocaleDateString('en-US', {
    month: 'long',
    year: 'numeric',
  })

const yearOptions = Array.from({ length: 15 }, (_, idx) => 2015 + idx)

const TEAM_LOGOS = {
  ATL: 1610612737,
  BOS: 1610612738,
  BKN: 1610612751,
  CHA: 1610612766,
  CHI: 1610612741,
  CLE: 1610612739,
  DAL: 1610612742,
  DEN: 1610612743,
  DET: 1610612765,
  GSW: 1610612744,
  HOU: 1610612745,
  IND: 1610612754,
  LAC: 1610612746,
  LAL: 1610612747,
  MEM: 1610612763,
  MIA: 1610612748,
  MIL: 1610612749,
  MIN: 1610612750,
  NOP: 1610612740,
  NYK: 1610612752,
  OKC: 1610612760,
  ORL: 1610612753,
  PHI: 1610612755,
  PHX: 1610612756,
  POR: 1610612757,
  SAC: 1610612758,
  SAS: 1610612759,
  TOR: 1610612761,
  UTA: 1610612762,
  WAS: 1610612764,
}

const getTeamLogo = (abbr) =>
  abbr && TEAM_LOGOS[abbr]
    ? `https://cdn.nba.com/logos/nba/${TEAM_LOGOS[abbr]}/primary/L/logo.svg`
    : null

const DEFAULT_DATE = new Date('2024-02-14T00:00:00')

function MatchupsPage() {
  const [selectedDate, setSelectedDate] = useState(() => DEFAULT_DATE)
  const [viewDate, setViewDate] = useState(() => DEFAULT_DATE)
  const [matchups, setMatchups] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const navigate = useNavigate()
  const lastChangeRef = useRef(0)

  const key = useMemo(() => dateKey(selectedDate), [selectedDate])
  const viewKey = useMemo(() => dateKey(viewDate), [viewDate])

  useEffect(() => {
    setViewDate(new Date(selectedDate))
  }, [selectedDate])

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      setError('')
      try {
        const data = await fetchMatchups(key)
        setMatchups(data)
      } catch (err) {
        setError('Could not load matchups. Please retry.')
        setMatchups([])
      } finally {
        setLoading(false)
      }
    }

    load()
  }, [key])

  const canChangeDate = () => {
    const now = Date.now()
    if (loading || now - lastChangeRef.current < 500) {
      return false
    }
    lastChangeRef.current = now
    return true
  }

  const shiftMonth = (delta) => {
    setViewDate((prev) => {
      const shifted = new Date(prev)
      shifted.setMonth(prev.getMonth() + delta)
      return shifted
    })
  }

  const handleYearChange = (evt) => {
    const nextYear = Number(evt.target.value)
    setViewDate((prev) => new Date(nextYear, prev.getMonth(), 1))
  }

  const handleDaySelect = (day) => {
    if (!canChangeDate()) return
    const next = new Date(viewDate)
    next.setDate(day)
    setSelectedDate(next)
  }

  const daysInMonth = new Date(viewDate.getFullYear(), viewDate.getMonth() + 1, 0).getDate()
  const startWeekday = new Date(viewDate.getFullYear(), viewDate.getMonth(), 1).getDay()
  const blanks = Array.from({ length: startWeekday }, (_, idx) => idx)
  const days = Array.from({ length: daysInMonth }, (_, idx) => idx + 1)

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">NBA Prediction Studio</p>
          <h1>See model picks by matchup</h1>
          <p className="lede">Switch dates, browse games, and open a matchup to view the model winner and supporting news context.</p>
        </div>
        <div className="date-summary">
          <span>{friendlyDate(selectedDate)}</span>
        </div>
      </header>

      <main className="matchups-layout">
        <section className="calendar-panel">
          <div className="calendar-header">
            <button onClick={() => shiftMonth(-1)} aria-label="Previous month">
              ‹
            </button>
            <div>
              <h3>{monthLabel(viewDate)}</h3>
              <select value={viewDate.getFullYear()} onChange={handleYearChange} aria-label="Select year">
                {yearOptions.map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
            </div>
            <button onClick={() => shiftMonth(1)} aria-label="Next month">
              ›
            </button>
          </div>
          <div className="calendar-grid">
            {['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'].map((day) => (
              <div key={day} className="calendar-day-label">
                {day}
              </div>
            ))}
            {blanks.map((blank) => (
              <div key={`blank-${blank}`} className="calendar-day blank" />
            ))}
            {days.map((day) => {
              const isSelected = viewDate.getFullYear() === selectedDate.getFullYear() &&
                viewDate.getMonth() === selectedDate.getMonth() &&
                day === selectedDate.getDate()
              return (
                <button
                  key={day}
                  className={`calendar-day ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleDaySelect(day)}
                  disabled={loading}
                >
                  {day}
                </button>
              )
            })}
          </div>
        </section>

        <section className="schedule-panel">
          <div className="section-heading">
            <div>
              <h2>Games Schedule</h2>
              <p>{friendlyDate(selectedDate)}</p>
            </div>
            <span className="pill">{loading ? 'Loading…' : `${matchups.length} games`}</span>
          </div>

          {error && <div className="banner error">{error}</div>}
          {loading && <div className="banner">Loading matchups…</div>}
          {!loading && !error && !matchups.length && <div className="banner">No games scheduled for this date.</div>}

          <div className="schedule-list">
            {matchups.map((game) => (
              <button
                key={game.id}
                className="schedule-card"
                onClick={() => navigate(`/game/${game.gameDate}/${game.id}`)}
              >
                <div className="schedule-meta">
                  <span>{game.tipoff}</span>
                </div>
                <div className="schedule-teams">
                  <div className="schedule-team">
                    {getTeamLogo(game.awayAbbr) ? (
                      <img src={getTeamLogo(game.awayAbbr)} alt={game.awayTeam} />
                    ) : (
                      <div className="team-logo-fallback" />
                    )}
                    <div>
                      <p>{game.awayTeam}</p>
                      <span>{game.awayAbbr}</span>
                    </div>
                  </div>
                  <div className="versus">vs</div>
                  <div className="schedule-team">
                    {getTeamLogo(game.homeAbbr) ? (
                      <img src={getTeamLogo(game.homeAbbr)} alt={game.homeTeam} />
                    ) : (
                      <div className="team-logo-fallback" />
                    )}
                    <div>
                      <p>{game.homeTeam}</p>
                      <span>{game.homeAbbr}</span>
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

export default MatchupsPage
