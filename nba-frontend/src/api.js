const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

export const fetchMatchups = async (key) => {
  const url = `${API_BASE}/api/matchups?date=${encodeURIComponent(key)}`
  const res = await fetch(url)
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`)
  }
  const body = await res.json()
  const games = Array.isArray(body.games) ? body.games : []

  return games.map((g, idx) => ({
    id: g.id || `${key}-${idx}`,
    homeTeam: g.home_team,
    awayTeam: g.away_team,
    homeAbbr: g.home_abbr,
    awayAbbr: g.away_abbr,
    tipoff: g.tipoff || 'Time TBA',
    predictedWinner: null,
    confidence: null,
    modelSummary: 'Select a matchup to generate the model pick.',
    newsPoints: [],
    gameDate: g.game_date || key,
  }))
}

export const fetchPrediction = async ({ gameId, date }) => {
  const url = `${API_BASE}/api/predict`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ game_id: gameId, date }),
  })
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`)
  }
  return res.json()
}
