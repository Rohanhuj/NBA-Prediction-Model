import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { fetchMatchups, fetchPrediction } from './api.js'

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

const toTitleDate = (dateStr) => {
  const [year, month, day] = dateStr.split('-').map(Number)
  const date = new Date(year, month - 1, day)
  return date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })
}

const parseNewsPreview = (text) => {
  if (!text) return null
  const [rankingSection = '', summariesSection = ''] = text.split('Summaries:')
  const [summariesRaw = '', unifiedRaw = ''] = summariesSection.split('Unified Game Preview:')

  const evidenceMatch = unifiedRaw.match(/Evidence Summary\s*([\s\S]*?)Why this supports the pick/i)
  const whyMatch = unifiedRaw.match(/Why this supports the pick\s*([\s\S]*)/i)

  const evidenceLines = evidenceMatch
    ? evidenceMatch[1]
        .split(/\n+/)
        .map((line) => line.replace(/^[-•*]+\s*/, '').trim())
        .filter(Boolean)
    : []
  const whyText = whyMatch ? whyMatch[1].trim() : ''

  const rankingMatches = Array.from(
    rankingSection.matchAll(/\d+\.\s*\[\d+\]\s*Title:\s*(.*?)\s*\* Rating:\s*([^*]+)\* Reason:\s*([\s\S]*?)(?=\n\s*\d+\.|$)/gi)
  ).map((match) => ({
    title: match[1]?.trim(),
    rating: match[2]?.trim(),
    reason: match[3]?.trim().replace(/\*\*?/g, ''),
  }))

  const articles = summariesRaw
    .split(/\n\s*---\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      const title = block.match(/Title:\s*(.*)/i)?.[1]?.trim()
      const source = block.match(/Source:\s*(.*)/i)?.[1]?.trim()
      const url = block.match(/URL:\s*(.*)/i)?.[1]?.trim()
      const summaryMatch = block.match(/Summary:\s*([\s\S]*)/i)
      const summary = summaryMatch ? summaryMatch[1].trim() : block
      const ranking = rankingMatches.find((rank) =>
        rank.title && title
          ? rank.title.toLowerCase().includes(title.toLowerCase()) || title.toLowerCase().includes(rank.title.toLowerCase())
          : false
      )
      return { title, source, url, summary, ranking }
    })

  return {
    whyText,
    evidenceLines,
    articles,
  }
}

const parseArticleSummary = (summary) => {
  if (!summary) return []
  const cleaned = summary
    .replace(/^Here is the summary of the article:\s*/i, '')
    .replace(/^Here's the summary:\s*/i, '')
    .trim()
  const labels = [
    'Player Availability',
    'Team Momentum',
    'Betting Odds',
    'Injuries',
    'Analyst Picks',
    'Overall Sentiment',
  ]
  const labelPattern = labels.join('|')
  const sectionMatches = Array.from(
    cleaned.matchAll(
      new RegExp(
        `(?:\\*\\*|__)?(${labelPattern})(?:\\*\\*|__)?\\s*:\\s*([\\s\\S]*?)(?=(?:\\*\\*|__)?(?:${labelPattern})(?:\\*\\*|__)?\\s*:|$)`,
        'gi'
      )
    )
  )
  if (!sectionMatches.length) {
    return [{ title: 'Summary', items: [cleaned] }]
  }
  return sectionMatches.map((match) => {
    const title = match[1].trim()
    const body = match[2].trim()
    const items = body
      .split(/\n|\s\*\s|\s•\s/)
      .map((line) => line.replace(/^\*+\s*/, '').trim())
      .filter(Boolean)
    return { title, items }
  })
}

const buildCombinedSummary = (articles) => {
  const combined = new Map()
  articles.forEach((article) => {
    parseArticleSummary(article.summary).forEach((section) => {
      const key = section.title
      if (!combined.has(key)) {
        combined.set(key, [])
      }
      combined.get(key).push(...section.items)
    })
  })
  return Array.from(combined.entries()).map(([title, items]) => {
    const unique = Array.from(new Set(items.map((item) => item.trim()).filter(Boolean)))
    const limited = unique.slice(0, 3).map((item) => (item.length > 140 ? `${item.slice(0, 137)}…` : item))
    return { title, items: limited }
  })
}

function MatchupDetail() {
  const { date, gameId } = useParams()
  const [game, setGame] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      setError('')
      setGame(null)
      setPrediction(null)
      try {
        const games = await fetchMatchups(date)
        const found = games.find((item) => String(item.id) === String(gameId))
        if (!found) {
          if (active) {
            setError('Game not found for that date.')
            setLoading(false)
          }
          return
        }
        if (active) {
          setGame(found)
        }
        const result = await fetchPrediction({ gameId: Number(gameId), date })
        if (active) {
          setPrediction(result)
        }
      } catch (err) {
        if (active) {
          setError('Could not load prediction. Please retry.')
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    if (date && gameId) {
      load()
    }

    return () => {
      active = false
    }
  }, [date, gameId])

  const parsedPreview = prediction?.news_preview_text
    ? parseNewsPreview(prediction.news_preview_text)
    : null
  const combinedSummary = parsedPreview ? buildCombinedSummary(parsedPreview.articles.slice(0, 3)) : []

  return (
    <div className="page">

      <main className="layout layout-single">
        <section className="detail">
          {error && <div className="banner error">{error}</div>}
          {loading && <div className="banner">Loading prediction…</div>}
          {!loading && !error && !game && <div className="banner">Matchup not found.</div>}

          {game && (
            <div className="detail-card">
              <div className="detail-header">
                <div>
                  <p className="eyebrow">Featured matchup</p>
                  <h3 className="teams-large">
                    {game.awayTeam} @ {game.homeTeam}
                  </h3>
                  <p className="subtitle">{game.tipoff}</p>
                  <Link className="pill" to="/">← Back to matchups</Link>
                </div>
                <div className="pick-pill pick-pill-large">
                  {prediction?.predicted_winner && getTeamLogo(prediction.predicted_winner) ? (
                    <img
                      src={getTeamLogo(prediction.predicted_winner)}
                      alt={prediction.predicted_winner}
                    />
                  ) : (
                    <div className="team-logo-fallback" />
                  )}
                  <span className="pick-team">{prediction?.predicted_winner || '—'}</span>
                  <span className="confidence-chip">
                    {typeof prediction?.confidence === 'number'
                      ? `${Math.round(prediction.confidence * 100)}% conf.`
                      : 'Confidence TBD'}
                  </span>
                </div>
              </div>

              <div className="model-summary">
                {parsedPreview ? (
                  <>
                    <div className="summary-section">
                      <h4>Why this supports the pick</h4>
                      <p>{parsedPreview.whyText || 'No explanation available.'}</p>
                    </div>
                    <div className="summary-section">
                      <h4>Evidence summary</h4>
                      <ul>
                        {parsedPreview.evidenceLines.length ? (
                          parsedPreview.evidenceLines.map((line, idx) => <li key={idx}>{line}</li>)
                        ) : (
                          <li>No evidence summary available.</li>
                        )}
                      </ul>
                    </div>
                    <div className="summary-section">
                      <h4>Combined context</h4>
                      <div className="news-summary-grid combined-grid">
                        {combinedSummary.length ? (
                          combinedSummary.map((section, idx) => (
                            <div className="news-summary-section" key={idx}>
                              <h6>{section.title}</h6>
                              <ul>
                                {section.items.map((item, itemIdx) => (
                                  <li key={itemIdx}>{item}</li>
                                ))}
                              </ul>
                            </div>
                          ))
                        ) : (
                          <div className="news-summary-section">
                            <h6>Summary</h6>
                            <ul>
                              <li>No combined summary available.</li>
                            </ul>
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="summary-section">
                      <h4>Top supporting articles</h4>
                      <div className="news-grid">
                        {parsedPreview.articles.slice(0, 3).map((article, idx) => (
                          <div className="news-card" key={idx}>
                            <div className="news-card-header">
                              <h5>{article.title || `Article ${idx + 1}`}</h5>
                              {article.ranking?.rating && (
                                <span className="pill subtle">Rating: {article.ranking.rating}</span>
                              )}
                            </div>
                            {article.source && <p className="news-source">{article.source}</p>}
                            {article.ranking?.reason && (
                              <p className="news-reason">{article.ranking.reason}</p>
                            )}
                            {article.url && (
                              <a href={article.url} target="_blank" rel="noreferrer">
                                {article.url}
                              </a>
                            )}
                            <div className="news-summary">
                              <div className="news-summary-grid">
                                {parseArticleSummary(article.summary).map((section, sectionIdx) => (
                                  <div className="news-summary-section" key={sectionIdx}>
                                    <h6>{section.title}</h6>
                                    <ul>
                                      {section.items.map((item, itemIdx) => (
                                        <li key={itemIdx}>{item}</li>
                                      ))}
                                    </ul>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </>
                ) : (
                  <p>
                    {prediction?.news_preview_text ||
                      prediction?.error ||
                      (loading ? 'Generating model explanation…' : 'No explanation available.')}
                  </p>
                )}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default MatchupDetail
