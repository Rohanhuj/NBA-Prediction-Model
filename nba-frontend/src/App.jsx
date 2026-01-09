import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import './App.css'
import MatchupDetail from './MatchupDetail.jsx'
import MatchupsPage from './MatchupsPage.jsx'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MatchupsPage />} />
        <Route path="/game/:date/:gameId" element={<MatchupDetail />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
