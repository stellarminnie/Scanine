import { useEffect, useState } from 'react'
import './SplashScreen.css'

export default function SplashScreen({ onStart }) {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setReady(true), 100)
    return () => clearTimeout(t)
  }, [])

  return (
    <div className={`home ${ready ? 'ready' : ''}`}>

      <div className="home-header">
        <div className="home-title-row">
          <h1 className="home-title">Scanine</h1>
          <span className="status-pill">
            <span className="status-dot" />
            Offline ready
          </span>
        </div>
        <p className="home-location">Tacloban City</p>
      </div>

      <div className="home-hero">
        <div className="screening-card">
          <div className="card-glow" aria-hidden="true" />
          <span className="card-tag">New screening</span>
          <h2 className="card-title">Capture five frames to begin</h2>
          <p className="card-desc">
            Point the camera at the affected area. No connection needed.
          </p>
          <button className="btn-start" onClick={onStart}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
              <circle cx="12" cy="13" r="4"></circle>
            </svg>
            Start screening
          </button>
        </div>
      </div>

      <div className="home-footer">
        <p>Runs entirely on this device. No internet connection required.</p>
      </div>

    </div>
  )
}
