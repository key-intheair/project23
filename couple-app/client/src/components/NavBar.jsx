import React from 'react'
import './NavBar.css'

const tabs = [
  { id: 'promises', label: '약속', icon: '🤝' },
  { id: 'wishes',   label: '소원', icon: '⭐' },
  { id: 'roses',    label: '장미', icon: '🌹' },
]

export default function NavBar({ activeTab, onTabChange, currentUser, onResetUser }) {
  const displayName = currentUser === 'userA' ? 'A' : 'B'

  return (
    <nav className="navbar">
      <div className="navbar__tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`navbar__tab ${activeTab === tab.id ? 'navbar__tab--active' : ''}`}
            onClick={() => onTabChange(tab.id)}
          >
            <span className="navbar__tab-icon">{tab.icon}</span>
            <span className="navbar__tab-label">{tab.label}</span>
          </button>
        ))}
      </div>
      <button className="navbar__user" onClick={onResetUser} title="사용자 변경">
        <span className="navbar__user-badge">{displayName}</span>
      </button>
    </nav>
  )
}
