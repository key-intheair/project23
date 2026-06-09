import React, { useState, useCallback } from 'react'
import UserSelector from './components/UserSelector'
import NavBar from './components/NavBar'
import PromisePage from './components/PromisePage'
import WishPage from './components/WishPage'
import RoseDisplay from './components/RoseDisplay'
import './styles/App.css'

function Toast({ message, onDone }) {
  return <div className="toast" onAnimationEnd={onDone}>{message}</div>
}

export default function App() {
  const [currentUser, setCurrentUser] = useState(() => localStorage.getItem('currentUser'))
  const [activeTab, setActiveTab] = useState('promises')
  const [toast, setToast] = useState(null)

  function handleSelect(user) {
    localStorage.setItem('currentUser', user)
    setCurrentUser(user)
  }

  function handleResetUser() {
    localStorage.removeItem('currentUser')
    setCurrentUser(null)
  }

  const showToast = useCallback((msg) => {
    setToast(msg)
    setTimeout(() => setToast(null), 2500)
  }, [])

  if (!currentUser) {
    return (
      <div className="app-container">
        <UserSelector onSelect={handleSelect} />
      </div>
    )
  }

  return (
    <div className="app-container">
      {activeTab === 'promises' && (
        <PromisePage currentUser={currentUser} showToast={showToast} />
      )}
      {activeTab === 'wishes' && (
        <WishPage currentUser={currentUser} showToast={showToast} />
      )}
      {activeTab === 'roses' && (
        <RoseDisplay />
      )}

      <NavBar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        currentUser={currentUser}
        onResetUser={handleResetUser}
      />

      {toast && <Toast message={toast} onDone={() => setToast(null)} />}
    </div>
  )
}
