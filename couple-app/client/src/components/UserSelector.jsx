import React from 'react'
import './UserSelector.css'

export default function UserSelector({ onSelect }) {
  return (
    <div className="user-selector">
      <div className="user-selector__inner">
        <div className="user-selector__hero">
          <div className="user-selector__hearts">💕</div>
          <h1 className="user-selector__title">우리만의 공간</h1>
          <p className="user-selector__subtitle">어떤 역할로 들어올까요?</p>
        </div>

        <div className="user-selector__buttons">
          <button className="user-selector__btn user-selector__btn--a" onClick={() => onSelect('userA')}>
            <span className="user-selector__btn-icon">🌸</span>
            <span className="user-selector__btn-label">나 (A)</span>
            <span className="user-selector__btn-hint">첫 번째 파트너</span>
          </button>

          <button className="user-selector__btn user-selector__btn--b" onClick={() => onSelect('userB')}>
            <span className="user-selector__btn-icon">🌹</span>
            <span className="user-selector__btn-label">파트너 (B)</span>
            <span className="user-selector__btn-hint">두 번째 파트너</span>
          </button>
        </div>

        <p className="user-selector__note">같은 서버를 쓰기 때문에<br />두 사람이 각각 선택하면 돼요 💌</p>
      </div>
    </div>
  )
}
