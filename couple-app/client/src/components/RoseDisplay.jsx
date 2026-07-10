import React, { useEffect, useRef, useState } from 'react'
import { usePolling } from '../hooks/usePolling'
import { api } from '../api'
import './RoseDisplay.css'

function RoseSVG({ size = 48 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Stem */}
      <line x1="24" y1="40" x2="24" y2="46" stroke="#5A9A5A" strokeWidth="2" strokeLinecap="round"/>
      <line x1="24" y1="43" x2="20" y2="47" stroke="#5A9A5A" strokeWidth="1.5" strokeLinecap="round"/>
      {/* Leaves */}
      <ellipse cx="18" cy="36" rx="5" ry="3" fill="#6DB86D" transform="rotate(-30 18 36)"/>
      <ellipse cx="30" cy="38" rx="5" ry="3" fill="#6DB86D" transform="rotate(30 30 38)"/>
      {/* Outer petals */}
      <ellipse cx="24" cy="18" rx="8" ry="10" fill="#E8637A" transform="rotate(0 24 24)"/>
      <ellipse cx="24" cy="18" rx="8" ry="10" fill="#E8637A" transform="rotate(72 24 24)"/>
      <ellipse cx="24" cy="18" rx="8" ry="10" fill="#D4527A" transform="rotate(144 24 24)"/>
      <ellipse cx="24" cy="18" rx="8" ry="10" fill="#E8637A" transform="rotate(216 24 24)"/>
      <ellipse cx="24" cy="18" rx="8" ry="10" fill="#D4527A" transform="rotate(288 24 24)"/>
      {/* Inner petals */}
      <ellipse cx="24" cy="20" rx="5" ry="6" fill="#F28BA0" transform="rotate(36 24 24)"/>
      <ellipse cx="24" cy="20" rx="5" ry="6" fill="#EF7A93" transform="rotate(108 24 24)"/>
      <ellipse cx="24" cy="20" rx="5" ry="6" fill="#F28BA0" transform="rotate(180 24 24)"/>
      <ellipse cx="24" cy="20" rx="5" ry="6" fill="#EF7A93" transform="rotate(252 24 24)"/>
      <ellipse cx="24" cy="20" rx="5" ry="6" fill="#F28BA0" transform="rotate(324 24 24)"/>
      {/* Center */}
      <circle cx="24" cy="24" r="5" fill="#C94A62"/>
      <circle cx="24" cy="24" r="3" fill="#E8637A"/>
    </svg>
  )
}

function PetalSVG({ filled, animating }) {
  return (
    <div className={`petal-slot ${filled ? 'petal-slot--filled' : ''} ${animating ? 'petal-slot--new' : ''}`}>
      <svg width="28" height="32" viewBox="0 0 28 32" fill="none" xmlns="http://www.w3.org/2000/svg">
        {filled ? (
          <ellipse cx="14" cy="16" rx="10" ry="14"
            fill="#F28BA0" stroke="#E8637A" strokeWidth="1"
            transform="rotate(0 14 16)"
          />
        ) : (
          <ellipse cx="14" cy="16" rx="10" ry="14"
            fill="transparent" stroke="#F9D6DC" strokeWidth="2"
            transform="rotate(0 14 16)"
          />
        )}
      </svg>
    </div>
  )
}

export default function RoseDisplay() {
  const { data: roses, loading } = usePolling(() => api.getRoses(), 5000)
  const prevPetalsRef = useRef(null)
  const [newPetalIndex, setNewPetalIndex] = useState(null)

  useEffect(() => {
    if (!roses) return
    const prev = prevPetalsRef.current
    if (prev !== null && roses.petals > prev) {
      const idx = (roses.petals - 1) % 10
      setNewPetalIndex(idx)
      const timer = setTimeout(() => setNewPetalIndex(null), 600)
      prevPetalsRef.current = roses.petals
      return () => clearTimeout(timer)
    }
    prevPetalsRef.current = roses.petals
  }, [roses])

  if (loading && !roses) {
    return <div className="loading-spinner">불러오는 중...</div>
  }

  const { petals = 0, fullRoses = 0, remainingPetals = 0 } = roses ?? {}

  return (
    <div className="page-content rose-page">
      <div className="page-header">
        <h1>🌹 우리의 장미</h1>
      </div>

      <div className="rose-summary card">
        <div className="rose-summary__text">
          {petals === 0 ? (
            <p className="rose-summary__empty">소원을 들어주면<br />꽃잎이 쌓여요 🌸</p>
          ) : (
            <>
              <p className="rose-summary__count">
                {fullRoses > 0 && <span>{fullRoses}송이</span>}
                {fullRoses > 0 && remainingPetals > 0 && <span className="rose-summary__plus"> + </span>}
                {remainingPetals > 0 && <span>{remainingPetals}꽃잎</span>}
              </p>
              <p className="rose-summary__total">총 {petals}개의 소원을 들어줬어요 💕</p>
            </>
          )}
        </div>
        <div className="rose-summary__equation">
          <span className="rose-eq-item">
            <span className="rose-eq-icon">⭐</span>
            <span className="rose-eq-label">수락 1개</span>
          </span>
          <span className="rose-eq-arrow">=</span>
          <span className="rose-eq-item">
            <PetalSVG filled />
            <span className="rose-eq-label">꽃잎 1장</span>
          </span>
          <span className="rose-eq-arrow">×10</span>
          <span className="rose-eq-item">
            <RoseSVG size={32} />
            <span className="rose-eq-label">장미 1송이</span>
          </span>
        </div>
      </div>

      {fullRoses > 0 && (
        <div className="full-roses card">
          <p className="full-roses__label">완성된 장미</p>
          <div className="full-roses__row">
            {Array.from({ length: fullRoses }).map((_, i) => (
              <div key={i} className="full-rose-icon">
                <RoseSVG size={52} />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="petal-garden card">
        <p className="petal-garden__label">
          {remainingPetals > 0
            ? `다음 장미까지 ${10 - remainingPetals}장 남았어요`
            : petals === 0
              ? '첫 번째 꽃잎을 모아보세요'
              : '새 장미를 시작해요!'}
        </p>
        <div className="petal-row">
          {Array.from({ length: 10 }).map((_, i) => (
            <PetalSVG
              key={i}
              filled={i < remainingPetals}
              animating={i === newPetalIndex}
            />
          ))}
        </div>
        <div className="petal-progress">
          <div
            className="petal-progress__bar"
            style={{ width: `${(remainingPetals / 10) * 100}%` }}
          />
        </div>
        <p className="petal-count">{remainingPetals} / 10 꽃잎</p>
      </div>

      {petals === 0 && (
        <div className="empty-state">
          <div className="empty-icon">🌹</div>
          <p>소원 탭에서 소원을 수락하면<br />꽃잎이 쌓이기 시작해요</p>
        </div>
      )}
    </div>
  )
}
