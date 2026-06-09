import React from 'react'
import './PromiseItem.css'

export default function PromiseItem({ promise, currentUser, onKeep }) {
  const isKept = promise.is_kept === 1
  const isMine = promise.created_by === currentUser
  const displayOwner = isMine ? '내가 한 약속' : '파트너의 약속'

  const dateStr = new Date(promise.created_at + 'Z').toLocaleDateString('ko-KR', {
    month: 'long', day: 'numeric'
  })

  return (
    <div className={`promise-item ${isKept ? 'promise-item--kept' : ''}`}>
      <div className="promise-item__check" onClick={() => !isKept && onKeep(promise.id)}>
        {isKept ? (
          <span className="promise-item__check-icon promise-item__check-icon--done">✓</span>
        ) : (
          <span className="promise-item__check-icon promise-item__check-icon--empty" />
        )}
      </div>
      <div className="promise-item__body">
        <p className="promise-item__content">{promise.content}</p>
        <div className="promise-item__meta">
          <span className="promise-item__owner">{displayOwner}</span>
          <span className="promise-item__date">{dateStr}</span>
        </div>
      </div>
      {isKept && (
        <span className="promise-item__badge">지킴 ✓</span>
      )}
    </div>
  )
}
