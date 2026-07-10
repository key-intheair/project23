import React from 'react'
import './WishItem.css'

const STATUS_LABEL = { pending: '대기중', accepted: '수락됨', rejected: '거절됨' }

export default function WishItem({ wish, currentUser, onRespond }) {
  const isRequester = wish.requested_by === currentUser
  const canRespond = !isRequester && wish.status === 'pending'
  const displayOwner = isRequester ? '내 소원' : '파트너의 소원'

  const dateStr = new Date(wish.created_at + 'Z').toLocaleDateString('ko-KR', {
    month: 'long', day: 'numeric'
  })

  return (
    <div className={`wish-item wish-item--${wish.status}`}>
      <div className="wish-item__header">
        <span className="wish-item__owner">{displayOwner}</span>
        <span className={`wish-item__status wish-item__status--${wish.status}`}>
          {STATUS_LABEL[wish.status]}
        </span>
      </div>
      <p className="wish-item__content">{wish.content}</p>
      <div className="wish-item__footer">
        <span className="wish-item__date">{dateStr}</span>
        {canRespond && (
          <div className="wish-item__actions">
            <button
              className="btn btn-sm btn-outline wish-item__reject"
              onClick={() => onRespond(wish.id, 'rejected')}
            >
              거절
            </button>
            <button
              className="btn btn-sm btn-primary"
              onClick={() => onRespond(wish.id, 'accepted')}
            >
              수락 🌸
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
