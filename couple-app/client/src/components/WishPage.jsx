import React, { useState } from 'react'
import { usePolling } from '../hooks/usePolling'
import { api } from '../api'
import WishItem from './WishItem'
import './WishPage.css'

export default function WishPage({ currentUser, showToast }) {
  const { data: wishes, loading, refetch } = usePolling(() => api.getWishes())
  const [content, setContent] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!content.trim() || submitting) return
    setSubmitting(true)
    try {
      await api.addWish(currentUser, content.trim())
      setContent('')
      await refetch()
      showToast('소원을 올렸어요 ⭐')
    } catch (err) {
      showToast(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  async function handleRespond(id, status) {
    try {
      await api.respondWish(id, status)
      await refetch()
      showToast(status === 'accepted' ? '소원을 수락했어요 🌸' : '소원을 거절했어요')
    } catch (err) {
      showToast(err.message)
    }
  }

  const pending = wishes?.filter(w => w.status === 'pending') ?? []
  const resolved = wishes?.filter(w => w.status !== 'pending') ?? []

  return (
    <div className="page-content">
      <div className="page-header">
        <h1>⭐ 소원</h1>
        <span className="wish-stats">
          수락 {wishes?.filter(w => w.status === 'accepted').length ?? 0}개
        </span>
      </div>

      <form className="add-form card" onSubmit={handleSubmit}>
        <textarea
          value={content}
          onChange={e => setContent(e.target.value)}
          placeholder="파트너에게 소원을 빌어보세요! (예: 오늘 저녁 같이 요리하기)"
          rows={3}
          maxLength={200}
        />
        <div className="add-form__footer">
          <span className="add-form__count">{content.length}/200</span>
          <button className="btn btn-primary btn-sm" type="submit" disabled={!content.trim() || submitting}>
            {submitting ? '...' : '소원 올리기 ⭐'}
          </button>
        </div>
      </form>

      {loading && !wishes ? (
        <div className="loading-spinner">불러오는 중...</div>
      ) : wishes?.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">⭐</div>
          <p>아직 소원이 없어요<br />파트너에게 소원을 빌어보세요!</p>
        </div>
      ) : (
        <div className="wish-list">
          {pending.length > 0 && (
            <div className="wish-section">
              <p className="wish-section__label">대기 중인 소원</p>
              {pending.map(w => (
                <WishItem key={w.id} wish={w} currentUser={currentUser} onRespond={handleRespond} />
              ))}
            </div>
          )}
          {resolved.length > 0 && (
            <div className="wish-section">
              <p className="wish-section__label">처리된 소원</p>
              {resolved.map(w => (
                <WishItem key={w.id} wish={w} currentUser={currentUser} onRespond={handleRespond} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
