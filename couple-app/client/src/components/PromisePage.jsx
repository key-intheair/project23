import React, { useState } from 'react'
import { usePolling } from '../hooks/usePolling'
import { api } from '../api'
import PromiseItem from './PromiseItem'
import './PromisePage.css'

export default function PromisePage({ currentUser, showToast }) {
  const { data: promises, loading, refetch } = usePolling(() => api.getPromises())
  const [content, setContent] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!content.trim() || submitting) return
    setSubmitting(true)
    try {
      await api.addPromise(currentUser, content.trim())
      setContent('')
      await refetch()
      showToast('약속을 추가했어요 🤝')
    } catch (err) {
      showToast(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  async function handleKeep(id) {
    try {
      await api.keepPromise(id)
      await refetch()
      showToast('약속을 지켰어요! ✓')
    } catch (err) {
      showToast(err.message)
    }
  }

  const kept = promises?.filter(p => p.is_kept) ?? []
  const pending = promises?.filter(p => !p.is_kept) ?? []

  return (
    <div className="page-content">
      <div className="page-header">
        <h1>🤝 약속</h1>
        <span className="promise-stats">
          {kept.length}/{(promises?.length ?? 0)} 지킴
        </span>
      </div>

      <form className="add-form card" onSubmit={handleSubmit}>
        <textarea
          value={content}
          onChange={e => setContent(e.target.value)}
          placeholder="어떤 약속을 할까요? (예: 매주 금요일 데이트하기)"
          rows={3}
          maxLength={200}
        />
        <div className="add-form__footer">
          <span className="add-form__count">{content.length}/200</span>
          <button className="btn btn-primary btn-sm" type="submit" disabled={!content.trim() || submitting}>
            {submitting ? '...' : '약속 추가'}
          </button>
        </div>
      </form>

      {loading && !promises ? (
        <div className="loading-spinner">불러오는 중...</div>
      ) : promises?.length === 0 ? (
        <div className="empty-state">
          <div className="empty-icon">🤝</div>
          <p>아직 약속이 없어요<br />첫 번째 약속을 만들어보세요!</p>
        </div>
      ) : (
        <div className="promise-list">
          {pending.length > 0 && (
            <div className="promise-section">
              {pending.map(p => (
                <PromiseItem key={p.id} promise={p} currentUser={currentUser} onKeep={handleKeep} />
              ))}
            </div>
          )}
          {kept.length > 0 && (
            <div className="promise-section">
              <p className="promise-section__label">지킨 약속 ✓</p>
              {kept.map(p => (
                <PromiseItem key={p.id} promise={p} currentUser={currentUser} onKeep={handleKeep} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
