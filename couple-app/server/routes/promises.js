import { Router } from 'express'
import { db } from '../db.js'

const router = Router()

router.get('/', (req, res) => {
  const rows = db.prepare('SELECT * FROM promises ORDER BY created_at DESC').all()
  res.json(rows)
})

router.post('/', (req, res) => {
  const { created_by, content } = req.body
  if (!['userA', 'userB'].includes(created_by)) {
    return res.status(400).json({ error: '올바른 사용자가 아니에요' })
  }
  if (!content || content.trim().length === 0) {
    return res.status(400).json({ error: '약속 내용을 입력해주세요' })
  }
  if (content.length > 200) {
    return res.status(400).json({ error: '약속은 200자 이내로 입력해주세요' })
  }
  const stmt = db.prepare('INSERT INTO promises (created_by, content) VALUES (?, ?)')
  const result = stmt.run(created_by, content.trim())
  const created = db.prepare('SELECT * FROM promises WHERE id = ?').get(result.lastInsertRowid)
  res.status(201).json(created)
})

router.patch('/:id/keep', (req, res) => {
  const { id } = req.params
  const promise = db.prepare('SELECT * FROM promises WHERE id = ?').get(id)
  if (!promise) return res.status(404).json({ error: '약속을 찾을 수 없어요' })
  if (promise.is_kept) return res.json(promise)
  db.prepare("UPDATE promises SET is_kept = 1, kept_at = datetime('now', 'localtime') WHERE id = ?").run(id)
  const updated = db.prepare('SELECT * FROM promises WHERE id = ?').get(id)
  res.json(updated)
})

export default router
