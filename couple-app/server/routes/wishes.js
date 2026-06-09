import { Router } from 'express'
import { db } from '../db.js'

const router = Router()

router.get('/roses', (req, res) => {
  const { c } = db.prepare("SELECT COUNT(*) as c FROM wishes WHERE status = 'accepted'").get()
  const petals = c
  const fullRoses = Math.floor(petals / 10)
  const remainingPetals = petals % 10
  res.json({ petals, fullRoses, remainingPetals })
})

router.get('/', (req, res) => {
  const rows = db.prepare('SELECT * FROM wishes ORDER BY created_at DESC').all()
  res.json(rows)
})

router.post('/', (req, res) => {
  const { requested_by, content } = req.body
  if (!['userA', 'userB'].includes(requested_by)) {
    return res.status(400).json({ error: '올바른 사용자가 아니에요' })
  }
  if (!content || content.trim().length === 0) {
    return res.status(400).json({ error: '소원 내용을 입력해주세요' })
  }
  if (content.length > 200) {
    return res.status(400).json({ error: '소원은 200자 이내로 입력해주세요' })
  }
  const stmt = db.prepare('INSERT INTO wishes (requested_by, content) VALUES (?, ?)')
  const result = stmt.run(requested_by, content.trim())
  const created = db.prepare('SELECT * FROM wishes WHERE id = ?').get(result.lastInsertRowid)
  res.status(201).json(created)
})

router.patch('/:id/respond', (req, res) => {
  const { id } = req.params
  const { status } = req.body
  if (!['accepted', 'rejected'].includes(status)) {
    return res.status(400).json({ error: '올바른 응답이 아니에요' })
  }
  const wish = db.prepare('SELECT * FROM wishes WHERE id = ?').get(id)
  if (!wish) return res.status(404).json({ error: '소원을 찾을 수 없어요' })
  if (wish.status !== 'pending') {
    return res.status(400).json({ error: '이미 처리된 소원이에요' })
  }
  db.prepare("UPDATE wishes SET status = ?, resolved_at = datetime('now', 'localtime') WHERE id = ?").run(status, id)
  const updated = db.prepare('SELECT * FROM wishes WHERE id = ?').get(id)
  res.json(updated)
})

export default router
