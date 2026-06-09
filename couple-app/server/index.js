import express from 'express'
import cors from 'cors'
import { initDb } from './db.js'
import promisesRouter from './routes/promises.js'
import wishesRouter from './routes/wishes.js'

const app = express()

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:4173'] }))
app.use(express.json())

initDb()

app.use('/api/promises', promisesRouter)
app.use('/api/wishes', wishesRouter)

const PORT = 3001
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`))
