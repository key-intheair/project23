import express from 'express'
import cors from 'cors'
import path from 'path'
import { fileURLToPath } from 'url'
import { initDb } from './db.js'
import promisesRouter from './routes/promises.js'
import wishesRouter from './routes/wishes.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const isProd = process.env.NODE_ENV === 'production'

const app = express()

if (!isProd) {
  app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:4173'] }))
}

app.use(express.json())

initDb()

app.use('/api/promises', promisesRouter)
app.use('/api/wishes', wishesRouter)

if (isProd) {
  const clientDist = path.join(__dirname, '../client/dist')
  app.use(express.static(clientDist))
  app.get('*', (req, res) => res.sendFile(path.join(clientDist, 'index.html')))
}

const PORT = process.env.PORT || 3001
app.listen(PORT, () => console.log(`Server running on port ${PORT}`))
