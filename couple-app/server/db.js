import Database from 'better-sqlite3'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
export const db = new Database(path.join(__dirname, 'database.sqlite'))

export function initDb() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS promises (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      created_by  TEXT    NOT NULL,
      content     TEXT    NOT NULL,
      is_kept     INTEGER NOT NULL DEFAULT 0,
      created_at  TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
      kept_at     TEXT
    );

    CREATE TABLE IF NOT EXISTS wishes (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      requested_by TEXT    NOT NULL,
      content      TEXT    NOT NULL,
      status       TEXT    NOT NULL DEFAULT 'pending',
      created_at   TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
      resolved_at  TEXT
    );
  `)
}
