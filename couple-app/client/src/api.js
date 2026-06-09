const BASE = '/api'

async function request(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || '오류가 발생했어요')
  return data
}

export const api = {
  getPromises: () => request(`${BASE}/promises`),
  addPromise: (created_by, content) =>
    request(`${BASE}/promises`, { method: 'POST', body: JSON.stringify({ created_by, content }) }),
  keepPromise: (id) =>
    request(`${BASE}/promises/${id}/keep`, { method: 'PATCH', body: JSON.stringify({}) }),

  getWishes: () => request(`${BASE}/wishes`),
  addWish: (requested_by, content) =>
    request(`${BASE}/wishes`, { method: 'POST', body: JSON.stringify({ requested_by, content }) }),
  respondWish: (id, status) =>
    request(`${BASE}/wishes/${id}/respond`, { method: 'PATCH', body: JSON.stringify({ status }) }),
  getRoses: () => request(`${BASE}/wishes/roses`),
}
