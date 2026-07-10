import { useState, useEffect, useCallback, useRef } from 'react'

export function usePolling(fetchFn, intervalMs = 5000) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const fetchFnRef = useRef(fetchFn)
  fetchFnRef.current = fetchFn

  const refetch = useCallback(async () => {
    try {
      const result = await fetchFnRef.current()
      setData(result)
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refetch()
    const id = setInterval(refetch, intervalMs)
    return () => clearInterval(id)
  }, [refetch, intervalMs])

  return { data, loading, error, refetch }
}
