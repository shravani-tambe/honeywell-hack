const API_BASE = 'http://localhost:8001/api'

async function request(path, options) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const getTrend = (endIndex) => request(`/trend?end_index=${endIndex}`)
export const getRisk = (endIndex) => request(`/risk?end_index=${endIndex}`)
export const getCorrelations = () => request('/correlations')
export const getRecommendations = (status = 'pending') => request(`/recommendations?status=${status}`)
export const getAccuracy = () => request('/accuracy')

export const postDecision = (suggestionId, decision) =>
  request(`/recommendations/${suggestionId}/decision`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision }),
  })