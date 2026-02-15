'use client'

import { useEffect } from 'react'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Unhandled error:', error)
  }, [error])

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      fontFamily: 'system-ui, sans-serif',
      color: '#e0e0e0',
      backgroundColor: '#0a0a0a',
      gap: '1rem',
    }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 600 }}>Something went wrong</h2>
      <p style={{ color: '#888', maxWidth: '400px', textAlign: 'center' }}>
        {error.message || 'An unexpected error occurred.'}
      </p>
      <button
        onClick={reset}
        style={{
          padding: '0.5rem 1.5rem',
          borderRadius: '6px',
          border: '1px solid #333',
          backgroundColor: '#1a1a1a',
          color: '#e0e0e0',
          cursor: 'pointer',
          fontSize: '0.9rem',
        }}
      >
        Try again
      </button>
      <button
        onClick={() => window.location.reload()}
        style={{
          padding: '0.5rem 1.5rem',
          borderRadius: '6px',
          border: 'none',
          backgroundColor: 'transparent',
          color: '#666',
          cursor: 'pointer',
          fontSize: '0.85rem',
        }}
      >
        Reload page
      </button>
    </div>
  )
}
