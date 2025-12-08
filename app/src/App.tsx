import { useState, useEffect } from 'react'
import './App.css'
import AlignmentForm from './components/AlignmentForm'
import ResultsDisplay from './components/ResultsDisplay'
import ServerStatus from './components/ServerStatus'
import { ALIGNMENT_MODELS } from './constants/models'

interface ApiKeyStatus {
  exists: boolean;
  masked_value: string | null;
  create_url: string;
}

interface ApiKeysResponse {
  lattifai: ApiKeyStatus;
  gemini: ApiKeyStatus;
}

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [alignmentModel, setAlignmentModel] = useState('Lattifai/Lattice-1')
  const [apiKeys, setApiKeys] = useState<ApiKeysResponse | null>(null)
  const [editingKeys, setEditingKeys] = useState(false)
  const [lattifaiKeyInput, setLattifaiKeyInput] = useState('')
  const [savingKeys, setSavingKeys] = useState(false)
  const [saveToFile, setSaveToFile] = useState(true) // Default to save to file
  const [serverUrl, setServerUrl] = useState('')

  // Fetch API key status on mount
  useEffect(() => {
    fetchApiKeys()
  }, [])

  const fetchApiKeys = async () => {
    try {
      const response = await fetch('/api/keys')
      if (response.ok) {
        const data = await response.json()
        setApiKeys(data)
      }
    } catch (error) {
      console.error('Failed to fetch API keys:', error)
    }
  }

  const saveApiKeys = async () => {
    setSavingKeys(true)
    try {
      const response = await fetch('/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          lattifai_key: lattifaiKeyInput,
          gemini_key: '', // For now, only handle LattifAI key
          save_to_file: saveToFile, // Send user's choice
        }),
      })

      if (response.ok) {
        await fetchApiKeys() // Refresh key status
        setEditingKeys(false)
        setLattifaiKeyInput('')
        const result = await response.json()
        alert(result.message || 'API Key saved successfully!')
      } else {
        const error = await response.json()
        alert(`Failed to save API key: ${error.error || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Failed to save API keys:', error)
      alert('Failed to save API key. Please try again.')
    } finally {
      setSavingKeys(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>LattifAI Alignment</h1>
        <p>Text-Speech Forced Alignment Tool</p>
      </header>

      {/* Backend Server Status */}
      <ServerStatus serverUrl={serverUrl} onServerUrlChange={setServerUrl} />

      {/* Alignment Model Selector - Futuristic glassmorphism design */}
      <div style={{
        maxWidth: '900px',
        margin: '2rem auto 0',
        padding: '1.5rem',
        background: 'linear-gradient(135deg, rgba(74, 144, 226, 0.9) 0%, rgba(80, 201, 195, 0.9) 100%)',
        backdropFilter: 'blur(10px)',
        borderRadius: '16px',
        border: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 8px 32px 0 rgba(74, 144, 226, 0.4), inset 0 1px 0 0 rgba(255, 255, 255, 0.3)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Glow effect */}
        <div style={{
          position: 'absolute',
          top: '-50%',
          right: '-20%',
          width: '200px',
          height: '200px',
          background: 'radial-gradient(circle, rgba(255, 255, 255, 0.3) 0%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none'
        }} />

        <div style={{ position: 'relative', zIndex: 1 }}>
          {/* Title */}
          <label style={{
            fontWeight: 700,
            fontSize: '1.05rem',
            color: 'white',
            textShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
            letterSpacing: '0.5px',
            display: 'block',
            marginBottom: '0.75rem'
          }}>
            🎯 Alignment Model
          </label>

          {/* Grid layout: Model selector and API Key input side by side */}
          <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '1rem', alignItems: 'start' }}>
            {/* Left: Model Selector */}
            <div>
              <select
                value={alignmentModel}
                onChange={e => setAlignmentModel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.75rem',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.4)',
                  backgroundColor: 'rgba(255, 255, 255, 0.95)',
                  fontSize: '0.95rem',
                  marginBottom: '0.75rem',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
                  outline: 'none'
                }}
              >
                {ALIGNMENT_MODELS.map(model => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
              <div style={{
                fontSize: '0.9rem',
                color: 'rgba(255, 255, 255, 0.95)',
                lineHeight: '1.5',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                padding: '0.5rem 0.75rem',
                borderRadius: '8px',
                backdropFilter: 'blur(5px)'
              }}>
                ✨ {ALIGNMENT_MODELS.find(m => m.value === alignmentModel)?.languages}
              </div>
            </div>

            {/* Right: API Key Status/Input */}
            {apiKeys && (
              <div>
                {apiKeys.lattifai.exists && !editingKeys ? (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    backgroundColor: 'rgba(255, 255, 255, 0.2)',
                    padding: '0.75rem',
                    borderRadius: '8px',
                    fontSize: '0.85rem',
                    color: 'white'
                  }}>
                    <span style={{ color: '#90EE90' }}>✓</span>
                    <span style={{ fontFamily: 'monospace', letterSpacing: '0.5px', flex: 1 }}>{apiKeys.lattifai.masked_value}</span>
                    <button
                      onClick={() => {
                        setEditingKeys(true)
                        setLattifaiKeyInput('')
                      }}
                      style={{
                        padding: '0.35rem 0.75rem',
                        fontSize: '0.8rem',
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        color: 'white',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Edit
                    </button>
                  </div>
                ) : (
                  <div style={{
                    backgroundColor: 'rgba(255, 100, 100, 0.3)',
                    padding: '0.75rem',
                    borderRadius: '8px',
                    border: '1px solid rgba(255, 255, 255, 0.3)'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <span style={{ color: '#FFB6C1', fontSize: '1rem' }}>⚠</span>
                      <span style={{ color: 'white', fontSize: '0.85rem', fontWeight: 600 }}>LattifAI API Key Required</span>
                    </div>
                    <input
                      type="text"
                      value={lattifaiKeyInput}
                      onChange={e => setLattifaiKeyInput(e.target.value)}
                      placeholder="lf_..."
                      style={{
                        width: '100%',
                        padding: '0.5rem',
                        marginBottom: '0.5rem',
                        borderRadius: '4px',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        fontSize: '0.8rem',
                        fontFamily: 'monospace'
                      }}
                    />
                    <label style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem',
                      marginBottom: '0.5rem',
                      color: 'white',
                      fontSize: '0.75rem',
                      cursor: 'pointer'
                    }}>
                      <input
                        type="checkbox"
                        checked={saveToFile}
                        onChange={e => setSaveToFile(e.target.checked)}
                        style={{ cursor: 'pointer' }}
                      />
                      <span>Save to .env</span>
                    </label>
                    <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                      <button
                        onClick={saveApiKeys}
                        disabled={!lattifaiKeyInput.trim() || savingKeys}
                        style={{
                          flex: 1,
                          minWidth: '80px',
                          padding: '0.5rem',
                          backgroundColor: lattifaiKeyInput.trim() ? '#4CAF50' : 'rgba(255, 255, 255, 0.3)',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: lattifaiKeyInput.trim() ? 'pointer' : 'not-allowed',
                          fontSize: '0.75rem',
                          fontWeight: 600
                        }}
                      >
                        {savingKeys ? 'Saving...' : 'Save'}
                      </button>
                      <a
                        href={apiKeys.lattifai.create_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{
                          flex: 1,
                          minWidth: '80px',
                          padding: '0.5rem',
                          backgroundColor: 'rgba(255, 255, 255, 0.2)',
                          color: 'white',
                          border: '1px solid rgba(255, 255, 255, 0.3)',
                          borderRadius: '4px',
                          textAlign: 'center',
                          textDecoration: 'none',
                          fontSize: '0.75rem',
                          fontWeight: 600
                        }}
                      >
                        Get Key
                      </a>
                      {editingKeys && (
                        <button
                          onClick={() => {
                            setEditingKeys(false)
                            setLattifaiKeyInput('')
                          }}
                          style={{
                            padding: '0.5rem',
                            backgroundColor: 'rgba(255, 255, 255, 0.2)',
                            color: 'white',
                            border: '1px solid rgba(255, 255, 255, 0.3)',
                            borderRadius: '4px',
                            cursor: 'pointer',
                            fontSize: '0.75rem'
                          }}
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <main className="app-main">
        <AlignmentForm
          onResult={setResult}
          onLoading={setLoading}
          alignmentModel={alignmentModel}
          geminiApiKey={apiKeys?.gemini ?? null}
          serverUrl={serverUrl}
        />

        {loading && <div className="loading-spinner">Processing alignment... Please wait.</div>}

        <ResultsDisplay data={result} />
      </main>
    </div>
  )
}

export default App
