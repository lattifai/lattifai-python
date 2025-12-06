import { useState } from 'react'
import './App.css'
import AlignmentForm from './components/AlignmentForm'
import ResultsDisplay from './components/ResultsDisplay'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>LattifAI Alignment</h1>
        <p>Text-Speech Forced Alignment Tool</p>
      </header>

      <main className="app-main">
        <AlignmentForm onResult={setResult} onLoading={setLoading} />

        {loading && <div className="loading-spinner">Processing alignment... Please wait.</div>}

        <ResultsDisplay data={result} />
      </main>
    </div>
  )
}

export default App
