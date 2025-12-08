import React, { useState, useCallback, useRef, useEffect } from 'react';
import axios from 'axios';
import { TRANSCRIPTION_MODELS } from '../constants/models';

interface ApiKeyStatus {
    exists: boolean;
    masked_value: string | null;
    create_url: string;
}

interface AlignmentFormProps {
    onResult: (data: any) => void;
    onLoading: (isLoading: boolean) => void;
    alignmentModel: string;
    geminiApiKey: ApiKeyStatus | null;
    serverUrl: string;
}

type InputMode = 'upload' | 'local' | 'youtube';

const AlignmentForm: React.FC<AlignmentFormProps> = ({ onResult, onLoading, alignmentModel, geminiApiKey, serverUrl }) => {
    const [mode, setMode] = useState<InputMode>('upload');

    // File Upload State
    const [mediaFile, setMediaFile] = useState<File | null>(null);
    const [captionText, setCaptionText] = useState('');
    const [captionExpanded, setCaptionExpanded] = useState(false);
    const [captionFilename, setCaptionFilename] = useState('');
    const [dragActive, setDragActive] = useState(false);

    // Local Path State
    const [localMediaPath, setLocalMediaPath] = useState('');
    const [localCaptionPath, setLocalCaptionPath] = useState('');
    const [localOutputDir, setLocalOutputDir] = useState('');

    // YouTube State
    const [youtubeUrl, setYoutubeUrl] = useState('https://www.youtube.com/watch?v=DQacCB9tDaw');
    const [youtubeOutputDir, setYoutubeOutputDir] = useState(() => {
        const today = new Date();
        const dateStr = today.toISOString().split('T')[0]; // YYYY-MM-DD
        return `~/Downloads/${dateStr}`;
    });

    // Options
    const [splitSentence, setSplitSentence] = useState(true);
    const [normalizeText, setNormalizeText] = useState(true);
    const outputFormat = 'srt'; // Fixed output format
    const [transcriptionModel, setTranscriptionModel] = useState('nvidia/parakeet-tdt-0.6b-v3');
    const [transcriptionExpanded, setTranscriptionExpanded] = useState(false);
    const [editingGeminiKey, setEditingGeminiKey] = useState(false);
    const [geminiKeyInput, setGeminiKeyInput] = useState('');
    const [savingGeminiKey, setSavingGeminiKey] = useState(false);
    const [saveGeminiToFile, setSaveGeminiToFile] = useState(true);

    const [loading, setLoading] = useState(false);
    const [selectingDir, setSelectingDir] = useState(false);
    const [logs, setLogs] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Browse Directory on Server
    const handleBrowsing = async () => {
        setSelectingDir(true);
        try {
            const baseUrl = serverUrl || '';
            const response = await fetch(`${baseUrl}/api/utils/select-directory`, {
                method: 'POST'
            });
            if (response.ok) {
                const data = await response.json();
                if (data.path) {
                    setLocalOutputDir(data.path);
                }
            }
        } catch (error) {
            console.error("Failed to browse directory:", error);
        } finally {
            setSelectingDir(false);
        }
    };

    // Save Gemini API Key
    const saveGeminiApiKey = async () => {
        setSavingGeminiKey(true);
        try {
            const response = await fetch('/api/keys', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    lattifai_key: '', // Don't change LattifAI key
                    gemini_key: geminiKeyInput,
                    save_to_file: saveGeminiToFile,
                }),
            });

            if (response.ok) {
                // Refresh the page or fetch new key status from parent
                setEditingGeminiKey(false);
                setGeminiKeyInput('');
                const result = await response.json();
                alert(result.message || 'Gemini API Key saved successfully!');
                // Reload to get updated key status
                window.location.reload();
            } else {
                const error = await response.json();
                alert(`Failed to save Gemini API key: ${error.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Failed to save Gemini API key:', error);
            alert('Failed to save Gemini API key. Please try again.');
        } finally {
            setSavingGeminiKey(false);
        }
    };

    // Format error message for web display
    const formatError = (errorText: string) => {
        // Remove ANSI color codes
        const cleanText = errorText.replace(/\x1b\[[0-9;]*m/g, '').replace(/\[[\d;]+m/g, '');

        // Try to parse structured error
        const errorMatch = cleanText.match(/\[(.*?)\]\s*(.*?)(?:\s+Context:\s*(.*))?$/);

        if (errorMatch) {
            const [, errorType, message, context] = errorMatch;
            return {
                type: errorType,
                message: message.trim(),
                context: context ? parseContext(context) : null
            };
        }

        return {
            type: 'Error',
            message: cleanText,
            context: null
        };
    };

    // Parse context string into key-value pairs
    const parseContext = (contextStr: string) => {
        const items: Record<string, string> = {};
        const parts = contextStr.split(', ');

        parts.forEach(part => {
            const [key, ...valueParts] = part.split('=');
            if (key && valueParts.length > 0) {
                items[key.trim()] = valueParts.join('=').trim();
            }
        });

        return items;
    };

    // Helpers
    const isMediaFile = (file: File) => file.type.startsWith('audio/') || file.type.startsWith('video/') || /\.(mp3|wav|mp4|mkv|mov|flac|m4a|avi|webm)$/i.test(file.name);

    const isCaptionFile = (file: File) => /\.(srt|vtt|ass|ssa|sub|sbv|txt)$/i.test(file.name);

    const handleFiles = useCallback((files: FileList | File[]) => {
        const fileArray = Array.from(files);
        fileArray.forEach(file => {
            if (isMediaFile(file)) {
                setMediaFile(file);
            } else if (isCaptionFile(file)) {
                // If it's a caption file, read its content
                const reader = new FileReader();
                reader.onload = (e) => {
                    const content = e.target?.result as string;
                    setCaptionText(content);
                    setCaptionFilename(file.name);
                    setCaptionExpanded(true); // Expand to show the loaded content
                };
                reader.readAsText(file);
            }
            // If it's neither media nor caption, ignore it
        });
    }, []);

    const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFiles(e.dataTransfer.files);
        }
    }, [handleFiles]);

    const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
    }, []);

    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [logs]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        setError(null);
        setLogs([]);
        setLoading(true);
        onLoading(true);
        onResult(null);

        const formData = new FormData();

        // Add media based on mode
        if (mode === 'upload') {
            if (!mediaFile) {
                setError('Please select a media file');
                setLoading(false);
                onLoading(false);
                return;
            }
            formData.append('media_file', mediaFile);

            // Add caption text if provided
            if (captionText.trim()) {
                const captionBlob = new Blob([captionText], { type: 'text/plain' });
                // Use original filename if available to preserve format extension
                const filename = captionFilename || 'pasted_caption.txt';
                formData.append('caption_file', captionBlob, filename);
            }
            if (localOutputDir) {
                formData.append('local_output_dir', localOutputDir);
            }
        } else if (mode === 'local') {
            if (!localMediaPath) {
                setError('Please enter a media file path');
                setLoading(false);
                onLoading(false);
                return;
            }
            formData.append('local_media_path', localMediaPath);
            if (localCaptionPath) {
                formData.append('local_caption_path', localCaptionPath);
            }
            if (localOutputDir) {
                formData.append('local_output_dir', localOutputDir);
            }
        } else if (mode === 'youtube') {
            if (!youtubeUrl) {
                setError('Please enter a YouTube URL');
                setLoading(false);
                onLoading(false);
                return;
            }
            formData.append('youtube_url', youtubeUrl);
            if (youtubeOutputDir.trim()) {
                formData.append('youtube_output_dir', youtubeOutputDir);
            }
        }

        // Add options
        formData.append('split_sentence', splitSentence.toString());
        formData.append('normalize_text', normalizeText.toString());
        formData.append('output_format', outputFormat);

        formData.append('transcription_model', transcriptionModel);
        formData.append('alignment_model', alignmentModel);

        try {
            const baseUrl = serverUrl || '';
            const response = await axios.post(`${baseUrl}/align`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            // Determine media stem for filename
            let mediaStem = 'alignment_result';
            if (mode === 'upload' && mediaFile) {
                mediaStem = mediaFile.name.replace(/\.[^/.]+$/, "");
            } else if (mode === 'local' && localMediaPath) {
                // Get filename from path
                const filename = localMediaPath.split(/[/\\]/).pop() || '';
                mediaStem = filename.replace(/\.[^/.]+$/, "");
            } else if (mode === 'youtube' && youtubeUrl) {
                // Try to get video ID or use generic name
                try {
                    const urlObj = new URL(youtubeUrl);
                    const v = urlObj.searchParams.get('v');
                    if (v) mediaStem = `youtube_${v}`;
                    else mediaStem = 'youtube_video';
                } catch (e) {
                    mediaStem = 'youtube_video';
                }
            }

            onResult({
                ...response.data,
                media_stem: mediaStem
            });
            addLog('✅ Alignment completed successfully!');
        } catch (err: any) {
            const errorMsg = err.response?.data?.error || err.message || 'Unknown error occurred';
            setError(errorMsg);
            addLog(`❌ Error: ${errorMsg}`);
            onResult(null);
        } finally {
            setLoading(false);
            onLoading(false);
        }
    };

    const addLog = (message: string) => {
        setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] ${message}`]);
    };

    const canSubmit = () => {
        if (mode === 'upload') return !!mediaFile;
        if (mode === 'local') return !!localMediaPath;
        if (mode === 'youtube') return !!youtubeUrl;
        return false;
    };

    return (
        <div className="card alignment-card">
            <div className="tabs">
                <button
                    className={`tab-btn ${mode === 'upload' ? 'active' : ''}`}
                    onClick={() => setMode('upload')}
                >
                    File Upload
                </button>
                <button
                    className={`tab-btn ${mode === 'youtube' ? 'active' : ''}`}
                    onClick={() => setMode('youtube')}
                >
                    YouTube URL
                </button>
                <button
                    className={`tab-btn ${mode === 'local' ? 'active' : ''}`}
                    onClick={() => setMode('local')}
                >
                    Server-Side Path
                </button>
            </div>



            <form onSubmit={handleSubmit}>
                {mode === 'upload' && (
                    <div className="upload-section">
                        <div
                            className={`unified-drop-zone ${dragActive ? 'active' : ''}`}
                            onDragEnter={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onDragOver={handleDragOver}
                            onDrop={handleDrop}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                multiple
                                style={{ display: 'none' }}
                                onChange={e => e.target.files && handleFiles(e.target.files)}
                            />

                            {/* Icon */}
                            <div className="upload-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
                                </svg>
                            </div>

                            <div className="drop-text-main">
                                Drop Media & Caption files here
                            </div>
                            <div className="drop-text-sub">
                                or click to browse (supports multiple files)
                            </div>
                        </div>

                        <div className="file-list">
                            {mediaFile && (
                                <div className="file-item">
                                    <div className="file-type-icon">M</div>
                                    <div className="file-info">
                                        <div className="file-name">{mediaFile.name}</div>
                                        <div className="file-meta">{(mediaFile.size / 1024 / 1024).toFixed(2)} MB</div>
                                    </div>
                                    <button type="button" className="remove-file-btn" onClick={() => setMediaFile(null)}>
                                        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* Caption Input Section - Collapsible */}
                        <div style={{ marginTop: '1rem', marginBottom: '1rem', backgroundColor: 'var(--card-bg)', borderRadius: '8px', border: '1px solid var(--border-color)', overflow: 'hidden' }}>
                            <div
                                onClick={() => setCaptionExpanded(!captionExpanded)}
                                style={{
                                    width: '100%',
                                    padding: '0.75rem 1rem',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'space-between',
                                    backgroundColor: 'transparent',
                                    border: 'none',
                                    cursor: 'pointer',
                                    color: 'var(--text-color)',
                                    userSelect: 'none'
                                }}
                            >
                                <span style={{ fontWeight: 700, fontSize: '1rem', display: 'flex', alignItems: 'center' }}>
                                    Caption (Paste Text)
                                    {captionFilename ? (
                                        <span style={{ marginLeft: '0.5rem', color: 'var(--primary-color)', fontSize: '0.85rem', display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                                            <span>✓ File: {captionFilename}</span>
                                            <button
                                                type="button"
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    setCaptionText('');
                                                    setCaptionFilename('');
                                                }}
                                                style={{
                                                    background: 'none',
                                                    border: 'none',
                                                    color: 'var(--text-secondary)',
                                                    cursor: 'pointer',
                                                    padding: '4px',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    borderRadius: '50%',
                                                }}
                                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.1)'}
                                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                                                title="Remove caption file"
                                            >
                                                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                                            </button>
                                        </span>
                                    ) : (
                                        captionText.trim() && <span style={{ marginLeft: '0.5rem', color: 'var(--primary-color)', fontSize: '0.85rem' }}>✓ Has content</span>
                                    )}
                                </span>
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="20"
                                    height="20"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="2"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    style={{
                                        transition: 'transform 0.2s ease',
                                        transform: captionExpanded ? 'rotate(180deg)' : 'rotate(0deg)'
                                    }}
                                >
                                    <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                            </div>
                            {captionExpanded && (
                                <div style={{ padding: '0 1rem 1rem 1rem' }}>
                                    <textarea
                                        value={captionText}
                                        onChange={e => !captionFilename && setCaptionText(e.target.value)}
                                        readOnly={!!captionFilename}
                                        placeholder={captionFilename ? `Content from ${captionFilename}` : "Paste your caption text here...\n\nExample:\nHello, welcome to this video.\nToday we will learn about AI."}
                                        style={{
                                            width: '100%',
                                            minHeight: '120px',
                                            padding: '0.75rem',
                                            borderRadius: '8px',
                                            border: '1px solid var(--border-color)',
                                            backgroundColor: captionFilename ? 'rgba(128, 128, 128, 0.05)' : 'var(--bg-color)',
                                            color: captionFilename ? 'var(--text-secondary)' : 'var(--text-color)',
                                            fontSize: '0.95rem',
                                            lineHeight: '1.5',
                                            resize: 'vertical',
                                            fontFamily: 'inherit',
                                            boxSizing: 'border-box',
                                            cursor: captionFilename ? 'not-allowed' : 'text',
                                            opacity: captionFilename ? 0.9 : 1
                                        }}
                                    />
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {mode === 'local' && (
                    <div className="local-section">
                        <div className="path-input-group">
                            <label>Media File Absolute Path</label>
                            <input
                                type="text"
                                value={localMediaPath}
                                onChange={e => setLocalMediaPath(e.target.value)}
                                placeholder="/home/user/media.mp4"
                            />
                        </div>
                        <div className="path-input-group">
                            <label>Caption File Absolute Path (Optional)</label>
                            <input
                                type="text"
                                value={localCaptionPath}
                                onChange={e => setLocalCaptionPath(e.target.value)}
                                placeholder="/home/user/caption.srt"
                            />
                        </div>
                        <div className="path-input-group">
                            <label>Output Directory (Optional)</label>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <input
                                    type="text"
                                    value={localOutputDir}
                                    onChange={e => setLocalOutputDir(e.target.value)}
                                    placeholder="/home/user/output/"
                                    style={{ flex: 1 }}
                                />
                                <button
                                    type="button"
                                    onClick={handleBrowsing}
                                    disabled={selectingDir}
                                    style={{
                                        padding: '0 1rem',
                                        backgroundColor: 'var(--card-bg)',
                                        border: '1px solid var(--border-color)',
                                        borderRadius: '4px',
                                        cursor: 'pointer',
                                        color: 'var(--text-color)',
                                        whiteSpace: 'nowrap'
                                    }}
                                >
                                    {selectingDir ? '...' : '📁 Browse'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {mode === 'youtube' && (
                    <div className="youtube-section">
                        <div className="path-input-group">
                            <label>YouTube Video URL</label>
                            <input
                                type="text"
                                value={youtubeUrl}
                                onChange={e => setYoutubeUrl(e.target.value)}
                                placeholder="https://www.youtube.com/watch?v=..."
                            />
                        </div>
                        <div className="path-input-group">
                            <label>Output Directory</label>
                            <input
                                type="text"
                                value={youtubeOutputDir}
                                onChange={e => setYoutubeOutputDir(e.target.value)}
                                placeholder="~/Downloads/YYYY-MM-DD"
                            />
                            <div style={{
                                marginTop: '0.5rem',
                                padding: '0.75rem',
                                backgroundColor: 'rgba(33, 150, 243, 0.1)',
                                border: '1px solid rgba(33, 150, 243, 0.3)',
                                borderRadius: '6px',
                                fontSize: '0.85rem',
                                lineHeight: '1.5',
                                color: 'var(--text-color)'
                            }}>
                                <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem' }}>
                                    <span style={{ fontSize: '1.1rem', flexShrink: 0 }}>💡</span>
                                    <div>
                                        <strong>Caption Strategy:</strong>
                                        <br />
                                        The system will first attempt to download YouTube's native captions. If unavailable, it will automatically transcribe the audio using the selected transcription model.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                <div className="options-grid">
                    <label className="checkbox-card">
                        <input type="checkbox" checked={splitSentence} onChange={e => setSplitSentence(e.target.checked)} />
                        <span>Split Sentence</span>
                    </label>
                    <label className="checkbox-card">
                        <input type="checkbox" checked={normalizeText} onChange={e => setNormalizeText(e.target.checked)} />
                        <span>Normalize Text</span>
                    </label>
                </div>

                {/* Transcription Model Selector - Collapsible */}
                <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'var(--card-bg)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <div
                        onClick={() => setTranscriptionExpanded(!transcriptionExpanded)}
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            cursor: 'pointer',
                            marginBottom: transcriptionExpanded ? '0.75rem' : '0'
                        }}
                    >
                        <div>
                            <label style={{ fontWeight: 700, fontSize: '1rem', display: 'block', marginBottom: '0.25rem', cursor: 'pointer' }}>Transcription Model</label>
                            <div style={{
                                fontSize: '0.85rem',
                                color: '#666',
                                lineHeight: '1.4'
                            }}>
                                💡 Auto-triggered when no caption file provided
                            </div>
                        </div>
                        <div style={{
                            fontSize: '1.5rem',
                            transition: 'transform 0.3s ease',
                            transform: transcriptionExpanded ? 'rotate(180deg)' : 'rotate(0deg)'
                        }}>
                            ▼
                        </div>
                    </div>

                    {transcriptionExpanded && (
                        <div className="path-input-group" style={{ marginTop: '1rem' }}>

                        {/* Grid layout: Model selector and Gemini API Key input side by side (when Gemini selected) */}
                        <div style={{ display: 'grid', gridTemplateColumns: transcriptionModel.includes('gemini') ? '3fr 2fr' : '1fr', gap: '1rem', alignItems: 'start' }}>
                            {/* Left: Model Selector */}
                            <div>
                                <select
                                    value={transcriptionModel}
                                    onChange={e => setTranscriptionModel(e.target.value)}
                                    style={{
                                        width: '100%',
                                        padding: '0.5rem',
                                        borderRadius: '4px',
                                        border: '1px solid var(--border-color)',
                                        backgroundColor: 'var(--bg-color)',
                                        color: 'var(--text-color)',
                                        fontSize: '0.95rem',
                                        marginBottom: '0.5rem'
                                    }}
                                >
                                    {TRANSCRIPTION_MODELS.map(model => (
                                        <option key={model.value} value={model.value}>
                                            {model.label}
                                        </option>
                                    ))}
                                </select>
                                <div style={{
                                    fontSize: '0.9rem',
                                    color: 'var(--text-secondary)',
                                    marginTop: '0.5rem',
                                    lineHeight: '1.4'
                                }}>
                                    {TRANSCRIPTION_MODELS.find(m => m.value === transcriptionModel)?.languages}
                                </div>
                            </div>

                            {/* Right: Gemini API Key Status/Input - Only show when Gemini model is selected */}
                            {transcriptionModel.includes('gemini') && geminiApiKey && (
                                <div>
                                    {geminiApiKey.exists && !editingGeminiKey ? (
                                        <div style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.5rem',
                                            backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                            padding: '0.5rem 0.75rem',
                                            borderRadius: '6px',
                                            fontSize: '0.85rem',
                                            border: '1px solid rgba(76, 175, 80, 0.3)'
                                        }}>
                                            <span style={{ color: '#4CAF50' }}>✓</span>
                                            <span style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-secondary)', flex: 1 }}>{geminiApiKey.masked_value}</span>
                                            <button
                                                type="button"
                                                onClick={() => {
                                                    setEditingGeminiKey(true);
                                                    setGeminiKeyInput('');
                                                }}
                                                style={{
                                                    padding: '0.3rem 0.6rem',
                                                    fontSize: '0.75rem',
                                                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                                                    color: '#4CAF50',
                                                    border: '1px solid rgba(76, 175, 80, 0.3)',
                                                    borderRadius: '4px',
                                                    cursor: 'pointer'
                                                }}
                                            >
                                                Edit
                                            </button>
                                        </div>
                                    ) : (
                                        <div style={{
                                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                                            padding: '0.75rem',
                                            borderRadius: '6px',
                                            border: '1px solid rgba(244, 67, 54, 0.3)'
                                        }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                                <span style={{ color: '#F44336', fontSize: '0.9rem' }}>⚠</span>
                                                <span style={{ color: 'var(--text-color)', fontSize: '0.8rem', fontWeight: 600 }}>Gemini API Key Required</span>
                                            </div>
                                            <input
                                                type="text"
                                                value={geminiKeyInput}
                                                onChange={e => setGeminiKeyInput(e.target.value)}
                                                placeholder="AIza..."
                                                style={{
                                                    width: '100%',
                                                    padding: '0.4rem',
                                                    marginBottom: '0.5rem',
                                                    borderRadius: '4px',
                                                    border: '1px solid rgba(244, 67, 54, 0.3)',
                                                    backgroundColor: 'var(--bg-color)',
                                                    color: 'var(--text-color)',
                                                    fontSize: '0.75rem',
                                                    fontFamily: 'monospace'
                                                }}
                                            />
                                            <label style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '0.5rem',
                                                marginBottom: '0.5rem',
                                                color: 'var(--text-color)',
                                                fontSize: '0.7rem',
                                                cursor: 'pointer'
                                            }}>
                                                <input
                                                    type="checkbox"
                                                    checked={saveGeminiToFile}
                                                    onChange={e => setSaveGeminiToFile(e.target.checked)}
                                                    style={{ cursor: 'pointer' }}
                                                />
                                                <span>Save to .env</span>
                                            </label>
                                            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                                                <button
                                                    type="button"
                                                    onClick={saveGeminiApiKey}
                                                    disabled={!geminiKeyInput.trim() || savingGeminiKey}
                                                    style={{
                                                        flex: 1,
                                                        minWidth: '60px',
                                                        padding: '0.4rem',
                                                        backgroundColor: geminiKeyInput.trim() ? '#4CAF50' : 'rgba(200, 200, 200, 0.3)',
                                                        color: 'white',
                                                        border: 'none',
                                                        borderRadius: '4px',
                                                        cursor: geminiKeyInput.trim() ? 'pointer' : 'not-allowed',
                                                        fontSize: '0.7rem',
                                                        fontWeight: 600
                                                    }}
                                                >
                                                    {savingGeminiKey ? 'Saving...' : 'Save'}
                                                </button>
                                                <a
                                                    href={geminiApiKey.create_url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    style={{
                                                        flex: 1,
                                                        minWidth: '60px',
                                                        padding: '0.4rem',
                                                        backgroundColor: 'rgba(244, 67, 54, 0.1)',
                                                        color: '#F44336',
                                                        border: '1px solid rgba(244, 67, 54, 0.3)',
                                                        borderRadius: '4px',
                                                        textAlign: 'center',
                                                        textDecoration: 'none',
                                                        fontSize: '0.7rem',
                                                        fontWeight: 600
                                                    }}
                                                >
                                                    Get Key
                                                </a>
                                                {editingGeminiKey && (
                                                    <button
                                                        type="button"
                                                        onClick={() => {
                                                            setEditingGeminiKey(false);
                                                            setGeminiKeyInput('');
                                                        }}
                                                        style={{
                                                            padding: '0.4rem',
                                                            backgroundColor: 'rgba(200, 200, 200, 0.1)',
                                                            color: 'var(--text-color)',
                                                            border: '1px solid var(--border-color)',
                                                            borderRadius: '4px',
                                                            cursor: 'pointer',
                                                            fontSize: '0.7rem'
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
                    )}
                </div>


                {error && (() => {
                    const formattedError = formatError(error);
                    return (
                        <div style={{
                            marginTop: '1rem',
                            padding: '1rem',
                            backgroundColor: 'rgba(244, 67, 54, 0.1)',
                            border: '1px solid rgba(244, 67, 54, 0.3)',
                            borderRadius: '8px',
                            color: 'var(--text-color)'
                        }}>
                            <div style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.5rem',
                                marginBottom: '0.75rem',
                                paddingBottom: '0.75rem',
                                borderBottom: '1px solid rgba(244, 67, 54, 0.2)'
                            }}>
                                <span style={{ fontSize: '1.5rem' }}>🚨</span>
                                <div>
                                    <div style={{
                                        fontWeight: 700,
                                        fontSize: '1rem',
                                        color: '#F44336'
                                    }}>
                                        {formattedError.type}
                                    </div>
                                    <div style={{
                                        fontSize: '0.9rem',
                                        marginTop: '0.25rem',
                                        lineHeight: '1.5'
                                    }}>
                                        {formattedError.message}
                                    </div>
                                </div>
                            </div>

                            {formattedError.context && Object.keys(formattedError.context).length > 0 && (
                                <div>
                                    <div style={{
                                        fontWeight: 600,
                                        fontSize: '0.85rem',
                                        marginBottom: '0.5rem',
                                        color: '#F57C00'
                                    }}>
                                        📋 Context Details:
                                    </div>
                                    <div style={{
                                        backgroundColor: 'rgba(0, 0, 0, 0.05)',
                                        padding: '0.75rem',
                                        borderRadius: '4px',
                                        fontFamily: 'monospace',
                                        fontSize: '0.8rem'
                                    }}>
                                        {Object.entries(formattedError.context).map(([key, value]) => (
                                            <div key={key} style={{
                                                marginBottom: '0.25rem',
                                                display: 'grid',
                                                gridTemplateColumns: 'auto 1fr',
                                                gap: '0.75rem'
                                            }}>
                                                <span style={{ color: '#1976D2', fontWeight: 600 }}>{key}:</span>
                                                <span style={{ wordBreak: 'break-all' }}>{value}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    );
                })()}

                {/* Server Logs Display */}
                {loading && logs.length > 0 && (
                    <div style={{
                        marginTop: '1rem',
                        marginBottom: '1rem',
                        padding: '1rem',
                        backgroundColor: '#1e1e1e',
                        borderRadius: '8px',
                        border: '1px solid var(--border-color)',
                        maxHeight: '300px',
                        overflowY: 'auto',
                        fontFamily: 'monospace',
                        fontSize: '0.85rem',
                    }}>
                        <div style={{ color: '#888', marginBottom: '0.5rem', fontWeight: 600 }}>
                            📋 Server Logs:
                        </div>
                        {logs.map((log, index) => (
                            <div key={index} style={{
                                color: log.includes('Error') || log.includes('error') ? '#ff6b6b' :
                                    log.includes('Warning') || log.includes('warning') ? '#ffd93d' :
                                        log.includes('✅') || log.includes('Success') ? '#51cf66' :
                                            log.includes('🔄') || log.includes('Processing') ? '#4dabf7' :
                                                '#e0e0e0',
                                padding: '0.25rem 0',
                                lineHeight: '1.4',
                            }}>
                                {log}
                            </div>
                        ))}
                        <div ref={logsEndRef} />
                    </div>
                )}

                <button
                    type="submit"
                    className="submit-btn"
                    disabled={!canSubmit() || loading}
                >
                    {loading ? 'Processing...' : 'Start Alignment'}
                </button>
            </form>
        </div>
    );
};

export default AlignmentForm;
