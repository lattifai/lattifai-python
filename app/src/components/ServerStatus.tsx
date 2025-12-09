import React, { useState, useEffect, useCallback } from 'react';

interface ServerStatusProps {
    serverUrl: string;
    onServerUrlChange: (url: string) => void;
}

const ServerStatus: React.FC<ServerStatusProps> = ({ serverUrl, onServerUrlChange }) => {
    const [isOnline, setIsOnline] = useState<boolean | null>(null);
    const [checking, setChecking] = useState(false);
    const [showInstructions, setShowInstructions] = useState(false);

    const checkServerStatus = useCallback(async () => {
        setChecking(true);
        try {
            const url = serverUrl || 'http://localhost:8001';
            console.log('Checking server status at:', url);
            const response = await fetch(`${url}/health`, {
                method: 'GET',
                signal: AbortSignal.timeout(3000), // 3 second timeout
            });
            console.log('Server response:', response.ok);
            setIsOnline(response.ok);
        } catch (error) {
            console.log('Server check failed:', error);
            setIsOnline(false);
        } finally {
            setChecking(false);
        }
    }, [serverUrl]);

    useEffect(() => {
        checkServerStatus();
        // Check every 10 seconds
        const interval = setInterval(checkServerStatus, 10000);
        return () => clearInterval(interval);
    }, [checkServerStatus]);

    const copyToClipboard = (text: string) => {
        navigator.clipboard.writeText(text);
        alert('Command copied to clipboard!');
    };

    if (isOnline === null) {
        return null; // Still checking initially
    }

    if (isOnline) {
        return (
            <div style={{
                maxWidth: '900px',
                margin: '1rem auto',
                padding: '1.5rem',
                backgroundColor: 'rgba(76, 175, 80, 0.05)',
                border: '1px solid rgba(76, 175, 80, 0.3)',
                borderRadius: '12px'
            }}>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    marginBottom: '1.5rem',
                    paddingBottom: '1rem',
                    borderBottom: '1px solid rgba(76, 175, 80, 0.2)'
                }}>
                    <span style={{ fontSize: '1.5rem' }}>✅</span>
                    <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 600, color: '#4CAF50', fontSize: '1.05rem' }}>Backend Server Online</div>
                        <div style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.25rem' }}>
                            Connected to {serverUrl || 'http://localhost:8001'}
                        </div>
                    </div>
                    <button
                        type="button"
                        onClick={checkServerStatus}
                        disabled={checking}
                        style={{
                            padding: '0.5rem 1rem',
                            borderRadius: '4px',
                            border: '1px solid #4CAF50',
                            backgroundColor: 'white',
                            color: '#4CAF50',
                            cursor: checking ? 'not-allowed' : 'pointer',
                            fontSize: '0.85rem',
                            fontWeight: 600,
                            opacity: checking ? 0.6 : 1
                        }}
                    >
                        {checking ? 'Checking...' : 'Refresh'}
                    </button>
                </div>

                {/* Custom Backend Server URL Input */}
                <div>
                    <label style={{
                        fontWeight: 600,
                        fontSize: '0.9rem',
                        display: 'block',
                        marginBottom: '0.5rem',
                        color: '#333'
                    }}>
                        ⚙️ Custom Backend Server URL (Optional)
                    </label>
                    <input
                        type="text"
                        value={serverUrl}
                        onChange={e => onServerUrlChange(e.target.value)}
                        placeholder="Leave empty to use default (http://localhost:8001)"
                        style={{
                            width: '100%',
                            padding: '0.75rem',
                            borderRadius: '6px',
                            border: '1px solid #ddd',
                            backgroundColor: 'white',
                            fontSize: '0.9rem',
                            outline: 'none'
                        }}
                    />
                    <div style={{ fontSize: '0.8rem', color: '#666', marginTop: '0.5rem' }}>
                        Change this if your backend server is running on a different address
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div style={{
            maxWidth: '900px',
            margin: '1rem auto',
            padding: '1.5rem',
            backgroundColor: 'rgba(255, 152, 0, 0.1)',
            border: '2px solid rgba(255, 152, 0, 0.4)',
            borderRadius: '12px'
        }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                <span style={{ fontSize: '1.5rem' }}>⚠️</span>
                <div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem', color: '#FF9800' }}>
                        Backend Server Not Running
                    </div>
                    <div style={{ fontSize: '0.9rem', color: '#666', marginTop: '0.25rem' }}>
                        Please start the backend server to use the alignment service
                    </div>
                </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
                <button
                    type="button"
                    onClick={() => setShowInstructions(!showInstructions)}
                    style={{
                        padding: '0.75rem 1.5rem',
                        borderRadius: '6px',
                        border: 'none',
                        backgroundColor: '#FF9800',
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: '1rem',
                        fontWeight: 600,
                        boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                    }}
                >
                    {showInstructions ? '📖 Hide Instructions' : '🚀 Show How to Start'}
                </button>

                <button
                    type="button"
                    onClick={checkServerStatus}
                    disabled={checking}
                    style={{
                        padding: '0.75rem 1.5rem',
                        borderRadius: '6px',
                        border: '1px solid #FF9800',
                        backgroundColor: checking ? '#f5f5f5' : 'white',
                        color: '#FF9800',
                        cursor: checking ? 'not-allowed' : 'pointer',
                        fontSize: '1rem',
                        fontWeight: 600,
                        opacity: checking ? 0.6 : 1
                    }}
                >
                    {checking ? '🔄 Checking...' : '🔄 Check Again'}
                </button>
            </div>

            {showInstructions && (
                <div style={{
                    marginTop: '1.5rem',
                    padding: '1.5rem',
                    backgroundColor: 'rgba(255, 255, 255, 0.5)',
                    borderRadius: '8px',
                    border: '1px solid rgba(0, 0, 0, 0.1)'
                }}>
                    <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', fontWeight: 600 }}>
                        📋 Start Backend Server
                    </h3>

                    <div style={{ marginBottom: '1rem' }}>
                        <div style={{
                            fontSize: '0.9rem',
                            fontWeight: 600,
                            marginBottom: '0.5rem',
                            color: '#333'
                        }}>
                            Step 1: Open a new terminal window
                        </div>
                    </div>

                    <div style={{ marginBottom: '1rem' }}>
                        <div style={{
                            fontSize: '0.9rem',
                            fontWeight: 600,
                            marginBottom: '0.5rem',
                            color: '#333'
                        }}>
                            Step 2: Run this command:
                        </div>
                        <div style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.5rem',
                            backgroundColor: '#1e1e1e',
                            padding: '0.75rem',
                            borderRadius: '6px',
                            fontFamily: 'monospace',
                            fontSize: '0.95rem'
                        }}>
                            <code style={{ flex: 1, color: '#d4d4d4' }}>lai-server --port 8001</code>
                            <button
                                onClick={() => copyToClipboard('lai-server --port 8001')}
                                style={{
                                    padding: '0.4rem 0.8rem',
                                    borderRadius: '4px',
                                    border: 'none',
                                    backgroundColor: '#4CAF50',
                                    color: 'white',
                                    cursor: 'pointer',
                                    fontSize: '0.8rem',
                                    fontWeight: 600
                                }}
                            >
                                📋 Copy
                            </button>
                        </div>
                    </div>

                    <div style={{
                        padding: '0.75rem',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        borderLeft: '4px solid #2196F3',
                        borderRadius: '4px',
                        fontSize: '0.85rem',
                        color: '#1976D2'
                    }}>
                        <strong>💡 Tip:</strong> Keep the terminal window open while using the application.
                        The server will run at <code style={{
                            backgroundColor: 'rgba(0,0,0,0.1)',
                            padding: '0.2rem 0.4rem',
                            borderRadius: '3px',
                            fontFamily: 'monospace'
                        }}>http://localhost:8001</code>
                    </div>

                    <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#666' }}>
                        After starting the server, click "🔄 Check Again" above to verify the connection.
                    </div>
                </div>
            )}
        </div>
    );
};

export default ServerStatus;
