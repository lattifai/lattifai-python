import React from 'react';

interface Segment {
    start: number;
    end: number;
    text: string;
    speaker?: string;
}

interface ResultsDisplayProps {
    data: {
        status: string;
        segments: Segment[];
        caption_content: string;
        output_format: string;
        media_stem?: string;
    } | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ data }) => {
    const [format, setFormat] = React.useState('srt');
    const [content, setContent] = React.useState('');

    React.useEffect(() => {
        if (data) {
            // Default to the format returned by backend (usually srt)
            setFormat(data.output_format || 'srt');
            setContent(data.caption_content);
        }
    }, [data]);

    // Update content when format changes
    React.useEffect(() => {
        if (!data || !data.segments) return;

        let newContent = '';
        switch (format) {
            case 'srt':
                newContent = generateSRT(data.segments);
                break;
            case 'vtt':
                newContent = generateVTT(data.segments);
                break;
            case 'txt':
                newContent = data.segments.map(s => s.text).join('\n');
                break;
            case 'json':
                newContent = JSON.stringify(data.segments, null, 2);
                break;
            case 'sbv':
                newContent = generateSBV(data.segments);
                break;
            case 'ass':
                newContent = generateASS(data.segments);
                break;
            case 'ssa':
                newContent = generateSSA(data.segments);
                break;
            case 'lrc':
                newContent = generateLRC(data.segments);
                break;
            default:
                // Fallback to original content if format matches what backend sent, 
                // otherwise show warning or try best effort json
                if (format === data.output_format) {
                    newContent = data.caption_content;
                } else {
                    newContent = `Format conversion for ${format} is not supported in client-side mode yet.\n\nRaw Segments:\n` + JSON.stringify(data.segments, null, 2);
                }
        }
        setContent(newContent);
    }, [format, data]);

    if (!data) return null;

    return (
        <div className="results-display card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h2 style={{ margin: 0 }}>Alignment Results</h2>

                <div className="format-selector" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <label style={{ fontWeight: 600, fontSize: '0.9rem' }}>Format:</label>
                    <select
                        value={format}
                        onChange={e => setFormat(e.target.value)}
                        style={{
                            padding: '0.4rem',
                            borderRadius: '4px',
                            border: '1px solid var(--border-color)',
                            backgroundColor: 'var(--bg-color)',
                            color: 'var(--text-color)'
                        }}
                    >
                        <option value="srt">SRT</option>
                        <option value="vtt">VTT</option>
                        <option value="ass">ASS</option>
                        <option value="ssa">SSA</option>
                        <option value="lrc">LRC</option>
                        <option value="sbv">SBV (YouTube)</option>
                        <option value="txt">TXT</option>
                        <option value="json">JSON</option>
                        {/* Add legacy options if needed, but disable them for client-gen */}
                        {data.output_format !== 'srt' && !['vtt', 'sbv', 'txt', 'json'].includes(data.output_format) && (
                            <option value={data.output_format}>{data.output_format.toUpperCase()} (Original)</option>
                        )}
                    </select>
                </div>
            </div>

            <div className="download-section">
                <a
                    href={`data:text/plain;charset=utf-8,${encodeURIComponent(content)}`}
                    download={data.media_stem ? `${data.media_stem}_LattifAI.${format}` : `alignment.${format}`}
                    className="download-btn"
                >
                    Download {format.toUpperCase()}
                </a>
            </div>

            <div className="segments-list">
                <h3>Segments</h3>
                <ul>
                    {data.segments.map((seg, idx) => (
                        <li key={idx} className="segment-item">
                            <span className="timestamp">[{formatTimeSimple(seg.start)} - {formatTimeSimple(seg.end)}]</span>
                            {seg.speaker && <span className="speaker">{seg.speaker} </span>}
                            <span className="text">{seg.text}</span>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="raw-srt">
                <h3>{format.toUpperCase()} Output Preview</h3>
                <pre>{content}</pre>
            </div>
        </div>
    );
};

// --- Time Formatters ---

// Simple MM:SS.mmm for display
const formatTimeSimple = (seconds: number) => {
    const date = new Date(0);
    date.setMilliseconds(seconds * 1000);
    const mm = date.toISOString().substr(14, 5);
    const ms = date.toISOString().substr(20, 3);
    return `${mm}.${ms}`;
};

// SRT: HH:MM:SS,mmm
const formatTimeSRT = (seconds: number) => {
    const date = new Date(0);
    date.setMilliseconds(seconds * 1000);
    const hh = date.toISOString().substr(11, 2);
    const mm = date.toISOString().substr(14, 2);
    const ss = date.toISOString().substr(17, 2);
    const ms = date.toISOString().substr(20, 3);
    return `${hh}:${mm}:${ss},${ms}`;
};

// VTT: HH:MM:SS.mmm
const formatTimeVTT = (seconds: number) => {
    return formatTimeSRT(seconds).replace(',', '.');
};

// SBV: H:MM:SS.mmm (Time usually not zero-padded for hours if < 10, but standard ISO is fine)
const formatTimeSBV = (seconds: number) => {
    return formatTimeSRT(seconds).replace(',', '.');
};


// --- Generators ---

const generateSRT = (segments: Segment[]) => {
    return segments.map((seg, index) => {
        return `${index + 1}\n${formatTimeSRT(seg.start)} --> ${formatTimeSRT(seg.end)}\n${seg.text}\n`;
    }).join('\n');
};

const generateVTT = (segments: Segment[]) => {
    const body = segments.map((seg) => {
        return `${formatTimeVTT(seg.start)} --> ${formatTimeVTT(seg.end)}\n${seg.text}\n`;
    }).join('\n');
    return `WEBVTT\n\n${body}`;
};

const generateSBV = (segments: Segment[]) => {
    return segments.map((seg) => {
        return `${formatTimeSBV(seg.start)},${formatTimeSBV(seg.end)}\n${seg.text}\n`;
    }).join('\n');
};

const formatTimeASS = (seconds: number) => {
    const date = new Date(0);
    date.setMilliseconds(seconds * 1000);
    const h = date.toISOString().substr(11, 1); // 1 digit hour usually enough
    const mm = date.toISOString().substr(14, 2);
    const ss = date.toISOString().substr(17, 2);
    const cs = Math.floor(date.getMilliseconds() / 10).toString().padStart(2, '0');
    return `${h}:${mm}:${ss}.${cs}`;
};

const generateASS = (segments: Segment[]) => {
    const header = `[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
Timer: 100,0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
    const events = segments.map(seg => {
        return `Dialogue: 0,${formatTimeASS(seg.start)},${formatTimeASS(seg.end)},Default,,0,0,0,,${seg.text}`;
    }).join('\n');
    return header + events;
};

const generateSSA = (segments: Segment[]) => {
    const header = `[Script Info]
ScriptType: v4.00
Collisions: Normal
PlayDepth: 0
Timer: 100,0000

[V4 Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding
Style: Default,Arial,20,16777215,65535,0,0,0,0,1,2,2,2,10,10,10,0,1

[Events]
Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
`;
    const events = segments.map(seg => {
        return `Dialogue: Marked=0,${formatTimeASS(seg.start)},${formatTimeASS(seg.end)},Default,,0,0,0,,${seg.text}`;
    }).join('\n');
    return header + events;
};

const formatTimeLRC = (seconds: number) => {
    const date = new Date(0);
    date.setMilliseconds(seconds * 1000);
    const mm = date.toISOString().substr(14, 2);
    const ss = date.toISOString().substr(17, 2);
    const cs = Math.floor(date.getMilliseconds() / 10).toString().padStart(2, '0');
    return `${mm}:${ss}.${cs}`;
};

const generateLRC = (segments: Segment[]) => {
    return segments.map(seg => {
        return `[${formatTimeLRC(seg.start)}]${seg.text}`;
    }).join('\n');
};

export default ResultsDisplay;
