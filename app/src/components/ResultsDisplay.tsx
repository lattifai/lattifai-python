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
    } | null;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ data }) => {
    if (!data) return null;

    return (
        <div className="results-display card">
            <h2>Alignment Results</h2>

            <div className="download-section">
                <a
                    href={`data:text/plain;charset=utf-8,${encodeURIComponent(data.caption_content)}`}
                    download={`alignment.${data.output_format}`}
                    className="download-btn"
                >
                    Download {data.output_format.toUpperCase()}
                </a>
            </div>

            <div className="segments-list">
                <h3>Segments</h3>
                <ul>
                    {data.segments.map((seg, idx) => (
                        <li key={idx} className="segment-item">
                            <span className="timestamp">[{formatTime(seg.start)} - {formatTime(seg.end)}]</span>
                            {seg.speaker && <span className="speaker">{seg.speaker} </span>}
                            <span className="text">{seg.text}</span>
                        </li>
                    ))}
                </ul>
            </div>

            <div className="raw-srt">
                <h3>{data.output_format.toUpperCase()} Output</h3>
                <pre>{data.caption_content}</pre>
            </div>
        </div>
    );
};

const formatTime = (seconds: number) => {
    const date = new Date(0);
    date.setSeconds(seconds);
    date.setMilliseconds((seconds - Math.floor(seconds)) * 1000);
    // Format as MM:SS.mmm
    const isoString = date.toISOString();
    return isoString.substr(14, 9); // Extract MM:SS.mmm part (approx) 
    // Actually isoString is 1970-01-01T00:00:00.000Z
    // substr(11, 8) gives HH:mm:ss. substr(11, 12) gives HH:mm:ss.sss
};

export default ResultsDisplay;
