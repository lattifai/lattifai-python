import React, { useState } from 'react';
import axios from 'axios';

interface AlignmentFormProps {
    onResult: (data: any) => void;
    onLoading: (isLoading: boolean) => void;
}

const AlignmentForm: React.FC<AlignmentFormProps> = ({ onResult, onLoading }) => {
    const [mediaFile, setMediaFile] = useState<File | null>(null);
    const [captionFile, setCaptionFile] = useState<File | null>(null);
    const [youtubeUrl, setYoutubeUrl] = useState('');
    const [splitSentence, setSplitSentence] = useState(true);
    const [isTranscription, setIsTranscription] = useState(true);
    const [normalizeText, setNormalizeText] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        onLoading(true);
        onResult(null); // Clear previous results

        const formData = new FormData();
        if (mediaFile) formData.append('media_file', mediaFile);
        if (captionFile) formData.append('caption_file', captionFile);
        if (youtubeUrl) formData.append('youtube_url', youtubeUrl);
        formData.append('split_sentence', String(splitSentence));
        formData.append('is_transcription', String(isTranscription));
        formData.append('normalize_text', String(normalizeText));

        try {
            const response = await axios.post('/align', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            onResult(response.data);
        } catch (err: any) {
            console.error(err);
            const detail = err.response?.data?.detail;
            const errMsg = err.response?.data?.error;
            setError(
                errMsg ||
                (detail ? JSON.stringify(detail) : null) ||
                err.message ||
                'An error occurred during alignment.'
            );
        } finally {
            onLoading(false);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="alignment-form card">
            <h2>Alignment Request</h2>
            <div className="form-group">
                <label>Media File (Audio/Video):</label>
                <input type="file" accept="audio/*,video/*" onChange={e => setMediaFile(e.target.files?.[0] || null)} />
            </div>

            <div className="separator">OR</div>

            <div className="form-group">
                <label>YouTube URL:</label>
                <input type="text" value={youtubeUrl} onChange={e => setYoutubeUrl(e.target.value)} placeholder="https://www.youtube.com/watch?v=..." className="input-text" />
            </div>

            <div className="separator">AND</div>

            <div className="form-group">
                <label>Caption File (Optional):</label>
                <input type="file" accept=".srt,.vtt,.txt" onChange={e => setCaptionFile(e.target.files?.[0] || null)} />
            </div>

            <div className="options-group">
                <label className="checkbox-label">
                    <input type="checkbox" checked={splitSentence} onChange={e => setSplitSentence(e.target.checked)} />
                    Split Sentence
                </label>
                <label className="checkbox-label">
                    <input type="checkbox" checked={isTranscription} onChange={e => setIsTranscription(e.target.checked)} />
                    Use Transcription (if no caption)
                </label>
                <label className="checkbox-label">
                    <input type="checkbox" checked={normalizeText} onChange={e => setNormalizeText(e.target.checked)} />
                    Normalize Text
                </label>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button type="submit" disabled={!mediaFile && !youtubeUrl} className="submit-btn">Align</button>
        </form>
    );
};

export default AlignmentForm;
