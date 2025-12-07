// 转录模型配置
export interface TranscriptionModel {
    value: string;
    label: string;
    languages: string;
}

export const TRANSCRIPTION_MODELS: TranscriptionModel[] = [
    {
        value: 'nvidia/parakeet-tdt-0.6b-v3',
        label: 'NVIDIA Parakeet (Default)',
        languages: 'Supports 25 European languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk'
    },
    {
        value: 'iic/SenseVoiceSmall',
        label: 'Alibaba SenseVoice',
        languages: 'Supports: Chinese (zh), English (en), Japanese (ja), Korean (ko), Cantonese (yue)'
    },
    {
        value: 'gemini-2.5-pro',
        label: 'Google Gemini 2.5 Pro',
        languages: 'Supports 100+ languages (Requires Gemini API Key)'
    },
    {
        value: 'gemini-3-pro-preview',
        label: 'Google Gemini 3 Pro Preview',
        languages: 'Supports 100+ languages (Requires Gemini API Key)'
    },
    {
        value: 'nvidia/canary-1b-v2',
        label: 'NVIDIA Canary',
        languages: 'Multilingual support'
    }
];

// Alignment models configuration
export interface AlignmentModel {
    value: string;
    label: string;
    languages: string;
}

export const ALIGNMENT_MODELS: AlignmentModel[] = [
    {
        value: 'Lattifai/Lattice-1',
        label: 'Lattice-1 (Latest)',
        languages: 'Supports: English, Chinese, German, and mixed languages'
    },
    {
        value: 'Lattifai/Lattice-1-Alpha',
        label: 'Lattice-1-Alpha (Legacy)',
        languages: 'Supports: English only'
    }
];

