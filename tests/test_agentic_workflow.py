#!/usr/bin/env python3
"""
Test script for YouTube Agentic Workflow
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest


# Test 1: Check imports
def test_imports():
    """Test that all modules can be imported"""
    print('🧪 Testing imports...')

    from lattifai.workflows import YouTubeSubtitleAgent
    from lattifai.workflows.base import WorkflowAgent, WorkflowResult, WorkflowStep
    from lattifai.workflows.gemini import GeminiTranscriber
    from lattifai.workflows.youtube import YouTubeDownloader

    print('✅ All workflow modules imported successfully')
    assert YouTubeSubtitleAgent is not None
    assert WorkflowAgent is not None
    assert GeminiTranscriber is not None
    assert YouTubeDownloader is not None


# Test 2: Check YouTube downloader
@pytest.mark.asyncio
async def test_youtube_downloader():
    """Test YouTube downloader functionality"""
    print('\n🧪 Testing YouTube downloader...')

    from lattifai.workflows.youtube import YouTubeDownloader

    downloader = YouTubeDownloader()

    # Test a short video URL (you can replace with any valid YouTube URL)
    test_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Rick Roll (short)

    # Just test metadata extraction (no download)
    metadata = await downloader.get_video_info(test_url)
    print(f'✅ Video metadata extracted: {metadata.get("title", "Unknown")}')

    assert metadata is not None
    assert 'title' in metadata
    assert len(metadata['title']) > 0


# Test 3: Check Gemini configuration
def test_gemini_config():
    """Test Gemini API configuration"""
    print('\n🧪 Testing Gemini configuration...')

    from lattifai.workflows.gemini import GeminiTranscriber

    # Test with dummy API key - it should accept it at initialization
    # (validation happens when actually using the API)
    transcriber = GeminiTranscriber(api_key='test_key')
    gem_info = transcriber.get_gem_info()

    print(f'✅ Gemini configured: {gem_info["gem_name"]}')
    print('⚠️ Note: actual API key validation happens when making requests')

    assert transcriber is not None
    assert gem_info is not None
    assert 'gem_name' in gem_info


# Test 4: Check workflow setup
def test_workflow_setup():
    """Test workflow agent setup"""
    print('\n🧪 Testing workflow setup...')

    from unittest.mock import MagicMock

    from lattifai.workflows import YouTubeSubtitleAgent

    # Test agent creation with mock components
    mock_downloader = MagicMock()
    mock_transcriber = MagicMock()
    mock_aligner = MagicMock()

    agent = YouTubeSubtitleAgent(
        downloader=mock_downloader, transcriber=mock_transcriber, aligner=mock_aligner, max_retries=1
    )

    # Test step definition
    steps = agent.define_steps()
    print(f'✅ Workflow defined with {len(steps)} steps:')
    for i, step in enumerate(steps, 1):
        print(f'   {i}. {step.name}')

    assert agent is not None
    assert steps is not None
    assert len(steps) > 0
    assert len(steps) == 4  # Should have 4 steps


# Test 5: Check CLI command
def test_cli_command():
    """Test CLI command registration"""
    print('\n🧪 Testing CLI command...')

    import subprocess

    result = subprocess.run(
        ['lattifai', 'agent', '--help'],
        capture_output=True,
        text=True,
        cwd='/Users/feiteng/GEEK/OmniCaptions/lattifai-python',
    )

    print('✅ CLI command registered successfully')
    assert result.returncode == 0
    assert 'LattifAI Agentic Workflow Agent' in result.stdout or 'agent' in result.stdout.lower()


async def main():
    """Run all tests"""
    print('🚀 LattifAI Agentic Workflow Test Suite')
    print('=' * 50)

    tests = [
        ('Import Test', test_imports),
        ('YouTube Downloader Test', test_youtube_downloader),
        ('Gemini Configuration Test', test_gemini_config),
        ('Workflow Setup Test', test_workflow_setup),
        ('CLI Command Test', test_cli_command),
    ]

    results = []

    for test_name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))

    print('\n' + '=' * 50)
    print('📊 Test Results Summary:')

    passed = 0
    for test_name, result in results:
        status = '✅ PASS' if result else '❌ FAIL'
        print(f'  {status}: {test_name}')
        if result:
            passed += 1

    print(f'\n🎯 {passed}/{len(tests)} tests passed')

    if passed == len(tests):
        print('\n🎉 All tests passed! The agentic workflow is ready to use.')
        print('\nNext steps:')
        print('1. Set your Gemini API key in .env file or use --gemini-api-key')
        print('2. Run: lattifai agent --youtube <youtube_url>')
    else:
        print('\n⚠️ Some tests failed. Please check the implementation.')


if __name__ == '__main__':
    asyncio.run(main())
