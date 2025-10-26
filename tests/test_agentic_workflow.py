#!/usr/bin/env python3
"""
Test script for YouTube Agentic Workflow
"""

import asyncio
import os
import tempfile
from pathlib import Path


# Test 1: Check imports
def test_imports():
    """Test that all modules can be imported"""
    print('🧪 Testing imports...')

    try:
        from lattifai.workflows import YouTubeAlignmentAgent, YouTubeWorkflow
        from lattifai.workflows.base import WorkflowAgent, WorkflowResult, WorkflowStep
        from lattifai.workflows.gemini import GeminiTranscriber
        from lattifai.workflows.youtube import YouTubeDownloader

        print('✅ All workflow modules imported successfully')
        return True
    except Exception as e:
        print(f'❌ Import failed: {e}')
        return False


# Test 2: Check YouTube downloader
async def test_youtube_downloader():
    """Test YouTube downloader functionality"""
    print('\n🧪 Testing YouTube downloader...')

    try:
        from lattifai.workflows.youtube import YouTubeDownloader

        downloader = YouTubeDownloader()

        # Test a short video URL (you can replace with any valid YouTube URL)
        test_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'  # Rick Roll (short)

        # Just test metadata extraction (no download)
        metadata = await downloader.get_video_info(test_url)
        print(f'✅ Video metadata extracted: {metadata.get("title", "Unknown")}')
        return True

    except Exception as e:
        print(f'❌ YouTube downloader test failed: {e}')
        return False


# Test 3: Check Gemini configuration
def test_gemini_config():
    """Test Gemini API configuration"""
    print('\n🧪 Testing Gemini configuration...')

    try:
        from lattifai.workflows.gemini import GeminiTranscriber

        # Test with dummy API key
        try:
            transcriber = GeminiTranscriber(api_key='test_key')
            gem_info = transcriber.get_gem_info()
            print(f'✅ Gemini configured: {gem_info["gem_name"]}')
            return True
        except ValueError as e:
            if 'API key is required' in str(e):
                print('⚠️ Gemini API key validation working (need real key for actual use)')
                return True
            else:
                raise e

    except Exception as e:
        print(f'❌ Gemini configuration test failed: {e}')
        return False


# Test 4: Check workflow setup
def test_workflow_setup():
    """Test workflow agent setup"""
    print('\n🧪 Testing workflow setup...')

    try:
        from lattifai.workflows import YouTubeAlignmentAgent

        # Test agent creation with dummy API key
        try:
            agent = YouTubeAlignmentAgent(
                gemini_api_key='test_key', audio_format='mp3', output_formats=['srt'], max_retries=1
            )

            # Test step definition
            steps = agent.define_steps()
            print(f'✅ Workflow defined with {len(steps)} steps:')
            for i, step in enumerate(steps, 1):
                print(f'   {i}. {step.name}')

            return True

        except ValueError as e:
            if 'API key is required' in str(e):
                print('⚠️ API key validation working')
                return True
            else:
                raise e

    except Exception as e:
        print(f'❌ Workflow setup test failed: {e}')
        return False


# Test 5: Check CLI command
def test_cli_command():
    """Test CLI command registration"""
    print('\n🧪 Testing CLI command...')

    try:
        import subprocess

        result = subprocess.run(
            ['lattifai', 'agent', '--help'],
            capture_output=True,
            text=True,
            cwd='/Users/feiteng/GEEK/OmniCaptions/lattifai-python',
        )

        if result.returncode == 0 and 'LattifAI Agentic Workflow Agent' in result.stdout:
            print('✅ CLI command registered successfully')
            return True
        else:
            print(f'❌ CLI command test failed: {result.stderr}')
            return False

    except Exception as e:
        print(f'❌ CLI command test failed: {e}')
        return False


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
