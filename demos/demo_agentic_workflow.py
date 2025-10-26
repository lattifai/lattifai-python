#!/usr/bin/env python3
"""
Demo script for LattifAI Agentic Workflow
Complete functionality demonstration script
"""

import os
import sys
from pathlib import Path
import colorful

def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                 LattifAI Agentic Workflow                    ║
║              YouTube → Perfect Timeline Alignment            ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_setup():
    """Check setup status"""
    print("🔍 Checking environment setup...")

    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if 'GEMINI_API_KEY=your_gemini_api_key_here' in content:
                print("⚠️  Please set a real GEMINI_API_KEY in .env file")
                return False
            elif 'GEMINI_API_KEY=' in content:
                print("✅ Gemini API Key configured")
            else:
                print("⚠️  GEMINI_API_KEY not found in .env file")
                return False
    else:
        print("⚠️  .env file does not exist")
        return False

    # Check dependencies
    try:
        import yt_dlp
        from google import genai
        import lattifai.workflows
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

    return True

def show_examples():
    """Show usage examples"""
    print("\n📚 Usage Examples:")
    print("-" * 60)

    examples = [
        {
            "title": "Basic Usage - Generate SRT Subtitles",
            "command": "lattifai agent --youtube https://www.youtube.com/watch?v=example"
        },
        {
            "title": "Multiple Format Output - SRT + VTT + ASS",
            "command": "lattifai agent --youtube --output-formats srt,vtt,ass https://www.youtube.com/watch?v=example"
        },
        {
            "title": "Specify Audio Format and Output Directory",
            "command": "lattifai agent --youtube --audio-format wav --output-dir ./my_subtitles https://www.youtube.com/watch?v=example"
        },
        {
            "title": "Enable Retry and Verbose Logging",
            "command": "lattifai agent --youtube --max-retries 2 --verbose https://www.youtube.com/watch?v=example"
        },
        {
            "title": "Specify API Key (if not set in .env)",
            "command": "lattifai agent --youtube --gemini-api-key YOUR_API_KEY https://www.youtube.com/watch?v=example"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['command']}")

def show_workflow_steps():
    """Show workflow steps"""
    print("\n🔄 Workflow Process:")
    print("-" * 60)

    steps = [
        "🎥  Process YouTube URL - Extract metadata and download audio",
        "🎤 Gemini Transcription - Generate text transcription using Gemini 2.5 Pro",
        "🎯   LattifAI Alignment - Precise timeline alignment",
        "📤       Export Results - Generate multi-format subtitle files"
    ]

    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

def show_supported_formats():
    """Show supported formats"""
    print(colorful.bold_green("\n📝 Supported Formats:"))
    print("-" * 60)

    print("Audio Formats:")
    audio_formats = ["MP3 (Recommended)", "WAV", "M4A", "AAC"]
    for fmt in audio_formats:
        print(f"  • {fmt}")

    print("\nSubtitle Formats:")
    subtitle_formats = ["SRT (SubRip)", "VTT (WebVTT)", "ASS (Advanced SubStation Alpha)", "TXT (Plain Text)"]
    for fmt in subtitle_formats:
        print(f"  • {fmt}")

def interactive_demo():
    """Interactive demonstration"""
    print(colorful.bold_green("\n🎮 Interactive Demo"))
    print("-" * 60)

    while True:
        print("\nPlease select an option:")
        print("1. View command help")
        print("2. Run test URL (short video)")
        print("3. Custom YouTube URL")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()

        if choice == "1":
            os.system("lattifai agent --help")

        elif choice == "2":
            test_url = "https://www.youtube.com/shorts/cprOj8PWepY"
            print(f"\n🧪 Running test URL: {test_url}")
            print("Note: This is a test short video")

            confirm = input("Confirm to run? (y/N): ").strip().lower()
            if confirm == 'y':
                cmd = f"lattifai agent --youtube --verbose {test_url}"
                print(f"\nExecuting command: {cmd}")
                os.system(cmd)
            else:
                print("Cancelled")

        elif choice == "3":
            url = input("Please enter YouTube URL: ").strip()
            if url:
                print(f"\nProcessing URL: {url}")

                # Ask for configuration options
                audio_format = input("Audio format (mp3/wav/m4a/aac) [mp3]: ").strip() or "mp3"
                output_formats = input("Output formats (srt,vtt,ass,txt) [srt]: ").strip() or "srt"

                cmd = f"lattifai agent --youtube --audio-format {audio_format} --output-formats {output_formats} --verbose {url}"
                print(f"\nExecuting command: {cmd}")

                confirm = input("Confirm to run? (y/N): ").strip().lower()
                if confirm == 'y':
                    os.system(cmd)
                else:
                    print("Cancelled")
            else:
                print("URL cannot be empty")

        elif choice == "4":
            print("👋 Goodbye!")
            break

        else:
            print("Invalid choice, please try again")

def main():
    """Main function"""
    print_banner()

    # Check setup
    if not check_setup():
        print("\n❌ Environment setup incomplete, please configure according to documentation")
        print("\nConfiguration steps:")
        print("1. Set GEMINI_API_KEY in .env file")
        print("2. Ensure all dependencies are installed: pip install yt-dlp google-genai python-dotenv")
        return

    print("✅ Environment check passed!")

    # Show functionality description
    show_workflow_steps()
    show_supported_formats()
    show_examples()

    # Interactive demo
    interactive_demo()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 User interrupted, goodbye!")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        sys.exit(1)