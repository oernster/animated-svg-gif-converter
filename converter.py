#!/usr/bin/env python3
"""
SVG to GIF Converter
--------------------
This script converts an animated SVG to an animated GIF.
It captures frames of the SVG animation using Selenium and
compiles them into a GIF using Pillow.

Requirements:
- Python 3.6+
- Selenium (pip install selenium)
- Pillow (pip install pillow)
- webdriver_manager (pip install webdriver-manager)
- NumPy (pip install numpy)
- A web browser (Chrome is used by default)

Features:
- Multithreaded processing for faster conversion
- Frame interpolation for smoother animations
- Transparent background support
- Customizable settings (fps, duration, size, etc.)
- Animation speed control
"""

import os
import time
import argparse
import tempfile
import re
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import threading
import queue
import concurrent.futures
import multiprocessing


def create_interpolated_frame(params):
    """
    Create a single interpolated frame between two keyframes.
    
    Args:
        params: A tuple containing (img1, img2, alpha, output_path)
        
    Returns:
        str: Path to the created interpolated frame
    """
    img1, img2, alpha, output_path = params
    
    # Convert to numpy arrays for easier calculation
    arr1 = np.array(img1).astype(float)
    arr2 = np.array(img2).astype(float)
    
    # Linear interpolation
    arr_blend = arr1 * (1 - alpha) + arr2 * alpha
    
    # Convert back to PIL image
    blended = Image.fromarray(arr_blend.astype(np.uint8))
    
    # Apply a very slight blur to reduce artifacts
    r, g, b, a = blended.split()
    a = a.filter(ImageFilter.GaussianBlur(radius=0.3))
    blended = Image.merge('RGBA', (r, g, b, a))
    
    # Save the blended frame
    blended.save(output_path)
    return output_path


def process_frame(frame_path, background_color):
    """
    Process a single frame to handle transparency and apply effects.
    
    Args:
        frame_path (str): Path to the frame image
        background_color (str): Background color setting
        
    Returns:
        str: The path to the processed frame
    """
    if background_color.lower() == 'transparent':
        img = Image.open(frame_path).convert("RGBA")
        
        # Create a transparent version by keeping only non-background elements
        # This targets black/dark background since we're using a black background in Chrome
        datas = img.getdata()
        new_data = []
        for item in datas:
            # If pixel is black or very close to black (dark background), make it transparent
            if item[0] < 30 and item[1] < 30 and item[2] < 30:
                new_data.append((0, 0, 0, 0))  # Fully transparent
            else:
                # Keep the original pixel
                new_data.append(item)
        
        img.putdata(new_data)
        
        # Apply a slight blur to the alpha channel to soften edges
        r, g, b, a = img.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = Image.merge('RGBA', (r, g, b, a))
        
        img.save(frame_path, "PNG")
    return frame_path


def interpolate_frames(frames, factor, pbar=None):
    """
    Generate intermediate frames by blending consecutive frames.
    
    Args:
        frames (list): List of paths to frame images
        factor (int): Number of intermediate frames to generate between each pair
        pbar (optional): Progress bar to update (unused but kept for compatibility)

    Returns:
        list: List of paths to all frames (original + interpolated)
    """
    if factor <= 1:
        return frames  # No interpolation needed

    all_frames = []
    temp_dir = os.path.dirname(frames[0])
    
    # Get the number of CPU cores, but limit to a reasonable number
    max_workers = min(multiprocessing.cpu_count(), 8)
    
    # Create the function arguments for multithreading
    interp_tasks = []
    
    for i in range(len(frames) - 1):
        # Add the original frame
        all_frames.append(frames[i])
        
        # Create interpolated frames
        img1 = Image.open(frames[i]).convert('RGBA')
        img2 = Image.open(frames[i + 1]).convert('RGBA')
        
        # Apply slight blur to both images before interpolation to reduce flickering
        r1, g1, b1, a1 = img1.split()
        r2, g2, b2, a2 = img2.split()
        
        # Only blur the alpha channel to preserve details
        a1 = a1.filter(ImageFilter.GaussianBlur(radius=0.5))
        a2 = a2.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        img1 = Image.merge('RGBA', (r1, g1, b1, a1))
        img2 = Image.merge('RGBA', (r2, g2, b2, a2))
        
        # Create tasks for each interpolated frame
        for j in range(1, factor):
            # Calculate blend ratio
            alpha = j / factor
            blended_path = os.path.join(temp_dir, f"frame_{i:04d}_{j:02d}_interp.png")
            interp_tasks.append((img1.copy(), img2.copy(), alpha, blended_path))
    
    # Process interpolation in parallel
    interpolated_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(create_interpolated_frame, task): task[3] for task in interp_tasks}
        
        for future in concurrent.futures.as_completed(future_to_path):
            path = future.result()
            interpolated_paths.append(path)
    
    # Sort interpolated frames to ensure correct order
    interpolated_paths.sort()
    
    # Insert interpolated frames in the correct order
    for i in range(len(frames) - 1):
        base_frames = [f for f in interpolated_paths if f.startswith(os.path.join(temp_dir, f"frame_{i:04d}_"))]
        base_frames.sort()
        all_frames.extend(base_frames)
        
        # Don't add the last original frame until the end
        if i < len(frames) - 2:
            all_frames.append(frames[i+1])
    
    # Add the last original frame
    all_frames.append(frames[-1])
    
    return all_frames


def create_html_with_svg(svg_content, width, height, background_color, padding=20):
    """Create an HTML document with the SVG content."""
    
    # Extract viewBox from SVG if it exists to preserve aspect ratio
    viewbox_match = re.search(r'viewBox=["\']([^"\']+)["\']', svg_content)
    viewbox = None
    if viewbox_match:
        viewbox = viewbox_match.group(1)
        # Parse viewBox values
        try:
            vb_values = [float(x) for x in viewbox.split()]
            if len(vb_values) == 4:
                vb_width = vb_values[2]
                vb_height = vb_values[3]
                # Adjust height to maintain aspect ratio if needed
                if (vb_width / vb_height) != (width / height):
                    height = int(width * (vb_height / vb_width))
                    print(f"Adjusted height to {height}px to maintain aspect ratio")
        except (ValueError, ZeroDivisionError):
            pass
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            html {{
                margin: 0;
                padding: 0;
                background-color: {background_color if background_color.lower() != 'transparent' else 'transparent'};
                width: {width}px;
                height: {height}px;
                overflow: hidden;
            }}
            body {{
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
                padding: {padding}px;
                box-sizing: border-box;
                width: 100%;
                height: 100%;
                margin: 0;
            }}
            svg {{
                max-width: calc(100% - {padding*2}px);
                max-height: calc(100% - {padding*2}px);
                width: auto;
                height: auto;
                display: block;
                margin: auto;
            }}
            .svg-container {{
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
        </style>
        <script>
            // This script will be used to control the animation speed
            window.onload = function() {{
                // Find all animation elements
                const animations = document.querySelectorAll('svg animate, svg animateTransform, svg animateMotion');
                
                // Slow down all animations by modifying their durations
                animations.forEach(anim => {{
                    // Get current duration
                    const origDur = anim.getAttribute('dur');
                    if (origDur) {{
                        // Parse the duration value and unit
                        const match = origDur.match(/([\d.]+)([a-z]+)/);
                        if (match) {{
                            const value = parseFloat(match[1]);
                            const unit = match[2];
                            // Slow down by a significant factor
                            const newDur = (value * 50) + unit;
                            anim.setAttribute('dur', newDur);
                            console.log('Animation slowed: ' + origDur + ' to ' + newDur);
                        }}
                    }}
                    
                    // Force animations to have only one cycle
                    if (anim.getAttribute('repeatCount') && anim.getAttribute('repeatCount') !== 'indefinite') {{
                        anim.setAttribute('repeatCount', '1');
                    }}
                    
                    // Make sure all animations start at the beginning
                    anim.setAttribute('begin', '0s');
                }});
            }};
        </script>
    </head>
    <body>
        <div class="svg-container">
            {svg_content}
        </div>
    </body>
    </html>
    """


def convert_svg_to_gif(svg_path, output_gif=None, width=800, height=1500, 
                      duration=100000, fps=10, background_color='transparent', 
                      padding=100, interpolate_factor=5, animation_slowdown=50):
    """
    Convert an SVG animation to a GIF.
    
    Args:
        svg_path (str): Path to the SVG file
        output_gif (str, optional): Output GIF path. Defaults to SVG name with .gif extension.
        width (int, optional): Width of the output GIF. Defaults to 800.
        height (int, optional): Height of the output GIF. Defaults to 1500.
        duration (int, optional): Animation duration in milliseconds. Defaults to 100000.
        fps (int, optional): Frames per second. Defaults to 10.
        background_color (str, optional): Background color. Defaults to 'transparent'.
        padding (int, optional): Padding around the SVG in pixels. Defaults to 100.
        interpolate_factor (int, optional): Number of frames to generate between keyframes. Defaults to 5.
        animation_slowdown (int, optional): Factor to slow down the SVG animation. Defaults to 50.
    
    Returns:
        str: Path to the generated GIF
    """
    # Ensure output path
    if not output_gif:
        output_gif = os.path.splitext(svg_path)[0] + '.gif'
    
    # Read the SVG file
    with open(svg_path, 'r', encoding='utf-8') as f:
        original_svg_content = f.read()
    
    # Modify the SVG content to slow down animations if needed
    svg_content = original_svg_content
    if animation_slowdown > 1:
        # This is a basic string replacement approach for simple SVGs
        # The more robust approach is the JavaScript solution in the HTML
        svg_content = re.sub(r'dur="([\d.]+)([a-z]+)"', 
                            lambda m: f'dur="{float(m.group(1)) * animation_slowdown}{m.group(2)}"', 
                            svg_content)
    
    # Import browser-related modules here to prevent import errors if not needed
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.chrome.options import Options
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w', encoding='utf-8') as temp_html:
        html_content = create_html_with_svg(svg_content, width, height, background_color, padding)
        temp_html.write(html_content)
        html_path = temp_html.name
    
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--window-size={width},{height}")
    
    # Setup WebDriver
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    
    try:
        # Calculate total frames needed
        total_frames = int(duration / 1000 * fps)
        frame_duration_ms = 1000 // fps
        
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load the HTML file
            driver.get(f"file://{html_path}")
            
            # Wait for animations to initialize
            time.sleep(1)
            
            # Apply additional animation slowdown via JavaScript
            driver.execute_script(f"""
                // Apply animation slowdown dynamically
                const animations = document.querySelectorAll('svg animate, svg animateTransform, svg animateMotion');
                animations.forEach(anim => {{
                    // Get current duration
                    const origDur = anim.getAttribute('dur');
                    if (origDur) {{
                        // Parse the duration value and unit
                        const match = origDur.match(/([\d.]+)([a-z]+)/);
                        if (match) {{
                            const value = parseFloat(match[1]);
                            const unit = match[2];
                            // Slow down by specified factor
                            const newDur = (value * {animation_slowdown}) + unit;
                            anim.setAttribute('dur', newDur);
                            console.log('Animation slowed: ' + origDur + ' to ' + newDur);
                        }}
                    }}
                    
                    // Force animations to have only one repeat
                    if (anim.getAttribute('repeatCount') && anim.getAttribute('repeatCount') !== 'indefinite') {{
                        anim.setAttribute('repeatCount', '1');
                    }}
                    
                    // Make sure all animations start at the beginning
                    anim.setAttribute('begin', '0s');
                }});
            """)
            
            # Wait for the above JavaScript to take effect
            time.sleep(1)
            
            # Capture frames
            frames = []
            print("Capturing frames from SVG animation...")
            
            # Create frame paths in advance
            frame_paths = [os.path.join(temp_dir, f"frame_{frame_num:04d}.png") for frame_num in range(total_frames)]
            
            for frame_num in range(total_frames):
                # Calculate the time to set for this frame
                current_time_ms = frame_num * frame_duration_ms
                
                # Add a small pause to allow the animation to render properly
                time.sleep(0.01)
                
                # Set the animation to the current time
                driver.execute_script(f"""
                    // Set current animation time if possible
                    setTimeout(() => {{}}, {current_time_ms});
                """)
                
                # Capture screenshot
                driver.save_screenshot(frame_paths[frame_num])
                frames.append(frame_paths[frame_num])
                print(f"Captured frame {frame_num+1}/{total_frames}")
            
            # Process frames in parallel
            print("Processing frames with transparency...")
            max_workers = min(multiprocessing.cpu_count(), 8)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(process_frame, frame, background_color): frame for frame in frames}
                
                for future in concurrent.futures.as_completed(future_to_path):
                    future.result()  # This is just to catch any exceptions
                
            # Generate intermediate frames using interpolation if requested
            if interpolate_factor > 1:
                print("Generating interpolated frames...")
                original_frame_count = len(frames)
                frames = interpolate_frames(frames, interpolate_factor)
                print(f"Total frames after interpolation: {len(frames)}")
            
            # Create GIF
            print("Creating GIF...")
            images = []
            for frame in frames:
                images.append(Image.open(frame))
            
            # Save as GIF with transparency
            print("Saving GIF file...")
            
            images[0].save(
                output_gif,
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=frame_duration_ms,
                loop=0,
                transparency=0,
                disposal=2  # To properly handle transparency between frames
            )
            
            print(f"GIF saved to: {output_gif}")
            return output_gif
    
    finally:
        # Clean up
        driver.quit()
        os.unlink(html_path)


def main():
    parser = argparse.ArgumentParser(description='Convert an animated SVG to a GIF')
    parser.add_argument('svg_path', help='Path to the SVG file')
    parser.add_argument('--output', '-o', help='Output GIF path')
    parser.add_argument('--width', '-w', type=int, default=800, help='Width of the output GIF')
    parser.add_argument('--height', '--h', type=int, default=1500, help='Height of the output GIF')
    parser.add_argument('--padding', '-p', type=int, default=100, help='Padding around the SVG (in pixels)')
    parser.add_argument('--duration', '-d', type=int, default=100000, 
                      help='Animation duration in milliseconds')
    parser.add_argument('--fps', '-f', type=int, default=10, help='Frames per second')
    parser.add_argument('--interpolate', '-i', type=int, default=5, 
                      help='Interpolation factor: number of frames to generate between keyframes')
    parser.add_argument('--slowdown', '-s', type=int, default=50, 
                      help='Animation slowdown factor (higher number = slower animation)')
    parser.add_argument('--background', '-b', default='transparent', help='Background color (CSS format or "transparent")')
    
    args = parser.parse_args()
    
    convert_svg_to_gif(
        args.svg_path,
        args.output,
        args.width,
        args.height,
        args.duration,
        args.fps,
        args.background,
        args.padding,
        args.interpolate,
        args.slowdown
    )


if __name__ == "__main__":
    # Display some information about multithreading
    cpu_count = multiprocessing.cpu_count()
    print(f"SVG to GIF Converter (Multithreaded)")
    print(f"Detected {cpu_count} CPU cores, will use maximum of {min(cpu_count, 8)} threads")
    print("-" * 50)
    
    main()