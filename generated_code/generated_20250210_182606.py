from playwright.sync_api import sync_playwright
import os

def download_video(url, output_path):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        
        # Assuming the video element has a data-testid attribute that can be used to locate it
        video_element = page.query_selector('video[data-testid="video"]')
        if video_element:
            video_src = video_element.get_attribute("src")
            os.system(f"curl -o {output_path} '{video_src}'")
        else:
            print("Video element not found.")
        
        browser.close()

# Usage
download_video('https://www.facebook.com/beckerfornd/videos/1701155050704749', 'output_video.mp4')
