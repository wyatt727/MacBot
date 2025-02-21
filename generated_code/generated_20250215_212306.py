from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_viewport_size({"width": 600, "height": 400})
    page.goto("https://www.google.com")
    # Add code to draw an octogon here
    browser.close()
