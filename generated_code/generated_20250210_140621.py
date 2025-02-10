from playwright.sync_api import sync_playwright

def run(p):
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('https://www.example.com')
    print(page.title())
    browser.close()

with sync_playwright() as p:
    run(p)
