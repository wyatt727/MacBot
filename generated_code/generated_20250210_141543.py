from playwright.sync_api import sync_playwright

def whoami():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto('https://example.com')
        content = page.content()
        print(content)
        browser.close()

whoami()
