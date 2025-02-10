from playwright.sync_api import sync_playwright

def get_user():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto('http://example.com')  # Example URL, replace with actual command or action if needed
        user = page.evaluate("window.navigator.userAgent;")
        print(user)
        browser.close()
