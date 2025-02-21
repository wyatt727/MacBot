from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto('https://example.com')  # Replace with the actual URL of a calculator site
        page.locator('#number-input').fill('1')
        page.locator('#add-button').click()
        result = page.locator('#result').inner_text()
        print(f"Result: {result}")
        browser.close()

run()
