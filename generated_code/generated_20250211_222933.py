from playwright.sync_api import sync_playwright

def animate_triangle():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 800, "height": 600})
        
        # Add a yellow triangle to the page
        page.add_style_tag(content="""
            div {
                width: 0;
                height: 0;
                border-left: 50px solid yellow;
                border-right: 50px solid transparent;
                border-top: 50px solid transparent;
                border-bottom: 50px solid transparent;
            }
        """)
        page.add_css_property("#triangle", "position", "absolute")
        page.add_css_property("#triangle", "left", "400px")
        page.add_css_property("#triangle", "top", "250px")
        
        # Animate the triangle to rotate
        page.evaluate("""() => {
            const triangle = document.getElementById('triangle');
            let angle = 0;
            setInterval(() => {
                angle += 10;
                triangle.style.transform = `rotate(${angle}deg)`;
            }, 50);
        }""")
        
        page.wait_for_timeout(3000)  # Wait for 3 seconds to see the animation
        browser.close()

animate_triangle()
