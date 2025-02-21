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
        
        # Add a div element to act as the triangle
        page.eval_on_selector("#triangle", "element => { element.style.position = 'absolute'; element.style.left = '400px'; element.style.top = '250px'; }")
        
        # Animate the triangle to rotate
        page.evaluate("""() => {
            const triangles = document.querySelectorAll('div');
            let angle = 0;
            setInterval(() => {
                angle += 10;
                triangles.forEach(triangle => {
                    triangle.style.transform = `rotate(${angle}deg)`;
                });
            }, 50);
        }""")
        
        page.wait_for_timeout(3000)  # Wait for 3 seconds to see the animation
        browser.close()

animate_triangle()
