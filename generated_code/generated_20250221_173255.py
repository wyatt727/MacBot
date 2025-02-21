import turtle

def draw_triangle():
    for _ in range(3):
        turtle.forward(100)
        turtle.right(120)

def main():
    turtle.bgcolor("black")
    turtle.color("green")
    draw_triangle()
    turtle.done()

if __name__ == "__main__":
    main()
