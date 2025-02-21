import turtle

def draw_octagon(side_length):
    for _ in range(8):
        turtle.forward(side_length)
        turtle.right(45)

def main():
    turtle.bgcolor("black")
    turtle.color("blue")
    draw_octagon(100)
    turtle.done()

if __name__ == "__main__":
    main()
