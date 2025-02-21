import turtle

def draw_square(side_length):
    for _ in range(4):
        turtle.forward(side_length)
        turtle.right(90)

def main():
    turtle.bgcolor("black")
    turtle.color("red")
    draw_square(100)
    turtle.done()

if __name__ == "__main__":
    main()
