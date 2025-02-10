import random
import sys
import os

def clear_screen():
    if sys.platform == "win32":
        os.system('cls')
    else:
        os.system('clear')

def get_random_position(max_x, max_y):
    return random.randint(0, max_x - 1), random.randint(0, max_y - 1)

def create_snake():
    head = (5, 5)
    body = [head]
    direction = (-1, 0)
    return body, direction

def move_snake(body, direction):
    new_head = (body[0][0] + direction[0], body[0][1] + direction[1])
    body.insert(0, new_head)
    body.pop()
    return body

def grow_snake(body, food):
    if body[0] == food:
        body.append(food)

def check_collision(body, max_x, max_y):
    head = body[0]
    if head[0] < 0 or head[0] >= max_x or head[1] < 0 or head[1] >= max_y:
        return True
    for segment in body[1:]:
        if head == segment:
            return True
    return False

def draw_snake(body):
    for segment in body:
        print("*" * segment[0] + "*" + "*" * (max_x - segment[0] - 1))
        print("*" * segment[1] + "#" + "*" * (max_y - segment[1] - 1))

def main():
    max_x = 20
    max_y = 10
    body, direction = create_snake()
    food = get_random_position(max_x, max_y)

    while True:
        clear_screen()
        draw_snake(body)
        print("Food at:", food)
        if check_collision(body, max_x, max_y):
            break
        command = input("Enter a direction (w/a/s/d): ")
        if command == "w":
            direction = (-1, 0)
        elif command == "a":
            direction = (0, -1)
        elif command == "s":
            direction = (1, 0)
        elif command == "d":
            direction = (0, 1)
        body = move_snake(body, direction)
        grow_snake(body, food)
        if check_collision(body, max_x, max_y):
            break

if __name__ == "__main__":
    main()
