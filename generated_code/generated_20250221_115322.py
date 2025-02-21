import pygame, sys, random
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")
clock = pygame.time.Clock()
PADDLE_WIDTH, PADDLE_HEIGHT = 100, 20
paddle = pygame.Rect(WIDTH//2 - PADDLE_WIDTH//2, HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)
BALL_SIZE = 16
ball = pygame.Rect(WIDTH//2 - BALL_SIZE//2, HEIGHT//2 - BALL_SIZE//2, BALL_SIZE, BALL_SIZE)
ball_speed_x, ball_speed_y = 5 * random.choice((1, -1)), -5
BRICK_ROWS, BRICK_COLS = 6, 10
BRICK_WIDTH, BRICK_HEIGHT = 60, 20
bricks = []
for row in range(BRICK_ROWS):
    for col in range(BRICK_COLS):
        brick_x = col * (BRICK_WIDTH + 10) + 35
        brick_y = row * (BRICK_HEIGHT + 10) + 35
        bricks.append(pygame.Rect(brick_x, brick_y, BRICK_WIDTH, BRICK_HEIGHT))
font = pygame.font.SysFont("Arial", 36)
def reset():
    global paddle, ball, ball_speed_x, ball_speed_y, bricks
    paddle = pygame.Rect(WIDTH//2 - PADDLE_WIDTH//2, HEIGHT - 40, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = pygame.Rect(WIDTH//2 - BALL_SIZE//2, HEIGHT//2 - BALL_SIZE//2, BALL_SIZE, BALL_SIZE)
    ball_speed_x, ball_speed_y = 5 * random.choice((1, -1)), -5
    bricks = []
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            brick_x = col * (BRICK_WIDTH + 10) + 35
            brick_y = row * (BRICK_HEIGHT + 10) + 35
            bricks.append(pygame.Rect(brick_x, brick_y, BRICK_WIDTH, BRICK_HEIGHT))
running = True
game_over = False
win = False
while running:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and paddle.left > 0:
        paddle.x -= 7
    if keys[pygame.K_RIGHT] and paddle.right < WIDTH:
        paddle.x += 7
    if not game_over and not win:
        ball.x += ball_speed_x
        ball.y += ball_speed_y
        if ball.left <= 0 or ball.right >= WIDTH:
            ball_speed_x *= -1
        if ball.top <= 0:
            ball_speed_y *= -1
        if ball.colliderect(paddle) and ball_speed_y > 0:
            ball_speed_y *= -1
            offset = (ball.centerx - paddle.centerx) / (PADDLE_WIDTH/2)
            ball_speed_x = 5 * offset
        brick_hit_index = ball.collidelist(bricks)
        if brick_hit_index != -1:
            hit_brick = bricks.pop(brick_hit_index)
            if abs(ball.centerx - hit_brick.left) < BALL_SIZE and ball_speed_x > 0:
                ball_speed_x *= -1
            elif abs(ball.centerx - hit_brick.right) < BALL_SIZE and ball_speed_x < 0:
                ball_speed_x *= -1
            else:
                ball_speed_y *= -1
        if ball.bottom >= HEIGHT:
            game_over = True
        if not bricks:
            win = True
    screen.fill((0, 0, 0))
    pygame.draw.rect(screen, (255, 255, 255), paddle)
    pygame.draw.ellipse(screen, (255, 255, 255), ball)
    for brick in bricks:
        pygame.draw.rect(screen, (200, 0, 0), brick)
    if game_over:
        text = font.render("Game Over", True, (255, 255, 255))
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
        text2 = font.render("Press R to Restart", True, (255, 255, 255))
        screen.blit(text2, (WIDTH//2 - text2.get_width()//2, HEIGHT//2 + text.get_height()))
    if win:
        text = font.render("You Win!", True, (255, 255, 255))
        screen.blit(text, (WIDTH//2 - text.get_width()//2, HEIGHT//2 - text.get_height()//2))
        text2 = font.render("Press R to Restart", True, (255, 255, 255))
        screen.blit(text2, (WIDTH//2 - text2.get_width()//2, HEIGHT//2 + text.get_height()))
    pygame.display.flip()
    if game_over or win:
        if keys[pygame.K_r]:
            reset()
            game_over = False
            win = False
