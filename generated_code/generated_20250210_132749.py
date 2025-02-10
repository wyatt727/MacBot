import pygame
import sys
import random

pygame.init()
width, height = 640, 480
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
snake_pos = [[100, 50], [90, 50], [80, 50]]
snake_direction = 'RIGHT'
food_pos = [random.randrange(1, width // 10) * 10, random.randrange(1, height // 10) * 10]
score = 0

def game_over():
    font = pygame.font.SysFont('arial', 35)
    surface = font.render('Game Over! Score: ' + str(score), True, (255, 0, 0))
    rect = surface.get_rect()
    rect.midtop = (width / 2, height / 4)
    screen.blit(surface, rect)
    pygame.display.flip()
    pygame.time.wait(2000)
    pygame.quit()
    sys.exit()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP and snake_direction != 'DOWN':
                snake_direction = 'UP'
            elif event.key == pygame.K_DOWN and snake_direction != 'UP':
                snake_direction = 'DOWN'
            elif event.key == pygame.K_LEFT and snake_direction != 'RIGHT':
                snake_direction = 'LEFT'
            elif event.key == pygame.K_RIGHT and snake_direction != 'LEFT':
                snake_direction = 'RIGHT'
    if snake_direction == 'UP':
        new_head = [snake_pos[0][0], snake_pos[0][1] - 10]
    elif snake_direction == 'DOWN':
        new_head = [snake_pos[0][0], snake_pos[0][1] + 10]
    elif snake_direction == 'LEFT':
        new_head = [snake_pos[0][0] - 10, snake_pos[0][1]]
    elif snake_direction == 'RIGHT':
        new_head = [snake_pos[0][0] + 10, snake_pos[0][1]]
    snake_pos.insert(0, new_head)
    if snake_pos[0] == food_pos:
        score += 1
        food_pos = [random.randrange(1, width // 10) * 10, random.randrange(1, height // 10) * 10]
    else:
        snake_pos.pop()
    if snake_pos[0][0] < 0 or snake_pos[0][0] > width - 10 or snake_pos[0][1] < 0 or snake_pos[0][1] > height - 10:
        game_over()
    for block in snake_pos[1:]:
        if snake_pos[0] == block:
            game_over()
    screen.fill((0, 0, 0))
    for pos in snake_pos:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(food_pos[0], food_pos[1], 10, 10))
    font = pygame.font.SysFont('arial', 25)
    score_surface = font.render('Score : ' + str(score), True, (255, 255, 255))
    score_rect = score_surface.get_rect()
    score_rect.midtop = (width / 2, 10)
    screen.blit(score_surface, score_rect)
    pygame.display.flip()
    clock.tick(15)
