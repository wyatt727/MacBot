import pygame, sys, random, os, math
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Comprehensive Brick Breaker")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)
large_font = pygame.font.SysFont("Arial", 48)
FPS = 60
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (200,0,0)
GREEN = (0,200,0)
BLUE = (0,0,200)
YELLOW = (200,200,0)
ORANGE = (255,165,0)
PURPLE = (160,32,240)
GREY = (128,128,128)
HIGH_SCORE_FILE = "highscore.txt"
MIN_SPEED = 5
class Brick:
    def __init__(self, x, y, width, height, hits):
        self.rect = pygame.Rect(x, y, width, height)
        self.hits = hits
    def draw(self, surface):
        color = PURPLE if self.hits == 3 else ORANGE if self.hits == 2 else RED
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
class PowerUp:
    def __init__(self, x, y, type):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.type = type
        self.speed = 3
    def update(self):
        self.rect.y += self.speed
    def draw(self, surface):
        color = GREEN if self.type == "expand" else YELLOW if self.type == "life" else BLUE if self.type == "fast" else GREY
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
class Game:
    def __init__(self):
        self.state = "menu"
        self.level = 1
        self.score = 0
        self.lives = 3
        self.high_score = self.load_high_score()
        self.paddle = pygame.Rect(WIDTH//2 - 50, HEIGHT - 30, 100, 15)
        self.balls = []
        self.add_ball()
        self.bricks = []
        self.powerups = []
        self.create_level()
        self.paddle_effect_timer = 0
    def load_high_score(self):
        if os.path.exists(HIGH_SCORE_FILE):
            try:
                with open(HIGH_SCORE_FILE, "r") as f:
                    return int(f.read())
            except:
                return 0
        return 0
    def save_high_score(self):
        if self.score > self.high_score:
            self.high_score = self.score
            with open(HIGH_SCORE_FILE, "w") as f:
                f.write(str(self.high_score))
    def create_level(self):
        self.bricks = []
        rows = 5 + self.level
        cols = 10
        brick_width = (WIDTH - 100) // cols
        brick_height = 20
        for row in range(rows):
            for col in range(cols):
                x = 50 + col * brick_width
                y = 50 + row * (brick_height + 5)
                hits = random.choice([1,2,3]) if self.level >= 2 else 1
                self.bricks.append(Brick(x, y, brick_width - 5, brick_height, hits))
    def add_ball(self, pos=None, speed=None):
        if pos is None:
            pos = [WIDTH//2 - 10, HEIGHT//2 - 10]
        if speed is None:
            speed = [random.choice([-4,4]), -4]
        ball_rect = pygame.Rect(pos[0], pos[1], 20, 20)
        self.balls.append({"rect": ball_rect, "speed": speed})
    def reset_balls(self):
        self.balls = []
        self.add_ball()
    def update(self):
        if self.state == "playing":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] and self.paddle.left > 0:
                self.paddle.x -= 7
            if keys[pygame.K_RIGHT] and self.paddle.right < WIDTH:
                self.paddle.x += 7
            balls_to_remove = []
            for ball in self.balls:
                ball["rect"].x += ball["speed"][0]
                ball["rect"].y += ball["speed"][1]
                if ball["rect"].left <= 0 or ball["rect"].right >= WIDTH:
                    ball["speed"][0] *= -1
                if ball["rect"].top <= 0:
                    ball["speed"][1] *= -1
                if ball["rect"].colliderect(self.paddle) and ball["speed"][1] > 0:
                    hit_pos = (ball["rect"].centerx - self.paddle.left) / self.paddle.width
                    new_angle = math.radians(150 - hit_pos * 120)
                    current_speed = math.hypot(ball["speed"][0], ball["speed"][1])
                    if current_speed < MIN_SPEED:
                        current_speed = MIN_SPEED
                    ball["speed"][0] = current_speed * math.cos(new_angle)
                    ball["speed"][1] = -abs(current_speed * math.sin(new_angle))
                brick_index = ball["rect"].collidelist([brick.rect for brick in self.bricks])
                if brick_index != -1:
                    brick = self.bricks[brick_index]
                    brick.hits -= 1
                    self.score += 10
                    if brick.hits <= 0:
                        if random.random() < 0.2:
                            pu_type = random.choice(["expand", "life", "fast", "multi"])
                            self.powerups.append(PowerUp(brick.rect.centerx, brick.rect.centery, pu_type))
                        del self.bricks[brick_index]
                    ball["speed"][1] *= -1
                speed_mag = math.hypot(ball["speed"][0], ball["speed"][1])
                if speed_mag < MIN_SPEED:
                    factor = MIN_SPEED / speed_mag
                    ball["speed"][0] *= factor
                    ball["speed"][1] *= factor
                if ball["rect"].top > HEIGHT:
                    balls_to_remove.append(ball)
            for ball in balls_to_remove:
                if ball in self.balls:
                    self.balls.remove(ball)
            if not self.balls:
                self.lives -= 1
                if self.lives > 0:
                    self.reset_balls()
                else:
                    self.state = "gameover"
                    self.save_high_score()
            for pu in self.powerups[:]:
                pu.update()
                if pu.rect.colliderect(self.paddle):
                    if pu.type == "expand":
                        self.paddle.width += 30
                        self.paddle_effect_timer = 300
                    elif pu.type == "life":
                        self.lives += 1
                    elif pu.type == "fast":
                        for ball in self.balls:
                            ball["speed"][0] *= 1.2
                            ball["speed"][1] *= 1.2
                    elif pu.type == "multi":
                        self.add_ball([self.paddle.centerx, self.paddle.top - 25], [random.choice([-4,4]), -4])
                        self.add_ball([self.paddle.centerx, self.paddle.top - 25], [random.choice([-4,4]), -4])
                    self.powerups.remove(pu)
                elif pu.rect.top > HEIGHT:
                    self.powerups.remove(pu)
            if self.paddle_effect_timer > 0:
                self.paddle_effect_timer -= 1
                if self.paddle_effect_timer == 0:
                    self.paddle.width = 100
            if not self.bricks:
                self.level += 1
                if self.level > 3:
                    self.state = "win"
                    self.save_high_score()
                else:
                    self.reset_balls()
                    self.create_level()
    def draw(self):
        screen.fill(BLACK)
        if self.state == "menu":
            title = large_font.render("Brick Breaker", True, WHITE)
            prompt = font.render("Press ENTER to start", True, WHITE)
            hs_text = font.render("High Score: " + str(self.high_score), True, WHITE)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//3))
            screen.blit(prompt, (WIDTH//2 - prompt.get_width()//2, HEIGHT//2))
            screen.blit(hs_text, (WIDTH//2 - hs_text.get_width()//2, HEIGHT//2 + 50))
        elif self.state in ("playing", "paused"):
            pygame.draw.rect(screen, WHITE, self.paddle)
            for ball in self.balls:
                pygame.draw.ellipse(screen, WHITE, ball["rect"])
            for brick in self.bricks:
                brick.draw(screen)
            for pu in self.powerups:
                pu.draw(screen)
            score_text = font.render("Score: " + str(self.score), True, WHITE)
            lives_text = font.render("Lives: " + str(self.lives), True, WHITE)
            level_text = font.render("Level: " + str(self.level), True, WHITE)
            screen.blit(score_text, (10, HEIGHT - 30))
            screen.blit(lives_text, (WIDTH//2 - lives_text.get_width()//2, HEIGHT - 30))
            screen.blit(level_text, (WIDTH - level_text.get_width() - 10, HEIGHT - 30))
            if self.state == "paused":
                pause_text = large_font.render("PAUSED", True, WHITE)
                screen.blit(pause_text, (WIDTH//2 - pause_text.get_width()//2, HEIGHT//2 - pause_text.get_height()//2))
        elif self.state == "gameover":
            over_text = large_font.render("GAME OVER", True, RED)
            score_text = font.render("Score: " + str(self.score), True, WHITE)
            prompt = font.render("Press R to Restart or Q to Quit", True, WHITE)
            screen.blit(over_text, (WIDTH//2 - over_text.get_width()//2, HEIGHT//3))
            screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2))
            screen.blit(prompt, (WIDTH//2 - prompt.get_width()//2, HEIGHT//2 + 50))
        elif self.state == "win":
            win_text = large_font.render("YOU WIN!", True, GREEN)
            score_text = font.render("Score: " + str(self.score), True, WHITE)
            prompt = font.render("Press R to Restart or Q to Quit", True, WHITE)
            screen.blit(win_text, (WIDTH//2 - win_text.get_width()//2, HEIGHT//3))
            screen.blit(score_text, (WIDTH//2 - score_text.get_width()//2, HEIGHT//2))
            screen.blit(prompt, (WIDTH//2 - prompt.get_width()//2, HEIGHT//2 + 50))
        pygame.display.flip()
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if self.state == "menu":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    self.state = "playing"
            elif self.state in ("playing", "paused"):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.state = "paused" if self.state == "playing" else "playing"
            elif self.state in ("gameover", "win"):
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.__init__()
                        self.state = "playing"
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
    def run(self):
        while True:
            clock.tick(FPS)
            self.handle_events()
            if self.state != "paused":
                self.update()
            self.draw()
if __name__ == "__main__":
    Game().run()
