import pygame, sys, random, time, math
pygame.init()
CELL_SIZE = 30
ROWS, COLS = 16, 16
MINES_COUNT = 40
UI_HEIGHT = 50
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE + UI_HEIGHT
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minesweeper")
font = pygame.font.SysFont("comicsans", 20)
large_font = pygame.font.SysFont("comicsans", 40)
clock = pygame.time.Clock()
class Cell:
    def __init__(self, r, c):
        self.row = r
        self.col = c
        self.rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE + UI_HEIGHT, CELL_SIZE, CELL_SIZE)
        self.is_mine = False
        self.revealed = False
        self.flagged = False
        self.adjacent = 0
    def draw(self, surf, show_mine=False):
        if self.revealed:
            pygame.draw.rect(surf, (200,200,200), self.rect)
            if self.is_mine:
                pygame.draw.circle(surf, (0,0,0), self.rect.center, CELL_SIZE // 3)
            elif self.adjacent > 0:
                colors = {1:(0,0,255),2:(0,128,0),3:(255,0,0),4:(0,0,128),5:(128,0,0),6:(0,128,128),7:(0,0,0),8:(128,128,128)}
                txt = font.render(str(self.adjacent), True, colors.get(self.adjacent, (0,0,0)))
                surf.blit(txt, (self.rect.x + (CELL_SIZE - txt.get_width()) // 2, self.rect.y + (CELL_SIZE - txt.get_height()) // 2))
        else:
            pygame.draw.rect(surf, (100,100,100), self.rect)
            if self.flagged:
                pygame.draw.polygon(surf, (255,0,0), [(self.rect.x + CELL_SIZE//2, self.rect.y+5), (self.rect.x+5, self.rect.y+CELL_SIZE-5), (self.rect.x+CELL_SIZE-5, self.rect.y+CELL_SIZE-5)])
        pygame.draw.rect(surf, (0,0,0), self.rect, 1)
        if show_mine and self.is_mine and not self.revealed:
            pygame.draw.circle(surf, (0,0,0), self.rect.center, CELL_SIZE // 3)
class MinesweeperGame:
    def __init__(self, rows, cols, mines):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.board = [[Cell(r, c) for c in range(cols)] for r in range(rows)]
        self.state = "waiting"
        self.start_time = None
        self.elapsed = 0
    def place_mines(self, safe_r, safe_c):
        safe_zone = {(r, c) for r in range(safe_r-1, safe_r+2) for c in range(safe_c-1, safe_c+2) if 0 <= r < self.rows and 0 <= c < self.cols}
        available = [(r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in safe_zone]
        mines_positions = random.sample(available, self.mines)
        for r, c in mines_positions:
            self.board[r][c].is_mine = True
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c].is_mine:
                    continue
                count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc].is_mine:
                            count += 1
                self.board[r][c].adjacent = count
    def reveal_cell(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return
        cell = self.board[r][c]
        if cell.revealed or cell.flagged:
            return
        cell.revealed = True
        if cell.adjacent == 0 and not cell.is_mine:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    self.reveal_cell(r+dr, c+dc)
    def toggle_flag(self, r, c):
        cell = self.board[r][c]
        if not cell.revealed:
            cell.flagged = not cell.flagged
    def reveal_all(self):
        for row in self.board:
            for cell in row:
                cell.revealed = True
    def check_win(self):
        for row in self.board:
            for cell in row:
                if not cell.is_mine and not cell.revealed:
                    return False
        return True
    def update(self):
        if self.state == "playing":
            self.elapsed = time.time() - self.start_time
    def draw(self, surf):
        pygame.draw.rect(surf, (192,192,192), (0,0,WIDTH,UI_HEIGHT))
        flags = sum(1 for row in self.board for cell in row if cell.flagged)
        mines_left = self.mines - flags
        txt1 = font.render("Mines: " + str(mines_left), True, (0,0,0))
        surf.blit(txt1, (10, 10))
        txt2 = font.render("Time: " + str(int(self.elapsed)), True, (0,0,0))
        surf.blit(txt2, (WIDTH - txt2.get_width() - 10, 10))
        reset_btn = pygame.Rect(WIDTH//2 - 25, 5, 50, 40)
        pygame.draw.rect(surf, (220,220,220), reset_btn)
        pygame.draw.rect(surf, (0,0,0), reset_btn, 2)
        face = ":)" if self.state in ("waiting", "playing") else ":(" if self.state=="lost" else "B)"
        txt_face = font.render(face, True, (0,0,0))
        surf.blit(txt_face, (reset_btn.centerx - txt_face.get_width()//2, reset_btn.centery - txt_face.get_height()//2))
        if self.state == "lost":
            msg = large_font.render("Game Over", True, (255,0,0))
            surf.blit(msg, (WIDTH//2 - msg.get_width()//2, UI_HEIGHT + 10))
        elif self.state == "won":
            msg = large_font.render("You Win!", True, (0,255,0))
            surf.blit(msg, (WIDTH//2 - msg.get_width()//2, UI_HEIGHT + 10))
        for row in self.board:
            for cell in row:
                cell.draw(surf, show_mine=(self.state=="lost"))
        return reset_btn
    def reset(self):
        self.board = [[Cell(r, c) for c in range(self.cols)] for r in range(self.rows)]
        self.state = "waiting"
        self.start_time = None
        self.elapsed = 0
game = MinesweeperGame(ROWS, COLS, MINES_COUNT)
while True:
    clock.tick(60)
    game.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if my < UI_HEIGHT:
                reset_btn = pygame.Rect(WIDTH//2 - 25, 5, 50, 40)
                if reset_btn.collidepoint(mx, my):
                    game.reset()
                continue
            c = mx // CELL_SIZE
            r = (my - UI_HEIGHT) // CELL_SIZE
            if event.button == 1:
                if game.state == "waiting":
                    game.place_mines(r, c)
                    game.start_time = time.time()
                    game.state = "playing"
                if game.state == "playing":
                    cell = game.board[r][c]
                    if not cell.flagged:
                        if cell.is_mine:
                            cell.revealed = True
                            game.state = "lost"
                            game.reveal_all()
                        else:
                            game.reveal_cell(r, c)
                            if game.check_win():
                                game.state = "won"
                                game.reveal_all()
            elif event.button == 3:
                if game.state in ("waiting", "playing"):
                    game.toggle_flag(r, c)
    screen.fill((150,150,150))
    reset_button = game.draw(screen)
    pygame.display.flip()
