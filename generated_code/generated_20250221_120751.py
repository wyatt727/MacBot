import pygame, sys, random, time
pygame.init()
CELL_SIZE = 30
ROWS, COLS = 16, 16
MINES = 40
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE + 50
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Minesweeper")
font = pygame.font.SysFont("comicsans", 20)
large_font = pygame.font.SysFont("comicsans", 40)
clock = pygame.time.Clock()
class Cell:
    def __init__(self, r, c):
        self.row = r
        self.col = c
        self.rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE + 50, CELL_SIZE, CELL_SIZE)
        self.is_mine = False
        self.revealed = False
        self.flagged = False
        self.adjacent = 0
    def draw(self, surf):
        if self.revealed:
            pygame.draw.rect(surf, (200,200,200), self.rect)
            if self.is_mine:
                pygame.draw.circle(surf, (0,0,0), self.rect.center, CELL_SIZE // 3)
            elif self.adjacent > 0:
                txt = font.render(str(self.adjacent), True, (0,0,255))
                surf.blit(txt, (self.rect.x + (CELL_SIZE - txt.get_width()) // 2, self.rect.y + (CELL_SIZE - txt.get_height()) // 2))
        else:
            pygame.draw.rect(surf, (100,100,100), self.rect)
            if self.flagged:
                pygame.draw.polygon(surf, (255,0,0), [(self.rect.x + CELL_SIZE // 2, self.rect.y + 5), (self.rect.x + 5, self.rect.y + CELL_SIZE - 5), (self.rect.x + CELL_SIZE - 5, self.rect.y + CELL_SIZE - 5)])
        pygame.draw.rect(surf, (0,0,0), self.rect, 1)
def create_board():
    board = [[Cell(r, c) for c in range(COLS)] for r in range(ROWS)]
    mines_placed = 0
    while mines_placed < MINES:
        r = random.randrange(ROWS)
        c = random.randrange(COLS)
        if not board[r][c].is_mine:
            board[r][c].is_mine = True
            mines_placed += 1
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c].is_mine:
                continue
            count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and board[nr][nc].is_mine:
                        count += 1
            board[r][c].adjacent = count
    return board
def reveal_cell(board, r, c):
    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return
    cell = board[r][c]
    if cell.revealed or cell.flagged:
        return
    cell.revealed = True
    if cell.adjacent == 0 and not cell.is_mine:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                reveal_cell(board, r + dr, c + dc)
def reveal_all(board):
    for row in board:
        for cell in row:
            cell.revealed = True
def check_win(board):
    for row in board:
        for cell in row:
            if not cell.is_mine and not cell.revealed:
                return False
    return True
def draw_ui(surf, mines_left, elapsed, state):
    pygame.draw.rect(surf, (192,192,192), (0,0,WIDTH,50))
    txt1 = font.render("Mines: " + str(mines_left), True, (0,0,0))
    surf.blit(txt1, (10, 10))
    txt2 = font.render("Time: " + str(int(elapsed)), True, (0,0,0))
    surf.blit(txt2, (WIDTH - txt2.get_width() - 10, 10))
    if state == "lost":
        msg = large_font.render("Game Over", True, (255,0,0))
        surf.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 5))
    elif state == "won":
        msg = large_font.render("You Win!", True, (0,255,0))
        surf.blit(msg, (WIDTH // 2 - msg.get_width() // 2, 5))
    reset_btn = pygame.Rect(WIDTH // 2 - 25, 5, 50, 40)
    pygame.draw.rect(surf, (220,220,220), reset_btn)
    pygame.draw.rect(surf, (0,0,0), reset_btn, 2)
    pygame.draw.circle(surf, (255,255,0), reset_btn.center, 10)
    return reset_btn
def count_flags(board):
    cnt = 0
    for row in board:
        for cell in row:
            if cell.flagged:
                cnt += 1
    return cnt
board = create_board()
game_state = "playing"
start_time = time.time()
elapsed = 0
while True:
    clock.tick(60)
    if game_state == "playing":
        elapsed = time.time() - start_time
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if game_state == "playing":
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if my < 50:
                    pass
                else:
                    c = mx // CELL_SIZE
                    r = (my - 50) // CELL_SIZE
                    if event.button == 1:
                        cell = board[r][c]
                        if not cell.flagged:
                            cell.revealed = True
                            if cell.is_mine:
                                game_state = "lost"
                                reveal_all(board)
                            elif cell.adjacent == 0:
                                reveal_cell(board, r, c)
                            if check_win(board):
                                game_state = "won"
                                reveal_all(board)
                    elif event.button == 3:
                        cell = board[r][c]
                        if not cell.revealed:
                            cell.flagged = not cell.flagged
        else:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if WIDTH // 2 - 25 <= mx <= WIDTH // 2 + 25 and 5 <= my <= 45:
                    board = create_board()
                    game_state = "playing"
                    start_time = time.time()
                    elapsed = 0
    screen.fill((150,150,150))
    reset_button = draw_ui(screen, MINES - count_flags(board), elapsed, game_state)
    for r in range(ROWS):
        for c in range(COLS):
            board[r][c].draw(screen)
    pygame.display.flip()
