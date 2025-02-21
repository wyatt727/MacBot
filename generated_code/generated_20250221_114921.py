from tkinter import Tk, Button, Label

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.current_player = "X"
        self.board = [""] * 9
        self.buttons = []
        self.create_board()

    def create_board(self):
        for i in range(9):
            button = Button(self.root, text="", font=('normal', 40), width=5, height=2, command=lambda i=i: self.on_click(i))
            button.grid(row=i // 3, column=i % 3)
            self.buttons.append(button)

    def on_click(self, index):
        if self.board[index] == "" and not self.check_winner():
            self.board[index] = self.current_player
            self.buttons[index].config(text=self.current_player)
            if self.check_winner():
                Label(self.root, text=f"Player {self.current_player} wins!", font=('normal', 15)).grid(row=3, column=0, columnspan=3)
            elif "" not in self.board:
                Label(self.root, text="It's a tie!", font=('normal', 15)).grid(row=3, column=0, columnspan=3)
            else:
                self.current_player = "O" if self.current_player == "X" else "X"

    def check_winner(self):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for condition in win_conditions:
            if self.board[condition[0]] == self.board[condition[1]] == self.board[condition[2]] != "":
                return True
        return False

root = Tk()
root.title("Tic-Tac-Toe")
tic_tac_toe = TicTacToe(root)
root.mainloop()
