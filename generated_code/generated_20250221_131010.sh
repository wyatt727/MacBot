import random

deck = [i for i in range(2, 11) + ['T', 'J', 'Q', 'K', 'A']] * 4
random.shuffle(deck)

class Deck:
    def __init__(self):
        self.cards = deck[:]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal_card(self):
        return self.cards.pop()

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []

    def add_card(self, card):
        self.hand.append(card)

    def get_score(self):
        aces = self.hand.count('A')
        total = sum([10 if card == 'A' else int(card) for card in self.hand])
        while total > 21 and aces:
            total -= 10
            aces -= 1
        return total

    def show_hand(self):
        print("Your hand:", ', '.join([str(card) for card in self.hand]))

class Dealer:
    def __init__(self):
        self.shuffle()
        self.dealer_hand = []

    def deal_dealer_hand(self, n=1):
        for _ in range(n):
            self.dealer_hand.append(random.choice(deck))

    def show_dealer_hand(self):
        print("Dealer's hand:", ', '.join([str(card) for card in self.dealer_hand]))

class Game:
    def __init__(self, player, dealer):
        self.player = player
        self.dealer = dealer

    def play_game(self):
        while True:
            self.player.shuffle()
            self.dealer.shuffle()
            print("Player's turn")
            self.player.show_hand()
            play_input = input().split(' ')
            action = play_input[0]
            if action == "hit":
                card = self.player.deal_card()
                print(f"You drew a {card}")
                self.player.add_card(card)
                print("Your hand:", ', '.join([str(card) for card in self.player.hand]))
            elif action == "stand":
                return
            elif action == "score":
                print(f"{self.player.name}: {self.player.get_score()}")
                break

# Create player and dealer
player = Player("John")
dealer = Dealer()

game = Game(player, dealer)
game.play_game()
