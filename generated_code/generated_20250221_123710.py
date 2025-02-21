import random

def get_high_card():
    return random.randint(1, 13)

def main():
    print("Welcome to High Card Game!")
    player_card = get_high_card()
    bot_card = get_high_card()
    
    print(f"Your card: {player_card}")
    print(f"Bot's card: {bot_card}")
    
    if player_card > bot_card:
        print("You win!")
    elif player_card < bot_card:
        print("You lose!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    main()
