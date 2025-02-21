from random import randint

def get_high_card():
    return randint(1, 13)

def main():
    print("High Card Game")
    player_card = get_high_card()
    computer_card = get_high_card()
    
    print(f"Your card: {player_card}")
    print(f"Computer's card: {computer_card}")
    
    if player_card > computer_card:
        print("You win!")
    elif player_card < computer_card:
        print("You lose!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    main()
