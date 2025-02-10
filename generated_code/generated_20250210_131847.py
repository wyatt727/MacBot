import os
import sys
from time import sleep
if __name__ == "__main__":
    # Get the current script file path
    script_path = os.path.abspath(__file__)
    # Create a directory for saving the game code if it doesn't exist
    save_dir = os.path.dirname(script_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Define the file path to save the snake game code
    save_file_path = os.path.join(save_dir, "snake.py")
    # Write the snake game code to the file
    with open(save_file_path, 'w') as f:
        f.write("""import os
import sys
from time import sleep
# Snake Game Code Here
print("Welcome to the Snake Game!")
sleep(2)  # Wait for 2 seconds before exiting
""")
    print(f"Game code saved to {save_file_path}")
