
import random

def get_user_choice():
	choices = ['rock', 'paper', 'scissors', 'end']
	user_input = input("Enter rock, paper, scissors, or 'end' to quit: ").strip().lower()
	while user_input not in choices:
		print("Invalid choice. Please try again.")
		user_input = input("Enter rock, paper, scissors, or 'end' to quit: ").strip().lower()
	return user_input

def get_computer_choice():
	return random.choice(['rock', 'paper', 'scissors'])

def determine_winner(user, computer):
	if user == computer:
		return "It's a tie!"
	elif (
		(user == 'rock' and computer == 'scissors') or
		(user == 'scissors' and computer == 'paper') or
		(user == 'paper' and computer == 'rock')
	):
		return "You win!"
	else:
		return "Computer wins!"

def play():
	print("Welcome to Rock, Paper, Scissors!")
	user_score = 0
	computer_score = 0
	tie_score = 0
	while True:
		user = get_user_choice()
		if user == 'end':
			print("Thanks for playing!")
			print(f"Final Score: You: {user_score} | Computer: {computer_score} | Ties: {tie_score}")
			break
		computer = get_computer_choice()
		print(f"You chose: {user}")
		print(f"Computer chose: {computer}")
		result = determine_winner(user, computer)
		print(result)
		if result == "You win!":
			user_score += 1
		elif result == "Computer wins!":
			computer_score += 1
		else:
			tie_score += 1

if __name__ == "__main__":
	play()

