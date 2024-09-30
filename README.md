# Game playing agent for Power Connect-4

##Project Description
This project designs agents playing in Connect-4 game. Agent mainly employ **minimax algorithm** and **minimax with alpha-beta pruning strategy**. The evaluation function adopts an **Awarding-Punishing** mechanism to make a better instruction.

##Program File Clarification
### basic_minimax.py
This is an agent play based on minimax algorithm only.
### basic_alphabeta.py
This is an agent play based on minimax plus alpha-beta pruning.
### connect4_with_minimax.py
This is an agent play based on minimax with improved evaluation mechanism.
### connect4_with_improved_alpha_beta_pruning.py
This is an agent play based on minimax plus alpha-beta pruning with improved evaluation mechanism.
### log(basicWhite VS improvedBlack).txt
This file records the movements generate during the competition of a white agent playing on minimax with a black agent playing on improved_alpha-beta pruning.
### log(improvedWhite VS basicBlack).txt
This file records the movements generate during the competition of a white agent playing on improved_alpha-beta pruning with a black agent playing on minimax.

##Running
Running IMPROVED_WHITE.py by entering "python3 IMPROVED_WHITE.py" in the terminal.
Or running IMPROVED_BLACK.py by entering "python3 IMPROVED_BLACK.py" in the terminal.

The gameID and the colour are defined inside the program in the last row which should be modified if necessary: 
play_game_over_tcp(BLACK, 4, "game37", "black")
play_game_over_tcp(WHITE, 4, "game37", "white")

