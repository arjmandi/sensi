#Sensi

## Intro
Sensi is an agent designed to win the ARC AGI 3

### Design Boundaries: limitations and assumptions
db1: In Sensi we don't want the agent to learn how many game are there. Agent always will start a game must be able to figure out what's going on and then play
db2: in Sensi we train our model to do simple things, like actions, move, etc. (it can also do api calls instead of simple actions which makes the model more general and more complicated)

Rules
r1: if the agent is in tool discovery, it must be able to update the description of the tools it's calling

## Design
1. Sensi must learn what felt good to try and what didn't felt good 
1.1. we couldn't articulate that well enough, while we pushed towards the guesses and informed decisions
2. We will simplify the prompts and structure them to reflect game actions and game state with actions and scores, in case of poor performance in guess development, or figured out items we can make thme more comprehensive again

## Bets
1. We bet on implementing the tool calling ourselves to have more control over the logic of what function to call. we will limit the model output with pydantic or dspy, so we can always get the output correctly. but this bet is something that we're not sure and in case of failure it's better to also try giving the model the control for tool use.
2. we bet on using DSPy to minimize the errors in retrieving the objects from the prompts. it shouldn't for now interfere with our prompting, but later we will use the optimization for DSPy to test the performance


## considerations
- we're ignoring available actions from the function response
- 

## important design questions
- when do we decide to put some part of the logic in code, or in the prompt and the line of reasoning?
  - the way we did it previously was that we've abstracted away how to play the game and kept experiences outside the reasoning. 
  - regarding specifically scores, I'm still not sure how to model it:
    - doesn't give away strong signals of winning but to validate guesses about the game
      - so one design change that comes to mind is break down guesses and figured outs into what each action does? how the game is won? then add the score and monitor the score on level change to add this distinction
-
