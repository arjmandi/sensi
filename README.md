# Intro
Sensi is an intelligent system that is able to use its senses to perceive and interact with the world around it.
Sensi can receive instructions to do tasks but it might do something else(!)
Becuase Sensi needs to verify if the provided instruction is even possible. Or maybe something else is better than the provided instruction.

# Intuition 
Sensi starts by sensing, which is reading (in future, seeing and listening)
But how can a bot see the world through reading?
What is something that can translate a text into a model? Embedding?
This is important because seeing/modeling is the solution.

So the question is what is the closest compressed model to what I'm seeing. yes

Imagineing how something works, hypothesizing, playing, trying, -> but most importantly seeing 

How much wight we should give to memory, how much to new data? like ax+b, and b as the bias from memory.


To solve arc agi maybe it needs to learn rotation, colors, etc.. what skills does it need to learn?

then it comes the "Ask"

When you want to turn something into a model/analogy/simplify-it it must have certian qualities. in humans it happens a lot that we try to use a model in a wrong setting. 
So not just model building is important but how good you are in re-use or build is important. 
So models must have a quality to them that once you look at them, either you feel you need a new one or you can use them. 

How about a unified design that is happening in sensing through acting. like a turing machine.


So the question is what is the closest compressed model to what I'm seeing. yes

now, what is a model?
you do i1,i2,i3,i4 ... you get o1,o2,o3, o4, ...
model in it's simplest form is a funciton
:
what is a mathematical model for this?
you can compose models

a simple task is one models
a complex task is multiple models
how a smart person can quickly search if there's a path with learned things to a model?

my current hypothesis is that different models will lead to different kind of intelligence.
another abstraction is something like below, a separation of type of models for different usecases:
	•	White-box: choose f,h from physics; estimate \theta by least squares/maximum likelihood on o(t) (sensitivity/adjoint).
	•	Grey-box: part physics, part learned f_\theta (e.g., neural ODE term); train with simulation-in-the-loop.
	•	Black-box: neural ODE / operator learner with regularization to encode invariances (energy, symmetries).
	•	Uncertainty: SDEs, Bayesian posteriors over \theta, ensemble runs.
	•	Control: design i(t) by optimal control/MPC atop the DE model.

up to now, 3 options, transformers, rnn, ann itself (a neuron as a model)
what are our limitations when we talk about a model? how small or big it can be?

so if we're at this point, let's say the llm itself is the model.

ARC-AGI games need timely actions, so LLM must be able to act quickly.
1. one way is to decode LLM outputs in a way that can solve the puzzle, 
2. use another function to solve the puzzle

important tasks are to be able to : we'll give one puzzle to llm and tell him, someone has given this task to a person, as input 

"You are playing Game ID: ls19-fa137e247ce6.

Available Actions:
There are no instructions, intentionally. Play the game to discover controls, rules, and goal.

Press 'Start' to play.
Play to learn the rules of the game.
Win the game.

it must be able to find a way from out to input or vice versa, with actions, store learned actions and try them ..
rewarded path = memory
bigger win, more reward .
pruning the previous paths that don't work is key
we know what is a concept of a game, we need to tell this to the model, what is it to win, to lose, what is the pattern, what is the color, ..
is it needed to teach the model what is a move? what is space, what happens when you move something?

now I have some ideas, 
we give llm hands and eyes to interact with the board, and learn the  ssss

# How Sensi works
1. learn to build a model
2. treat things with your learned models
3. you can compose models , which builds new models

100. intenisify applied models and forget loose models




# Goal
Implement and test against ARC-AGI

# Future
- when we succussfully built a intelligent system  that can learn the skills and apply them, how it can become a sicentist?
how it can achieve to create something like markov chains?

- Sensing can be also seeing and listenting

- For hard problems, you need a clear mind and pure intention to solve them.

--

