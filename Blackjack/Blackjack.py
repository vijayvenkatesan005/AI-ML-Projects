import util, math, random
from collections import defaultdict
from util import ValueIteration

#Vijay Venkatesan

############################################################
# Problem 2.2

# If you decide 2.2 is true, prove it in your submission and put "return None" for
# the code blocks below.  If you decide that 2.2 is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE

        return 0
        
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE 

        actions = []

        actions.append('a1')
        actions.append('a2')

        return actions
        
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE 

        reachableStates = []

        if state != -2 and state != 2:

            if action == 'a1':

                probability_state_minus_one = 0.6
                probability_state_plus_one = 0.4

                if state == -1:
                    
                    newState = -2
                    reward = 20

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_minus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                    newState = 0
                    reward = -5

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_plus_one)
                    successorState.append(reward)

                    reachableStates.append(successorState)

                elif state == 0:

                    reward = -5

                    newState = -1

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_minus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                    newState = 1

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_plus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                else:

                    newState = 0
                    reward = -5

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_minus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                    newState = 2
                    reward = 100

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_plus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

            else:

                probability_state_minus_one = 0.45
                probability_state_plus_one = 0.55

                if state == -1:

                    newState = -2
                    reward = 20

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_minus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                    newState = 0
                    reward = -5

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_plus_one)
                    successorState.append(reward)

                    reachableStates.append(successorState)

                elif state == 0:

                    reward = -5

                    newState = -1

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_minus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                    newState = 1

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_plus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                else:

                    newState = 0
                    reward = -5

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_minus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

                    newState = 2
                    reward = 100

                    successorState = []
                    successorState.append(newState)
                    successorState.append(probability_state_plus_one)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    reachableStates.append(successorState)

        return reachableStates
        
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE 

        return 1

        # END_YOUR_CODE

############################################################
# Problem 3

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE 

        successors_list = list()

        deckCardCounts = state[2]

        if action == 'Take':

            currentTotalCardValueInHand = state[0]

            if deckCardCounts != None:

                peeked = state[1] != None

                if peeked:

                    peekIndex = state[1]

                    if (currentTotalCardValueInHand +
                        self.cardValues[peekIndex] <= self.threshold):

                        probability = 1
                        reward = 0

                        newState = []
                        successorState = []

                        updatedTotalCardValueInHand = (currentTotalCardValueInHand 
                        + self.cardValues[peekIndex])

                        newState.append(updatedTotalCardValueInHand)

                        nextCardIndexIfPeeked = None

                        newState.append(nextCardIndexIfPeeked)

                        updatedDeckCardCounts = list(deckCardCounts)
                        updatedDeckCardCounts[peekIndex] -= 1
                        updatedDeckCardCounts = tuple(updatedDeckCardCounts)

                        emptyDeck = sum(updatedDeckCardCounts) == 0

                        if emptyDeck:

                            updatedDeckCardCounts = None
                            newState.append(updatedDeckCardCounts)

                            reward = updatedTotalCardValueInHand

                            newState = tuple(newState)

                            successorState.append(newState)
                            successorState.append(probability)
                            successorState.append(reward)
                        
                        else:

                            newState.append(updatedDeckCardCounts)

                            newState = tuple(newState)

                            successorState.append(newState)
                            successorState.append(probability)
                            successorState.append(reward)

                        successorState = tuple(successorState)

                        successors_list.append(successorState)
                        
                    else:

                        probability = 1
                        reward = 0

                        newState = []
                        successorState = []

                        updatedTotalCardValueInHand = (currentTotalCardValueInHand 
                        + self.cardValues[peekIndex])

                        newState.append(updatedTotalCardValueInHand)

                        nextCardIndexIfPeeked = None

                        newState.append(nextCardIndexIfPeeked)

                        updatedDeckCardCounts = None

                        newState.append(updatedDeckCardCounts)

                        newState = tuple(newState)

                        successorState.append(newState)
                        successorState.append(probability)
                        successorState.append(reward)

                        successorState = tuple(successorState)

                        successors_list.append(successorState)
                else:

                    cardValueIndex = 0

                    currentTotalCardValueInHand = state[0]

                    reward = 0

                    totalNumberofCardsInDeck = sum(deckCardCounts)
                    
                    for cardValueCount in deckCardCounts:

                        if cardValueCount >= 1:

                            if (currentTotalCardValueInHand +
                                self.cardValues[cardValueIndex] <= self.threshold):
                                
                                newState = []
                                successorState = []

                                updatedTotalCardValueInHand = (currentTotalCardValueInHand 
                                + self.cardValues[cardValueIndex])

                                newState.append(updatedTotalCardValueInHand)

                                nextCardIndexIfPeeked = None

                                newState.append(nextCardIndexIfPeeked)

                                countOfCurrentCardValue = deckCardCounts[cardValueIndex]

                                probability = countOfCurrentCardValue / totalNumberofCardsInDeck


                                updatedDeckCardCounts = list(deckCardCounts)
                                updatedDeckCardCounts[cardValueIndex] -= 1
                                updatedDeckCardCounts = tuple(updatedDeckCardCounts)

                                emptyDeck = sum(updatedDeckCardCounts) == 0

                                if emptyDeck:

                                    updatedDeckCardCounts = None
                                    newState.append(updatedDeckCardCounts)

                                    reward = updatedTotalCardValueInHand

                                    newState = tuple(newState)

                                    successorState.append(newState)
                                    successorState.append(probability)
                                    successorState.append(reward)

                                else:

                                    newState.append(updatedDeckCardCounts)

                                    newState = tuple(newState)

                                    successorState.append(newState)
                                    successorState.append(probability)
                                    successorState.append(reward)

                                successorState = tuple(successorState)

                                successors_list.append(successorState)

                            else:

                                newState = []
                                successorState = []

                                updatedTotalCardValueInHand = (currentTotalCardValueInHand 
                                + self.cardValues[cardValueIndex])

                                newState.append(updatedTotalCardValueInHand)

                                nextCardIndexIfPeeked = None

                                newState.append(nextCardIndexIfPeeked)

                                countOfCurrentCardValue = deckCardCounts[cardValueIndex]

                                probability = countOfCurrentCardValue / totalNumberofCardsInDeck

                                updatedDeckCardCounts = None

                                newState.append(updatedDeckCardCounts)

                                newState = tuple(newState)

                                successorState.append(newState)
                                successorState.append(probability)
                                successorState.append(reward)

                                successorState = tuple(successorState)

                                successors_list.append(successorState)

                        cardValueIndex += 1

        elif action == 'Peek':
            
            peekedInPreviousRound = state[1] != None

            if deckCardCounts != None:

                if not peekedInPreviousRound:

                    cardValueIndex = 0

                    totalNumberofCardsInDeck = sum(deckCardCounts)

                    reward = -1 * self.peekCost

                    for cardValueCount in deckCardCounts:

                        if cardValueCount >= 1:

                            newState = []

                            successorState = []

                            currentTotalCardValueInHand = state[0]

                            newState.append(currentTotalCardValueInHand)

                            nextCardIndexIfPeeked = cardValueIndex

                            newState.append(nextCardIndexIfPeeked)

                            countOfCurrentCardValue = deckCardCounts[cardValueIndex]

                            probability = countOfCurrentCardValue / totalNumberofCardsInDeck

                            currentDeckCardCounts = state[2]
                            
                            newState.append(currentDeckCardCounts)

                            newState = tuple(newState)

                            successorState.append(newState)
                            successorState.append(probability)
                            successorState.append(reward)

                            successorState = tuple(successorState)

                            successors_list.append(successorState)

                        cardValueIndex += 1

        elif action == 'Quit':

            currentTotalCardValueInHand = state[0]

            if deckCardCounts != None:

                if currentTotalCardValueInHand <= self.threshold:

                    newState = []
                    successorState = []

                    probability = 1
                    reward = currentTotalCardValueInHand

                    newState.append(currentTotalCardValueInHand)

                    nextCardIndexIfPeeked = state[1]

                    newState.append(nextCardIndexIfPeeked)

                    updatedDeckCardCounts = None

                    newState.append(updatedDeckCardCounts)

                    newState = tuple(newState)

                    successorState.append(newState)
                    successorState.append(probability)
                    successorState.append(reward)

                    successorState = tuple(successorState)

                    successors_list.append(successorState)
        
        return successors_list
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE 

    cardValues = [5, 3, 2, 1, 25]
    # [5, 3, 2, 1, 25] m = 2 -> exceeds 0.1, passes!

    return BlackjackMDP(cardValues=cardValues, multiplicity=2, threshold=20, peekCost=1)

    # END_YOUR_CODE
