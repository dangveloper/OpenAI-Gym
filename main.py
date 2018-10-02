'''
        This code will solve CartPole-v0 by using Deep Q Learning.
        Solving means have a mean score of 195 in 100 consecutive
        episodes. 

Step 1. Gather some training data from the environment
        by running the environment and take random actions.
        The training data will consist of the observation/state as 
        input and the taken action as target. Only simulations with
        a score above a specific threshold will be saved in the 
        training data.

Step 2. Train a deep neural network (DNN) with the training data.

Step 3. Step 1 but instead of take random actions the DNN will provide
        the action based on the given observation/state. It has some
        exploration during this stage to avoid overfitting. Do step 2
        again.

Step 4. Do step 3 with as many DNN as demanded to get best result. 

Step 5. Test your finished DNN and hope you get a mean score over 195
        in 100 consecutive episodes.

'''

# Import libraries
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Create an env and choose game
env = gym.make('CartPole-v0')

def get_training_data(n_episodes = 500, threshold = 40, model = None):
    '''
    Collect training data with or without a model
    '''
    
    # [observation, move]
    training_data = []

    # List of scores
    scores = []
    
    for i_episode in range(n_episodes):
        
        # Reset everything before new episode
        observation = env.reset()
        current_game_data = []
        score = 0
        done = False

        while not done:
            
            #Render the first episodes
            #if i_episode < 10:
                #env.render()
            
            # Choose an action from trained model or random action
            if model == None:
                action = env.action_space.sample()
            else:
    
                action = model.predict_classes(observation.reshape(1,-1))[0][0]
                
                # To reduce the overfitting
                random = np.random.rand()
                if random < 0.05:
                    action = env.action_space.sample()

            # Save the observation and action
            if len(observation):
                current_game_data.append([observation, action])

            # Execute action
            observation, reward, done, _ = env.step(action)
            score += reward

        # If the episode was good enough save all the game data
        if score >= threshold:
            training_data.extend(current_game_data)

        # Save the training score to see improvements
        scores.append(score)
    
    print(np.mean(scores))
    return training_data


def create_neural_net(training_data, epochs = 3):
    
    # Setup the training data
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data])
   

    #Create the model
    model = Sequential()
    model.add(Dense(128, input_dim=len(training_data[0][0]), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Sigmoid because it is binary
    model.add(Dense(1, activation='sigmoid'))

    # Use adam cause it is nice
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Fit model to training data
    model.fit(np.array(X), np.array(y), epochs=epochs, verbose=2)        

    return model


def test_agent(n_episodes = 100, model = None):
    
    scores = []

    for i_episode in range(n_episodes):
        
        # Reset everything before new episode
        observation = env.reset()
        score = 0
        done = False
        while not done:
            
            #if i_episode < 10:
                #env.render()
                
            # Must reshape to fit the expected input
            action = model.predict_classes(observation.reshape(1,-1))[0][0]

            # Execute action
            observation, reward, done, info = env.step(action)
            
            score += reward

            if done:
                scores.append(score)
        
    print(np.mean(scores))

def main():

    # Setup number of episodes/threshold/epochs for the training
    # and data gathering
    n_episodes_threshold_epochs = np.array([[1000, 40, 3],
                                            [500, 130, 2],
                                            [400, 200, 3]])
    
    # Initialize the model
    model = None

    for data in n_episodes_threshold_epochs:
        
        # Gather training data
        training_data = get_training_data(n_episodes=data[0],threshold=data[1],model=model)

        # Train the model, inefficient to create a new model each loop... easy fix
        model = create_neural_net(training_data, data[2])

    # Test the final agent
    test_agent(model = model)

    # Close env to avoid error message
    env.close()


if __name__ == '__main__':
    main()