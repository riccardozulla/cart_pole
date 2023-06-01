# cart_pole

Hi everybody, we are Giacomo Ettore Rocco, and Riccardo Zulla. This is just a small application and implementation of the DQN in Python on the gymnasium cart-pole environment. 
At the moment, it works fine and successfully completes the task in less than 100 episodes.

Being that this is just an exercise, we do not save the weights of the neural networks. If you want to see the episodes and the cart pole in action, simply set the render mode to "human". 
If you're more interested in observing the performance, there is a commented part that plots the results.

Epsilon decay strategy: eps halved after every success, the training is not necessary anymore if it constantly succeed.
![image](https://github.com/riccardozulla/cart_pole/assets/40965802/91d8c315-425f-49d7-a473-833767f1b011)

As you can see it constantly succeed after 140 episodes more or less.

Notes: the notable thing of this code is that interacts just one time per action instead of: 
1) communicate the action
2) let the environment perform the action
3) observe the reward

Since we have to save S, A, R, S' on the replay memory it just communicate one time with the environment taking the reward of the previous action, saving in this way the tupla:

![image](https://github.com/riccardozulla/cart_pole/assets/40965802/8333eaf9-e446-4bc9-95ff-4c0e06033485)

In this way it minimizes the communication between the environment and the agent improving the decoupling between them. 

Feel free to contact us for any reason. We appreciate your feedback and suggestions. Enjoy exploring the code!
