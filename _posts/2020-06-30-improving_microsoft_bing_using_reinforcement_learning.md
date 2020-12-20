---
title: Improving Microsoft Bing Using Reinforcement Learning
published: True
keywords: [Reinforcement Learning]
---

## Internship at MSRA

I did an internship at [Microsoft Research Asia (MSRA)](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/) this spring, working directly with the Bing team to improve the performance of their search engine. I was mentored by [Dr. Qi Chen](https://www.microsoft.com/en-us/research/people/cheqi/).

My work was using reinforcement learning to generate match plans, a data structure used by the Inverted Index system to match web documents from a large document set. I proposed a new model that learns from the suboptimal match plans provided by Bing production team to generate match plans with better performance. My model was tested on Microsoft Bing’s simulation environment and is able to improve the quality of search results by over 50%. My proposal has been approved and is being transferred to production.

## Technical Detail

In the pipeline of processing a web search, Bing search engine first finds a large number of candidates, then uses a technology called match plan to select relevant web documents from the candidates.

A match plan is a sequence of match rules. Each match rule consists of a discreet match rule id and continuous stopping quotas.
For each different pair of user input and system state, we need a different match plan.
To generate such match plans, previous interns developed a reinforcement learning algorithm PASAC (Parameterized Action Soft Actor-Critic):

- Action: match plan
- Reward: defined by production team, related to quality of matched document and time spent on executing the plan
- State: system signal of the server that executes the plan, user input
- Network: RNN, input is a sequence of previous states

## My Role

When training our RL agent, we store the state-action pairs in experience replay. My agent always sampled experience from a small area in the reward space. I wanted to accelerate the training process by diversifying the samples. I proposed and implemented Stratified Prioritized Experience Replay (SPER) to replace the random sampling strategy. I also used demonstrations collected from human to accelerate training. I extracted experience from the simulator and modified the sampling rule to address the suboptimal demonstrations. Apart from the above contribution, I also tested the model in two gym environments.

## Challenges

The technologies I used were very specific and lacked enough academic papers to refer to. The performance of the search engine was already good, so it was not easy to improve it. To overcome it, I asked my advisor Dr. Chen to send me related academic papers and listed the advantages and disadvantages of all the techniques. Then, I broke down the project into different small parts and improved one part at one time. Because running every experiment took several hours, I documented the experiments carefully and used multiple git branches for the modification to different parts of the model. Moreover, I reported my progress to my advisor very frequently, so she could point out any problem in it before it was too late. In the end, I was able to finish the project on time and improve the performance a lot.

## Shortcomings

At present, this system only conducts experiments on a simulated production environment. In order to deploy the algorithm in this article online, it is still necessary to further optimize its response time. The design of the reward function in this article is relatively simple, and there is no guarantee that the model effect is better than the goal of manually formulating a matching scheme for all queries. Finally, compared to the double alpha optimization used in the project, there may be applicable and more refined strategies for the exploration in the parameterized action space.
