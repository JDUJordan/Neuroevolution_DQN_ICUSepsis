def vectorized_fitness(model_state_dict, num_envs=4, num_episodes=1000):
    """
    Fitness evaluation using vectorized environments with correct reward tracking
    """
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.make('Sepsis/ICU-Sepsis-v2') 
        for _ in range(num_envs)
    ])
    
    model = QNetwork(envs)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    start_time = time.perf_counter()
    all_episode_rewards = []
    episodes_completed = 0
    
    while episodes_completed < num_episodes:
        states, infos = envs.reset()
        episode_rewards = np.zeros(num_envs)
        dones = np.zeros(num_envs, dtype=bool)
        
        while not dones.all():
            # Encode states and create action masks
            states_encoded = torch.Tensor(np.eye(envs.single_observation_space.n)[states])
            
            masks = np.zeros((num_envs, envs.single_action_space.n))
            for i, info in enumerate(infos['admissible_actions']):
                masks[i][info] = 1
            action_masks = torch.FloatTensor(masks)
            
            with torch.no_grad():
                q_values = model(states_encoded, action_masks)
                actions = torch.argmax(q_values, dim=1).numpy()
            
            states, rewards, terminateds, truncateds, infos = envs.step(actions)
            
            episode_rewards += rewards
            dones = terminateds | truncateds
        
        # Store completed episode rewards
        all_episode_rewards.extend(episode_rewards)
        episodes_completed += num_envs
        
        if episodes_completed >= num_episodes:
            # Only take the first num_episodes rewards
            all_episode_rewards = all_episode_rewards[:num_episodes]
            break
    
    end_time = time.perf_counter()
    envs.close()
    
    fitness = np.mean(all_episode_rewards)
    print(f"Evaluated {len(all_episode_rewards)} episodes in {end_time - start_time:.2f} seconds")
    return -fitness