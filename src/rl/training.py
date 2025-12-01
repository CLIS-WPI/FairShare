"""
Training pipelines for RL agents.

Supports batch, distributed, and single-run training modes.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.rl.environment import FairnessRLEnvironment
from src.rl.agents.base_agent import BaseRLAgent
from src.rl.agents.ppo_agent import PPOAgent
from src.rl.agents.sac_agent import SACAgent
from src.rl.agents.dqn_agent import DQNAgent


class TrainingPipeline:
    """
    Training pipeline for RL agents.
    
    Handles training, evaluation, and logging.
    """
    
    def __init__(
        self,
        env: FairnessRLEnvironment,
        agent_type: str = "ppo",
        output_dir: str = "models/rl",
        **agent_kwargs
    ):
        """
        Initialize training pipeline.
        
        Args:
            env: RL environment
            agent_type: Type of agent ("ppo", "sac", "dqn")
            output_dir: Output directory for models
            **agent_kwargs: Agent-specific parameters
        """
        self.env = env
        self.agent_type = agent_type.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agent
        if self.agent_type == "ppo":
            self.agent: BaseRLAgent = PPOAgent(env, **agent_kwargs)
        elif self.agent_type == "sac":
            self.agent: BaseRLAgent = SACAgent(env, **agent_kwargs)
        elif self.agent_type == "dqn":
            self.agent: BaseRLAgent = DQNAgent(env, **agent_kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Training history
        self.training_history: List[Dict] = []
    
    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_freq: int = 50000,
        **kwargs
    ):
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_freq: Model save frequency
            **kwargs: Additional training parameters
        """
        print(f"Training {self.agent_type.upper()} agent for {total_timesteps} timesteps...")
        
        # Train in chunks with evaluation
        current_timesteps = 0
        episode = 0
        
        while current_timesteps < total_timesteps:
            # Determine timesteps for this chunk
            chunk_timesteps = min(eval_freq, total_timesteps - current_timesteps)
            
            # Train
            self.agent.train(chunk_timesteps, **kwargs)
            current_timesteps += chunk_timesteps
            
            # Evaluate
            metrics = self.evaluate(n_episodes=10)
            
            # Log
            log_entry = {
                "timestep": current_timesteps,
                "episode": episode,
                "metrics": metrics,
                "timestamp": datetime.now().isoformat()
            }
            self.training_history.append(log_entry)
            
            print(f"Step {current_timesteps}/{total_timesteps}: "
                  f"Reward={metrics['mean_reward']:.2f}, "
                  f"Fairness={metrics['mean_fairness']:.3f}")
            
            # Save checkpoint
            if current_timesteps % save_freq == 0:
                checkpoint_path = self.output_dir / f"{self.agent_type}_checkpoint_{current_timesteps}"
                self.agent.save(str(checkpoint_path))
                print(f"Saved checkpoint: {checkpoint_path}")
            
            episode += 1
        
        # Final save
        final_path = self.output_dir / f"{self.agent_type}_final"
        self.agent.save(str(final_path))
        
        # Save training history
        history_path = self.output_dir / f"{self.agent_type}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Training complete! Model saved to: {final_path}")
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the agent.
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        rewards = []
        fairness_scores = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            if 'fairness_scores' in info:
                fairness_scores.append(np.mean(info['fairness_scores']))
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_fairness": np.mean(fairness_scores) if fairness_scores else 0.0,
            "mean_episode_length": np.mean(episode_lengths)
        }
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.agent.load(path)
    
    def get_training_history(self) -> List[Dict]:
        """Get training history."""
        return self.training_history.copy()

