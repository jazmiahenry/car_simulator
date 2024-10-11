import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import anthropic
import logging
from typing import List, Dict, Any
from .car_rl_environment import CarRLEnvironment

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMCarRLWrapper(CarRLEnvironment):
    def __init__(self, num_cars=1, time_of_day="12:00", is_rainy=False, is_weekday=True, 
                 agent_prompt="You are an expert driving instructor. Provide guidance to improve the RL agent's driving performance.",
                 llm_call_limit=100, api_key=None):
        super().__init__(num_cars, time_of_day, is_rainy, is_weekday)
        self.agent_prompt = agent_prompt
        self.client = anthropic.Anthropic(api_key=api_key)
        self.llm_call_count = 0
        self.llm_call_limit = llm_call_limit
        self.conversation_history: List[Dict[str, str]] = []

    def reset(self, seed=None):
        self.llm_call_count = 0
        self.conversation_history = []
        return super().reset(seed)

    def step(self, action):
        if self.llm_call_count >= self.llm_call_limit:
            logger.warning("LLM call limit reached, using default RL action")
            return super().step(action)

        observation, reward, terminated, truncated, info = super().step(action)
        
        llm_guidance = self._get_llm_guidance(observation, reward, terminated)
        adjusted_action = self._adjust_action_based_on_guidance(action, llm_guidance)
        
        self.llm_call_count += 1
        return observation, reward, terminated, truncated, info

    def _get_llm_guidance(self, observation, reward, terminated):
        user_message = f"Current state: {observation}, Reward: {reward}, Terminated: {terminated}. What driving advice would you give?"

        messages = self.conversation_history + [
            {"role": "user", "content": user_message},
        ]

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=150,
            system=self.agent_prompt,
            messages=messages
        )

        ai_response = response.content[0].text
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        logger.debug(f"LLM guidance: {ai_response}")
        return ai_response

    def _adjust_action_based_on_guidance(self, action, guidance):
        # Simple rule-based adjustment based on keywords in the guidance
        if "accelerate" in guidance.lower():
            action[0] = min(action[0] + 0.1, 1.0)  # Increase acceleration
        elif "brake" in guidance.lower():
            action[0] = max(action[0] - 0.1, -1.0)  # Decrease acceleration
        elif "turn left" in guidance.lower():
            action[1] = max(action[1] - 0.1, -1.0)  # Increase left steering
        elif "turn right" in guidance.lower():
            action[1] = min(action[1] + 0.1, 1.0)  # Increase right steering
        return action

def make_env(llm_call_limit, api_key):
    """Create and return an instance of the LLMCarRLWrapper."""
    return lambda: LLMCarRLWrapper(num_cars=3, time_of_day="08:00", is_rainy=False, is_weekday=True, 
                                   llm_call_limit=llm_call_limit, api_key=api_key)

def train_and_evaluate(env, total_timesteps=100000, eval_episodes=10):
    # Initialize the PPO agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, 
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2)

    # Train the agent
    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return model, mean_reward

def main():
    api_key = "your_anthropic_api_key_here"  # Replace with your actual API key
    llm_call_limit = 1000  # Adjust this value based on your needs and API limits

    # Create a vectorized environment
    env = DummyVecEnv([make_env(llm_call_limit, api_key)])

    # Train and evaluate the model
    model, mean_reward = train_and_evaluate(env)

    # Save the trained model
    model.save("car_rl_llm_ppo_model")

    print("Training and evaluation completed.")
    print(f"Final mean reward: {mean_reward:.2f}")

if __name__ == "__main__":
    main()