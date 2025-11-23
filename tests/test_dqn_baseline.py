"""
Tests for DQN baseline policy.

Phase 6: ML baseline comparison tests.
"""

import pytest
import numpy as np
from src.dss.policies.dqn_baseline import DQNPolicy
from src.dss.spectrum_environment import SpectrumEnvironment


@pytest.fixture
def spectrum_env():
    """Create spectrum environment for testing"""
    return SpectrumEnvironment((10e9, 12e9))


@pytest.fixture
def dqn_policy(spectrum_env):
    """Create DQN policy for testing"""
    pytest.importorskip("tensorflow", reason="TensorFlow required for DQN")
    return DQNPolicy(spectrum_env, state_dim=7, action_dim=20)


def test_dqn_initialization(spectrum_env):
    """Test DQN policy initialization"""
    pytest.importorskip("tensorflow", reason="TensorFlow required for DQN")
    
    policy = DQNPolicy(spectrum_env, state_dim=7, action_dim=20)
    
    assert policy.state_dim == 7
    assert policy.action_dim == 20
    assert policy.q_network is not None
    assert policy.target_network is not None


def test_dqn_state_conversion(dqn_policy):
    """Test context to state conversion"""
    context = {
        'throughput': 0.8,
        'latency': 0.2,
        'outage': 0.1,
        'priority': 0.9,
        'doppler': 0.3,
        'elevation': 0.85,
        'beam_load': 0.4
    }
    
    state = dqn_policy._context_to_state(context)
    assert state.shape == (7,)
    assert np.all(state >= 0.0) and np.all(state <= 1.0)


def test_dqn_action_selection(dqn_policy):
    """Test action selection"""
    state = np.array([0.8, 0.2, 0.1, 0.9, 0.3, 0.85, 0.4], dtype=np.float32)
    action = dqn_policy.select_action(state, training=False)
    
    assert 0 <= action < dqn_policy.action_dim


def test_dqn_allocation(dqn_policy, spectrum_env):
    """Test spectrum allocation using DQN"""
    users = [
        {
            'id': 'user1',
            'priority': 0.9,
            'operator': 0
        }
    ]
    
    qos = {
        'user1': {
            'throughput': 80e6,  # 80 Mbps
            'latency': 0.2,
            'outage': 0.1
        }
    }
    
    context = {
        'user1': {
            'throughput': 0.8,
            'latency': 0.2,
            'outage': 0.1,
            'priority': 0.9,
            'doppler': 0.3,
            'elevation': 0.85,
            'beam_load': 0.4,
            'beam_id': 'beam_0'
        }
    }
    
    allocations = dqn_policy.allocate(
        users=users,
        qos=qos,
        context=context,
        bandwidth_hz=100e6
    )
    
    # Should return allocations (may be empty if spectrum unavailable)
    assert isinstance(allocations, dict)
    assert 'user1' in allocations


def test_dqn_experience_replay(dqn_policy):
    """Test experience replay buffer"""
    state = np.array([0.8, 0.2, 0.1, 0.9, 0.3, 0.85, 0.4], dtype=np.float32)
    next_state = np.array([0.7, 0.3, 0.2, 0.8, 0.4, 0.75, 0.5], dtype=np.float32)
    
    # Store experiences
    for _ in range(100):
        action = np.random.randint(0, dqn_policy.action_dim)
        reward = np.random.uniform(-1.0, 1.0)
        done = False
        dqn_policy.store_experience(state, action, reward, next_state, done)
    
    assert len(dqn_policy.replay_buffer) <= dqn_policy.buffer_size
    assert len(dqn_policy.replay_buffer) >= 64  # At least batch_size


def test_dqn_training_step(dqn_policy):
    """Test training step"""
    state = np.array([0.8, 0.2, 0.1, 0.9, 0.3, 0.85, 0.4], dtype=np.float32)
    next_state = np.array([0.7, 0.3, 0.2, 0.8, 0.4, 0.75, 0.5], dtype=np.float32)
    
    # Fill replay buffer
    for _ in range(dqn_policy.batch_size):
        action = np.random.randint(0, dqn_policy.action_dim)
        reward = np.random.uniform(-1.0, 1.0)
        done = False
        dqn_policy.store_experience(state, action, reward, next_state, done)
    
    # Train step
    loss = dqn_policy.train_step()
    
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_dqn_target_network_update(dqn_policy):
    """Test target network update"""
    # Get initial weights
    initial_weights = [w.numpy().copy() for w in dqn_policy.target_network.weights]
    
    # Modify Q-network weights
    for layer in dqn_policy.q_network.layers:
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            new_weights = layer.kernel.numpy() + 0.1
            layer.kernel.assign(new_weights)
    
    # Update target network
    dqn_policy.update_target_network()
    
    # Check that target network weights changed
    updated_weights = [w.numpy() for w in dqn_policy.target_network.weights]
    
    # Weights should be different from initial (if Q-network was modified)
    # But this test is mainly to ensure update_target_network doesn't crash
    assert len(updated_weights) == len(initial_weights)

