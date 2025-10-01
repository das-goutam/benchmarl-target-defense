"""
Target Defense Environment V3 in VMAS
Agents control only heading (direction) via first action dimension and always move at maximum speed
Variable number of attackers and defenders with sensing-based observations
V3: Smart Apollonius-based attacker policy when sensed
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import math
from dataclasses import dataclass

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Y, X

# Try importing analytical solver
try:
    from .apollonius_solver import solve_apollonius_optimization
    APOLLONIUS_AVAILABLE = True
except ImportError:
    try:
        from apollonius_solver import solve_apollonius_optimization
        APOLLONIUS_AVAILABLE = True
    except ImportError:
        APOLLONIUS_AVAILABLE = False
        print("Warning: apollonius_solver not available. Using fallback rewards.")


@dataclass
class TaskConfig:
    """Configuration for Target Defense Smart task"""
    max_steps: int = 1000
    num_defenders: int = 1
    num_attackers: int = 1
    sensing_radius: float = 0.3
    attacker_sensing_radius: float = 0.3
    speed_ratio: float = 0.3
    target_distance: float = 0.05
    randomize_attacker_x: bool = True
    num_spawn_positions: int = 0
    spawn_area_mode: bool = True
    spawn_area_width: float = 0.1
    enable_wall_constraints: bool = False
    wall_epsilon: float = 0.03
    fixed_attacker_policy: bool = False
    use_apollonius: bool = True


def compute_apollonius_circle(pos_a, pos_d, speed_ratio):
    """
    Compute the Apollonius circle for pursuit-evasion game.
    The Apollonius circle is the locus of points P such that |PA|/|PD| = speed_ratio.
    
    Args:
        pos_a: Position of attacker (evader) (np.array)
        pos_d: Position of defender (pursuer) (np.array)
        speed_ratio: Ratio of attacker_speed / defender_speed
    
    Returns:
        center: Center of the circle (np.array)
        radius: Radius of the circle (float)
        lowest_point: Lowest point on the circle (np.array)
    """
    if speed_ratio <= 0 or speed_ratio == 1:
        raise ValueError("Speed ratio must be positive and not equal to 1")
    
    da_vec = pos_a - pos_d
    distance = np.linalg.norm(da_vec)
    if distance < 1e-10:
        return pos_a, 0.0, pos_a
    
    da_unit = da_vec / distance
    k = speed_ratio
    
    if abs(k - 1) < 1e-10:
        raise ValueError("Speed ratio cannot be 1 (creates a line, not a circle)")
    
    # Internal and external division points
    p1 = (pos_a + k * pos_d) / (1 + k)
    p2 = (pos_a - k * pos_d) / (1 - k)
    
    # Center and radius
    center = (p1 + p2) / 2
    radius = np.linalg.norm(p1 - p2) / 2
    
    # Lowest point on the circle
    lowest_point = center - np.array([0, radius])
    
    return center, radius, lowest_point


class Scenario(BaseScenario):
    """
    Target Defense Scenario V3 for VMAS
    Defenders try to sense/intercept attackers before they reach the target line
    V3: Smart attackers use Apollonius circle lowest point when sensed
    
    Action Format:
    - Actions are 2D tensors with shape (batch_size, 2)
    - First dimension: heading angle NORMALIZED to [-1, 1] (maps to [-π, π] radians)
    - Second dimension: ignored (required by VMAS but not used)
    - Agents always move at maximum speed in the specified heading direction
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """
        Create the world with agents and parameters
        
        Args:
            batch_dim: Number of parallel environments
            device: Device to run on (cpu/cuda)
            **kwargs: Additional scenario parameters
        
        Returns:
            World: The created VMAS world
        """
        # Extract parameters from kwargs with defaults
        num_defenders = kwargs.get('num_defenders', 3)
        num_attackers = kwargs.get('num_attackers', 1)
        sensing_radius = kwargs.get('sensing_radius', 0.15)
        attacker_sensing_radius = kwargs.get('attacker_sensing_radius', 0.2)
        speed_ratio = kwargs.get('speed_ratio', 0.7)
        target_distance = kwargs.get('target_distance', 0.05)
        defender_color = kwargs.get('defender_color', (0.0, 0.0, 1.0))
        attacker_color = kwargs.get('attacker_color', (1.0, 0.0, 0.0))
        randomize_attacker_x = kwargs.get('randomize_attacker_x', False)
        fixed_attacker_policy = kwargs.get('fixed_attacker_policy', False)  # Smart policy by default
        num_spawn_positions = kwargs.get('num_spawn_positions', 3)
        max_steps = kwargs.get('max_steps', 200)
        enable_wall_constraints = kwargs.get('enable_wall_constraints', True)
        use_apollonius = kwargs.get('use_apollonius', True)
        spawn_area_mode = kwargs.get('spawn_area_mode', False)
        spawn_area_width = kwargs.get('spawn_area_width', 0.2)
        
        # Generate descriptive run name for WandB
        spawn_type = "area" if spawn_area_mode else "disc"
        run_name = f"{num_defenders}v{num_attackers}_sr{sensing_radius:.1f}_sp{speed_ratio:.1f}_{spawn_type}_smart"
        
        # Print run configuration for logging
        print(f"🎯 TARGET_DEFENSE_SMART: {run_name}")
        print(f"   Config: {num_defenders}v{num_attackers}, sensing={sensing_radius}, speed_ratio={speed_ratio}")
        print(f"   Spawn: {spawn_type}, smart_attacker={not fixed_attacker_policy}")
        
        # Store scenario parameters
        self.batch_dim = batch_dim
        self.device = device
        self.num_defenders = num_defenders
        self.run_name = run_name
        self.num_attackers = num_attackers
        self.sensing_radius = sensing_radius
        self.attacker_sensing_radius = attacker_sensing_radius
        self.capture_distance = 0.07
        self.speed_ratio = speed_ratio
        self.target_distance = target_distance
        self.randomize_attacker_x = randomize_attacker_x
        self.fixed_attacker_policy = fixed_attacker_policy
        self.num_spawn_positions = num_spawn_positions
        self.max_steps = max_steps
        self.spawn_area_mode = spawn_area_mode
        self.spawn_area_width = spawn_area_width
        
        # Speed settings
        self.defender_max_speed = 0.05
        self.attacker_max_speed = self.defender_max_speed * self.speed_ratio
        
        # Near-wall constraint controls
        self.enable_wall_constraints = bool(enable_wall_constraints)
        # self.wall_epsilon = float(wall_epsilon)
        
        # Apollonius solver controls
        self.use_apollonius = bool(use_apollonius) and APOLLONIUS_AVAILABLE
        
        # Create world - 1x1 space from 0 to 1
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=0.5,
            y_semidim=0.5,
            collision_force=0,
            substeps=1,
            dt=1.0
        )
        
        # Store world bounds for coordinate transformation
        self.world_min = 0.0
        self.world_max = 1.0
        
        # Create defender agents
        for i in range(num_defenders):
            agent = Agent(
                name=f"defender_{i}",
                shape=Sphere(radius=0.02),
                color=defender_color,
                max_speed=self.defender_max_speed,
                rotatable=False,
                silent=True
            )
            agent.is_defender = True
            agent.sensing_radius = sensing_radius
            world.add_agent(agent)
        
        # Create attacker agents (non-controllable in smart version when using smart policy)
        for i in range(num_attackers):
            agent = Agent(
                name=f"attacker_{i}",
                shape=Sphere(radius=0.02),
                color=attacker_color,
                max_speed=self.attacker_max_speed,
                rotatable=False,
                silent=True,
                controllable=False  # Mark as non-controllable for smart policy
            )
            agent.is_defender = False
            agent.sensing_radius = attacker_sensing_radius
            world.add_agent(agent)
        
        # Initialize tracking variables
        self.attacker_sensed = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_captured = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_intercepted = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_reached_target = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_sensing_rewards = torch.zeros((batch_dim, num_attackers), device=device)
        self.defender_has_sensed = torch.zeros((batch_dim, num_defenders), dtype=torch.bool, device=device)
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        
        # Trajectory tracking for visualization
        self.max_trajectory_length = 50
        self.agent_trajectories = {}
        
        return world
    
    def _world_to_vmas(self, coord):
        """Convert world coordinates [0,1] to VMAS coordinates [-0.5,0.5]"""
        if isinstance(coord, np.ndarray):
            return coord - 0.5
        return coord - 0.5
    
    def _vmas_to_world(self, coord):
        """Convert VMAS coordinates [-0.5,0.5] to world coordinates [0,1]"""
        if isinstance(coord, np.ndarray):
            return coord + 0.5
        return coord + 0.5
    
    def reset_world_at(self, env_index: Optional[int] = None):
        """
        Reset world to initial positions
        
        Args:
            env_index: Index of the environment to reset (for vectorized envs)
        """
        # Get defenders and attackers
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        # Position defenders evenly along bottom edge
        defender_spacing = 1.0 / (self.num_defenders + 1)
        for i, defender in enumerate(defenders):
            world_x = (i + 1) * defender_spacing
            vmas_x = self._world_to_vmas(world_x)
            vmas_y = self._world_to_vmas(0.0)
            
            if env_index is None:
                defender.state.pos[:, X] = vmas_x
                defender.state.pos[:, Y] = vmas_y
                defender.state.vel[:, :] = 0
            else:
                defender.state.pos[env_index, X] = vmas_x
                defender.state.pos[env_index, Y] = vmas_y
                defender.state.vel[env_index, :] = 0
        
        # Position attackers based on spawn mode
        if self.spawn_area_mode:
            # Area-based spawning
            if env_index is None:
                batch_size = attackers[0].state.pos.shape[0] if attackers else self.batch_dim
                
                for env_idx in range(batch_size):
                    for i, attacker in enumerate(attackers):
                        random_x_world = torch.rand(1).item()
                        random_x_vmas = self._world_to_vmas(random_x_world)
                        
                        spawn_area_min_y = 1.0 - self.spawn_area_width
                        random_y_world = spawn_area_min_y + torch.rand(1).item() * self.spawn_area_width
                        random_y_vmas = self._world_to_vmas(random_y_world)
                        
                        attacker.state.pos[env_idx, X] = random_x_vmas
                        attacker.state.pos[env_idx, Y] = random_y_vmas
                        attacker.state.vel[env_idx, :] = 0
            else:
                for i, attacker in enumerate(attackers):
                    random_x_world = torch.rand(1).item()
                    random_x_vmas = self._world_to_vmas(random_x_world)
                    
                    spawn_area_min_y = 1.0 - self.spawn_area_width
                    random_y_world = spawn_area_min_y + torch.rand(1).item() * self.spawn_area_width
                    random_y_vmas = self._world_to_vmas(random_y_world)
                    
                    attacker.state.pos[env_index, X] = random_x_vmas
                    attacker.state.pos[env_index, Y] = random_y_vmas
                    attacker.state.vel[env_index, :] = 0
        elif self.randomize_attacker_x:
            # Discrete spawn positions
            if self.num_spawn_positions == 1:
                world_spawn_positions = torch.tensor([0.5], device=self.device)
            else:
                spacing = 0.8 / (self.num_spawn_positions - 1)
                world_spawn_positions = torch.tensor(
                    [0.1 + i * spacing for i in range(self.num_spawn_positions)],
                    device=self.device
                )
            spawn_positions = self._world_to_vmas(world_spawn_positions)
            
            if env_index is None:
                batch_size = attackers[0].state.pos.shape[0] if attackers else self.batch_dim
                
                for env_idx in range(batch_size):
                    available_positions = spawn_positions.clone()
                    
                    if self.num_attackers > self.num_spawn_positions:
                        repeats = (self.num_attackers // self.num_spawn_positions) + 1
                        available_positions = available_positions.repeat(repeats)[:self.num_attackers]
                        perm = torch.randperm(len(available_positions))
                        available_positions = available_positions[perm]
                    else:
                        perm = torch.randperm(len(available_positions))[:self.num_attackers]
                        available_positions = available_positions[perm]
                    
                    for i, attacker in enumerate(attackers):
                        if i < len(available_positions):
                            attacker.state.pos[env_idx, X] = available_positions[i]
                        else:
                            attacker.state.pos[env_idx, X] = 0.0
                        
                        vmas_y = self._world_to_vmas(1.0)
                        attacker.state.pos[env_idx, Y] = vmas_y
                        attacker.state.vel[env_idx, :] = 0
            else:
                available_positions = spawn_positions.clone()
                
                if self.num_attackers > self.num_spawn_positions:
                    repeats = (self.num_attackers // self.num_spawn_positions) + 1
                    available_positions = available_positions.repeat(repeats)[:self.num_attackers]
                    perm = torch.randperm(len(available_positions))
                    available_positions = available_positions[perm]
                else:
                    perm = torch.randperm(len(available_positions))[:self.num_attackers]
                    available_positions = available_positions[perm]
                
                for i, attacker in enumerate(attackers):
                    if i < len(available_positions):
                        attacker.state.pos[env_index, X] = available_positions[i]
                    else:
                        attacker.state.pos[env_index, X] = 0.0
                    
                    vmas_y = self._world_to_vmas(1.0)
                    attacker.state.pos[env_index, Y] = vmas_y
                    attacker.state.vel[env_index, :] = 0
        else:
            # Fixed center position
            for i, attacker in enumerate(attackers):
                world_x = 0.5
                vmas_x = self._world_to_vmas(world_x)
                vmas_y = self._world_to_vmas(1.0)
                
                if env_index is None:
                    attacker.state.pos[:, X] = vmas_x
                    attacker.state.pos[:, Y] = vmas_y
                    attacker.state.vel[:, :] = 0
                else:
                    attacker.state.pos[env_index, X] = vmas_x
                    attacker.state.pos[env_index, Y] = vmas_y
                    attacker.state.vel[env_index, :] = 0
        
        # Reset episode tracking
        if env_index is None:
            batch_size = self.batch_dim if hasattr(self, 'batch_dim') else self.world.batch_dim
            device = self.device if hasattr(self, 'device') else self.world.device
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_captured = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_intercepted = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
            self.step_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            self.attacker_sensed[env_index, :] = False
            self.attacker_captured[env_index, :] = False
            self.attacker_intercepted[env_index, :] = False
            self.attacker_reached_target[env_index, :] = False
            self.attacker_sensing_rewards[env_index, :] = 0.0
            if hasattr(self, 'defender_has_sensed'):
                self.defender_has_sensed[env_index, :] = False
            if hasattr(self, 'distances'):
                self.distances[env_index, :, :] = 0.0
            if hasattr(self, 'step_count'):
                self.step_count[env_index] = 0
        
        # Reset step-based flags
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
        
        # Reset trajectories
        if hasattr(self, 'agent_trajectories'):
            for agent in self.world.agents:
                self.agent_trajectories[agent.name] = []
    
    def reward(self, agent: Agent) -> torch.Tensor:
        """Calculate reward for an agent - SMART VERSION"""
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        self.update_events()
        
        done_mask = self.done()
        r = torch.zeros(batch_size, device=device)
        
        if agent.is_defender:
            if done_mask.any():
                for env_idx in range(batch_size):
                    env_total_reward = 0.0
                    
                    if done_mask[env_idx]:
                        for a_idx in range(self.num_attackers):
                            # SMART: Reward based on CAPTURE (AC value at capture)
                            if self.attacker_captured[env_idx, a_idx]:
                                reward_val = self.attacker_sensing_rewards[env_idx, a_idx]
                                env_total_reward += reward_val
                    
                    r[env_idx] = env_total_reward
        
        return r
    
    def observation(self, agent: Agent) -> torch.Tensor:
        """
        Get observation for an agent with shared visibility logic:
        - If ANY defender has an attacker within sensing radius, ALL defenders see that attacker
        - If attacker leaves ALL sensing radii, position information is lost for ALL defenders
        """
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        if agent.is_defender:
            obs_dim = 2 + (self.num_defenders - 1) * 2 + self.num_attackers * 2
        else:
            obs_dim = 2 + self.num_defenders * 2
        
        obs = torch.zeros(batch_size, obs_dim, device=device)
        
        defenders = sorted([a for a in self.world.agents if a.is_defender], 
                          key=lambda x: x.name)
        attackers = sorted([a for a in self.world.agents if not a.is_defender], 
                          key=lambda x: x.name)
        
        idx = 0
        
        # 1. Own position (absolute coordinates)
        obs[:, idx:idx+2] = agent.state.pos
        idx += 2
        
        if agent.is_defender:
            # 2. Other defenders' relative positions (for coordination)
            for other_defender in defenders:
                if other_defender != agent:
                    relative_pos = other_defender.state.pos - agent.state.pos
                    obs[:, idx:idx+2] = relative_pos
                    idx += 2
            
            # 3. Attackers with SHARED VISIBILITY
            # Check which attackers are visible to ANY defender
            for attacker_idx, attacker in enumerate(attackers):
                # Initialize visibility mask for this attacker
                attacker_visible = torch.zeros(batch_size, dtype=torch.bool, device=device)
                
                # Check if ANY defender can see this attacker
                for defender in defenders:
                    dist = torch.norm(defender.state.pos - attacker.state.pos, dim=-1)
                    can_see = dist <= defender.sensing_radius
                    # If any defender can see, all defenders get the observation
                    attacker_visible = attacker_visible | can_see
                
                # Set attacker position for all environments where it's visible to ANY defender
                obs[attacker_visible, idx:idx+2] = attacker.state.pos[attacker_visible]
                idx += 2
        else:
            # For attackers: observe all defenders if within their own sensing radius
            for defender in defenders:
                dist = torch.norm(agent.state.pos - defender.state.pos, dim=-1)
                within_range = dist <= agent.sensing_radius
                
                obs[within_range, idx:idx+2] = defender.state.pos[within_range]
                idx += 2
        
        return obs
    
    def update_events(self):
        """Update sensing and capture events with AC rewards - SMART VERSION"""
        if hasattr(self, '_events_updated_this_step') and self._events_updated_this_step:
            return
        
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_captured = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_intercepted = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
        
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        # SMART VERSION: First check sensing (at 0.3 radius)
        for attacker_idx, attacker in enumerate(attackers):
            for defender_idx, defender in enumerate(defenders):
                dist = torch.norm(attacker.state.pos - defender.state.pos, dim=-1)
                self.distances[:, defender_idx, attacker_idx] = dist
                
                # SENSING EVENT (at sensing_radius)
                newly_sensed = (dist <= self.sensing_radius) & ~self.attacker_sensed[:, attacker_idx]
                
                if newly_sensed.any():
                    self.defender_has_sensed[:, defender_idx] |= newly_sensed
                    
                    for env_idx in torch.where(newly_sensed)[0]:
                        def_pos = defender.state.pos[env_idx]
                        att_pos = attacker.state.pos[env_idx]
                        
                        # Snap to sensing boundary
                        direction = att_pos - def_pos
                        direction_norm = torch.norm(direction)
                        
                        if direction_norm < self.sensing_radius and direction_norm > 0:
                            direction_normalized = direction / direction_norm
                            attacker.state.pos[env_idx] = def_pos + direction_normalized * self.sensing_radius
                        
                        # Don't stop velocity - smart attacker continues with Nash policy
                    
                    self.attacker_sensed[:, attacker_idx] |= newly_sensed
        
        # SMART VERSION: Check for capture (at capture_distance) - separate from sensing
        for attacker_idx, attacker in enumerate(attackers):
            for defender_idx, defender in enumerate(defenders):
                dist = torch.norm(attacker.state.pos - defender.state.pos, dim=-1)
                
                # CAPTURE EVENT (at capture_distance, only after sensing)
                can_capture = self.attacker_sensed[:, attacker_idx]  # Must be sensed first
                newly_captured = (dist <= self.capture_distance) & can_capture & ~self.attacker_captured[:, attacker_idx]
                
                if newly_captured.any():
                    for env_idx in torch.where(newly_captured)[0]:
                        # COMPUTE AC REWARD AT CAPTURE (Smart version)
                        if self.use_apollonius:
                            attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                            attacker_pos = self._vmas_to_world(attacker_pos_vmas)
                            
                            defender_positions = []
                            for def_agent in defenders:
                                defender_pos_vmas = def_agent.state.pos[env_idx].cpu().numpy()
                                defender_pos_global = self._vmas_to_world(defender_pos_vmas)
                                defender_positions.append(defender_pos_global)
                            
                            result = solve_apollonius_optimization(
                                attacker_pos=attacker_pos,
                                defender_positions=defender_positions,
                                nu=1.0 / self.speed_ratio
                            )
                            
                            if result['success']:
                                # Use AC defender payoff as reward at capture
                                self.attacker_sensing_rewards[env_idx, attacker_idx] = result['defender_payoff']
                                
                                if result['defender_payoff'] > 0:
                                    self.attacker_intercepted[env_idx, attacker_idx] = True
                            else:
                                # Fallback to y-coordinate
                                attacker_y_vmas = attacker.state.pos[env_idx, Y].item()
                                attacker_y_world = self._vmas_to_world(attacker_y_vmas)
                                self.attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
                        else:
                            # Fallback to y-coordinate
                            attacker_y_vmas = attacker.state.pos[env_idx, Y].item()
                            attacker_y_world = self._vmas_to_world(attacker_y_vmas)
                            self.attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
                        
                        # Stop attacker movement on capture
                        attacker.state.vel[env_idx] = torch.zeros_like(attacker.state.vel[env_idx])
                    
                    self.attacker_captured[:, attacker_idx] |= newly_captured
        
        # Check for target reached (only if not captured)
        target_y_vmas = self._world_to_vmas(0.0)
        for attacker_idx, attacker in enumerate(attackers):
            reached = (attacker.state.pos[:, Y] <= target_y_vmas + self.target_distance) & ~self.attacker_captured[:, attacker_idx]
            self.attacker_reached_target[:, attacker_idx] |= reached
        
        self._events_updated_this_step = True
        self._reset_events_flag_next_step = True

    def _compute_smart_attacker_heading(self, attacker_idx, batch_size, device):
        """
        V3: Compute smart attacker heading using Apollonius circle lowest point
        When attacker is sensed by defenders, it computes optimal escape route
        """
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        if attacker_idx >= len(attackers):
            return torch.full((batch_size,), -math.pi/2, device=device)
        
        attacker = attackers[attacker_idx]
        heading = torch.full((batch_size,), -math.pi/2, device=device)  # Default: straight down
        
        for env_idx in range(batch_size):
            # ALWAYS check if sensed first
            is_sensed = False
            if hasattr(self, 'attacker_sensed') and self.attacker_sensed is not None:
                is_sensed = self.attacker_sensed[env_idx, attacker_idx].item()
            
            if not is_sensed:
                # NOT SENSED: Move straight down
                heading[env_idx] = -math.pi/2  # Straight down
            else:
                # SENSED: Use Apollonius circle to find optimal escape route
                try:
                    attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                    attacker_pos_world = self._vmas_to_world(attacker_pos_vmas)
                    
                    # Find all defenders that can currently see this attacker
                    sensing_defenders = []
                    for defender in defenders:
                        defender_pos_vmas = defender.state.pos[env_idx].cpu().numpy()
                        defender_pos_world = self._vmas_to_world(defender_pos_vmas)
                        
                        dist = np.linalg.norm(attacker_pos_world - defender_pos_world)
                        # Include all defenders that have sensed (not just currently in range)
                        if self.attacker_sensed[env_idx, attacker_idx]:
                            sensing_defenders.append(defender_pos_world)
                    
                    if sensing_defenders:
                        lowest_points = []
                        
                        for defender_pos in sensing_defenders:
                            try:
                                center, radius, lowest_point = compute_apollonius_circle(
                                    pos_a=attacker_pos_world,
                                    pos_d=defender_pos,
                                    speed_ratio=self.speed_ratio
                                )
                                lowest_points.append(lowest_point)
                            except (ValueError, ZeroDivisionError):
                                continue
                        
                        if lowest_points:
                            # Choose the lowest point that gives best escape
                            best_point = min(lowest_points, key=lambda p: p[1])  # Minimize y
                            
                            target_vmas = self._world_to_vmas(best_point)
                            attacker_vmas = attacker.state.pos[env_idx].cpu().numpy()
                            
                            direction = target_vmas - attacker_vmas
                            if np.linalg.norm(direction) > 1e-6 and not np.isnan(direction).any():
                                heading_angle = np.arctan2(direction[1], direction[0])
                                heading[env_idx] = torch.tensor(heading_angle, device=device)
                            else:
                                # If at target point or invalid direction, continue down
                                heading[env_idx] = -math.pi/2
                        else:
                            # No valid Apollonius points, move straight down
                            heading[env_idx] = -math.pi/2
                    else:
                        # No defenders sensing, but was sensed before - move down
                        heading[env_idx] = -math.pi/2
                        
                except Exception as e:
                    # On any error, default to straight down
                    print(f"Apollonius computation error: {e}")
                    heading[env_idx] = -math.pi/2
        
        return heading

    def process_action(self, agent: Agent):
        """
        V3: Process agent action with smart Apollonius-based attacker policy
        - Defenders: Use action input (trainable policy)  
        - Attackers: Non-controllable with smart policy
        """
        if hasattr(self, '_reset_events_flag_next_step') and self._reset_events_flag_next_step:
            self._events_updated_this_step = False
            self._step_incremented_this_step = False
            self._reset_events_flag_next_step = False
        
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Process attacker actions with smart policy
        if not agent.is_defender:
            # ATTACKERS: Non-controllable, use smart policy
            attacker_idx = -1
            attackers = [a for a in self.world.agents if not a.is_defender]
            for idx, a in enumerate(attackers):
                if a == agent:
                    attacker_idx = idx
                    break
            
            # Ensure action structure exists for non-controllable agents
            if agent.action is None or not hasattr(agent.action, 'u'):
                from vmas.simulator.core import TorchVectorizedObject
                agent.action = TorchVectorizedObject()
                agent.action.u = torch.zeros((batch_size, 2), device=device)
            elif agent.action.u is None or torch.isnan(agent.action.u).any():
                agent.action.u = torch.zeros((batch_size, 2), device=device)
            
            if attacker_idx >= 0 and hasattr(self, 'attacker_captured'):
                is_inactive = self.attacker_captured[:, attacker_idx] | self.attacker_reached_target[:, attacker_idx]
                
                if self.fixed_attacker_policy:
                    heading = torch.full((batch_size,), -math.pi/2, device=device)
                else:
                    # V3 SMART POLICY: Use Apollonius circle when sensed
                    heading = self._compute_smart_attacker_heading(attacker_idx, batch_size, device)
            else:
                if self.fixed_attacker_policy:
                    heading = torch.full((batch_size,), -math.pi/2, device=device)
                else:
                    heading = self._compute_smart_attacker_heading(attacker_idx, batch_size, device)
        else:
            # DEFENDERS: Use trainable policy
            if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                if torch.isnan(agent.action.u).any():
                    print(f"Warning: NaN detected in defender {agent.name} action, using zero heading")
                    heading = torch.zeros(batch_size, device=device)
                else:
                    normalized_heading = agent.action.u[:, 0]
                    heading = normalized_heading * math.pi
            else:
                heading = torch.zeros(batch_size, device=device)

        # Add NaN safety check for heading
        if torch.isnan(heading).any() or torch.isinf(heading).any():
            print(f"Warning: NaN/Inf heading for {agent.name}, setting to default straight down")
            heading = torch.full((batch_size,), -math.pi/2, device=device)

        theta = torch.remainder(heading, 2 * math.pi)

        if agent.is_defender and hasattr(self, '_apply_wall_constraints'):
            theta = self._apply_wall_constraints(agent, theta)
        
        # Determine speed
        max_speed = self.attacker_max_speed if not agent.is_defender else self.defender_max_speed
        
        # For defenders in Smart environment, they continue moving after sensing to learn pursuit
        if agent.is_defender and hasattr(self, 'defender_has_sensed'):
            # Defenders stay active to learn pursuit behavior
            max_speed = torch.full((batch_size,), max_speed, device=device)
        elif not agent.is_defender and hasattr(self, 'attacker_captured'):
            attacker_idx = -1
            attackers = [a for a in self.world.agents if not a.is_defender]
            for idx, a in enumerate(attackers):
                if a == agent:
                    attacker_idx = idx
                    break
            
            if attacker_idx >= 0:
                is_inactive = self.attacker_captured[:, attacker_idx] | self.attacker_reached_target[:, attacker_idx]
                max_speed = torch.where(
                    is_inactive, 
                    torch.zeros(batch_size, device=device), 
                    torch.full((batch_size,), max_speed, device=device)
                )
        
        self.update_events()
        
        if not hasattr(self, '_step_incremented_this_step') or not self._step_incremented_this_step:
            if hasattr(self, 'step_count'):
                self.step_count += 1
            self._step_incremented_this_step = True
        
        # Set velocity with NaN safety
        velocity_x = max_speed * torch.cos(theta)
        velocity_y = max_speed * torch.sin(theta)
        
        if torch.isnan(velocity_x).any() or torch.isnan(velocity_y).any():
            print(f"Warning: NaN in velocity for {agent.name}, setting to zero")
            velocity_x = torch.zeros_like(velocity_x)
            velocity_y = torch.zeros_like(velocity_y)
        
        agent.action.u[:, 0] = velocity_x
        agent.action.u[:, 1] = velocity_y
        
        # Track trajectory
        if not hasattr(self, '_trajectory_initialized') or not self._trajectory_initialized:
            for a in self.world.agents:
                self.agent_trajectories[a.name] = []
            self._trajectory_initialized = True
        
        current_pos = agent.state.pos[0].clone().cpu().numpy()
        if agent.name not in self.agent_trajectories:
            self.agent_trajectories[agent.name] = []
        
        self.agent_trajectories[agent.name].append(current_pos.copy())
        
        if len(self.agent_trajectories[agent.name]) > self.max_trajectory_length:
            self.agent_trajectories[agent.name] = self.agent_trajectories[agent.name][-self.max_trajectory_length:]
    
    def _apply_wall_constraints(self, agent, theta_0_2pi):
        """Apply state-dependent heading constraints near walls for defenders"""
        if not self.enable_wall_constraints:
            return theta_0_2pi

        x = agent.state.pos[:, 0]
        y = agent.state.pos[:, 1]
        wx = getattr(self.world, "x_semidim", 0.5)
        wy = getattr(self.world, "y_semidim", 0.5)
        eps = float(self.wall_epsilon)

        near_right = (wx - x) <= eps
        near_left = (x + wx) <= eps
        near_top = (wy - y) <= eps
        near_bottom = (y + wy) <= eps

        theta = theta_0_2pi.clone()
        pi = math.pi
        two_pi = 2 * pi

        # Corner masks
        tr = near_top & near_right
        tl = near_top & near_left
        br = near_bottom & near_right
        bl = near_bottom & near_left

        theta = torch.where(tr, torch.clamp(theta, pi, 1.5 * pi), theta)
        theta = torch.where(tl, torch.clamp(theta, 1.5 * pi, two_pi), theta)
        theta = torch.where(br, torch.clamp(theta, 0.5 * pi, pi), theta)
        theta = torch.where(bl, torch.clamp(theta, 0.0, 0.5 * pi), theta)

        return theta
    
    def done(self) -> torch.Tensor:
        """Check if episodes are done - SMART VERSION"""
        if not hasattr(self, 'attacker_captured') or self.attacker_captured is None:
            batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
            device = self.world.device if hasattr(self.world, 'device') else self.device
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # SMART: Episode ends when ALL attackers are either CAPTURED or reached target
        all_attackers_done = (self.attacker_captured | self.attacker_reached_target).all(dim=1)
        
        max_steps_reached = torch.zeros_like(all_attackers_done, dtype=torch.bool)
        if hasattr(self, 'step_count'):
            max_steps_reached = self.step_count >= self.max_steps
        
        return all_attackers_done | max_steps_reached
    
    def info(self, agent: Agent) -> Dict:
        """Get info dictionary for an agent"""
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
            device = self.world.device if hasattr(self.world, 'device') else self.device
            return {
                "attackers_sensed": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_captured": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_intercepted": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_reached_target": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attacker_rewards": torch.zeros((batch_size, self.num_attackers), device=device),
                "sensing_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "capture_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "interception_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "target_reached": torch.zeros(batch_size, dtype=torch.bool, device=device)
            }
        
        return {
            "attackers_sensed": self.attacker_sensed.clone(),
            "attackers_captured": self.attacker_captured.clone(),
            "attackers_intercepted": self.attacker_intercepted.clone(),
            "attackers_reached_target": self.attacker_reached_target.clone(),
            "attacker_rewards": self.attacker_sensing_rewards.clone(),
            "sensing_occurred": self.attacker_sensed.any(dim=1),
            "capture_occurred": self.attacker_captured.any(dim=1),
            "interception_occurred": self.attacker_intercepted.any(dim=1),
            "target_reached": self.attacker_reached_target.any(dim=1)
        }

    def extra_render(self, env_index: int = 0):
        """Enhanced rendering with Apollonius circles and smart attacker visualization"""
        from vmas.simulator import rendering
        from vmas.simulator.utils import Color
        import numpy as np
        
        geoms = []
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        all_agents = defenders + attackers
        
        # 1. BLACK WALLS
        wall_lines = [
            ((-0.5, -0.5), (-0.5, 0.5)),
            ((0.5, -0.5), (0.5, 0.5)),
            ((-0.5, 0.5), (0.5, 0.5)),
        ]
        
        for start, end in wall_lines:
            wall = rendering.Line(start, end, width=8)
            wall_xform = rendering.Transform()
            wall.add_attr(wall_xform)
            wall.set_color(*Color.BLACK.value)
            geoms.append(wall)
        
        # 2. GREEN TARGET LINE
        target_line = rendering.Line((-0.5, -0.5), (0.5, -0.5), width=10)
        target_xform = rendering.Transform()
        target_line.add_attr(target_xform)
        target_line.set_color(*Color.GREEN.value)
        geoms.append(target_line)
        
        # 3. SENSING AND CAPTURE CIRCLES
        for defender in defenders:
            pos = defender.state.pos[env_index]
            
            sensing_circle = rendering.make_circle(self.sensing_radius, filled=False)
            sensing_xform = rendering.Transform()
            sensing_xform.set_translation(*pos.cpu().numpy())
            sensing_circle.add_attr(sensing_xform)
            sensing_circle.set_color(0.0, 0.0, 1.0, 0.6)
            geoms.append(sensing_circle)
            
            capture_circle = rendering.make_circle(self.capture_distance, filled=False)
            capture_xform = rendering.Transform()
            capture_xform.set_translation(*pos.cpu().numpy())
            capture_circle.add_attr(capture_xform)
            capture_circle.set_color(1.0, 0.0, 1.0, 0.8)
            geoms.append(capture_circle)
        
        # 4. SPAWN AREA
        if hasattr(self, 'spawn_area_mode') and self.spawn_area_mode:
            spawn_min_vmas = 0.5 - self.spawn_area_width
            
            spawn_borders = [
                rendering.Line((-0.5, 0.5), (0.5, 0.5), width=6),
                rendering.Line((-0.5, spawn_min_vmas), (0.5, spawn_min_vmas), width=6),
            ]
            for border in spawn_borders:
                border_xform = rendering.Transform()
                border.add_attr(border_xform)
                border.set_color(*Color.RED.value)
                geoms.append(border)
        
        # 5. HEADING ARROWS
        for agent in all_agents:
            if hasattr(agent.state, 'vel'):
                pos = agent.state.pos[env_index]
                vel = agent.state.vel[env_index]
                vel_norm = torch.norm(vel)
                
                if vel_norm > 0.001:
                    direction = vel / vel_norm
                    arrow_length = 0.15
                    end_pos = pos + direction * arrow_length
                    
                    heading_vector = rendering.Line(
                        tuple(pos.cpu().numpy()),
                        tuple(end_pos.cpu().numpy()),
                        width=6
                    )
                    vector_xform = rendering.Transform()
                    heading_vector.add_attr(vector_xform)
                    
                    if agent.is_defender:
                        heading_vector.set_color(0.0, 0.0, 1.0, 0.9)
                    else:
                        heading_vector.set_color(1.0, 0.0, 0.0, 0.9)
                    
                    geoms.append(heading_vector)
                    
                    # Arrow head
                    arrow_size = 0.04
                    direction_np = direction.cpu().numpy()
                    end_pos_np = end_pos.cpu().numpy()
                    
                    perp_dir = np.array([-direction_np[1], direction_np[0]])
                    arrow_tip = end_pos_np
                    arrow_left = end_pos_np - direction_np * arrow_size + perp_dir * arrow_size * 0.5
                    arrow_right = end_pos_np - direction_np * arrow_size - perp_dir * arrow_size * 0.5
                    arrow_points = [arrow_tip, arrow_left, arrow_right]
                    
                    try:
                        arrow_polygon = rendering.make_polygon(arrow_points, filled=True)
                        arrow_xform = rendering.Transform()
                        arrow_polygon.add_attr(arrow_xform)
                        
                        if agent.is_defender:
                            arrow_polygon.set_color(0.0, 0.0, 1.0, 1.0)
                        else:
                            arrow_polygon.set_color(1.0, 0.0, 0.0, 1.0)
                        
                        geoms.append(arrow_polygon)
                    except:
                        arrowhead = rendering.make_circle(0.025, filled=True)
                        arrowhead_xform = rendering.Transform()
                        arrowhead_xform.set_translation(*end_pos_np)
                        arrowhead.add_attr(arrowhead_xform)
                        
                        if agent.is_defender:
                            arrowhead.set_color(0.0, 0.0, 1.0, 1.0)
                        else:
                            arrowhead.set_color(1.0, 0.0, 0.0, 1.0)
                        
                        geoms.append(arrowhead)
        
        # 6. TRAJECTORIES
        if hasattr(self, 'agent_trajectories'):
            for agent in all_agents:
                if agent.name in self.agent_trajectories:
                    trajectory = self.agent_trajectories[agent.name]
                    
                    if len(trajectory) > 1:
                        for i in range(len(trajectory) - 1):
                            start_pos = trajectory[i]
                            end_pos = trajectory[i + 1]
                            
                            traj_segment = rendering.Line(
                                tuple(start_pos),
                                tuple(end_pos),
                                width=3
                            )
                            traj_xform = rendering.Transform()
                            traj_segment.add_attr(traj_xform)
                            
                            fade_factor = (i + 1) / len(trajectory)
                            
                            if agent.is_defender:
                                traj_segment.set_color(0.0, 0.0, 1.0, 0.4 * fade_factor)
                            else:
                                traj_segment.set_color(1.0, 0.0, 0.0, 0.4 * fade_factor)
                            
                            geoms.append(traj_segment)
        
        # 7. APOLLONIUS CIRCLES (when sensing occurs)
        if hasattr(self, 'attacker_sensed') and self.attacker_sensed is not None:
            for attacker_idx, attacker in enumerate(attackers):
                if self.attacker_sensed[env_index, attacker_idx]:
                    try:
                        attacker_pos_vmas = attacker.state.pos[env_index].cpu().numpy()
                        attacker_pos_world = self._vmas_to_world(attacker_pos_vmas)
                        
                        for defender in defenders:
                            defender_pos_vmas = defender.state.pos[env_index].cpu().numpy()
                            defender_pos_world = self._vmas_to_world(defender_pos_vmas)
                            
                            dist = np.linalg.norm(attacker_pos_world - defender_pos_world)
                            if dist <= self.sensing_radius + 0.1:
                                try:
                                    center, radius, lowest_point = compute_apollonius_circle(
                                        pos_a=attacker_pos_world,
                                        pos_d=defender_pos_world,
                                        speed_ratio=self.speed_ratio
                                    )
                                    
                                    # Draw Apollonius circle
                                    center_vmas = self._world_to_vmas(center)
                                    apo_circle = rendering.make_circle(radius, filled=False)
                                    apo_xform = rendering.Transform()
                                    apo_xform.set_translation(*center_vmas)
                                    apo_circle.add_attr(apo_xform)
                                    apo_circle.set_color(0.5, 0.0, 1.0, 0.8)  # Purple
                                    geoms.append(apo_circle)
                                    
                                    # Draw lowest point marker
                                    lowest_point_vmas = self._world_to_vmas(lowest_point)
                                    lowest_marker = rendering.make_circle(0.03, filled=True)
                                    lowest_xform = rendering.Transform()
                                    lowest_xform.set_translation(*lowest_point_vmas)
                                    lowest_marker.add_attr(lowest_xform)
                                    lowest_marker.set_color(0.0, 1.0, 1.0, 1.0)  # Cyan
                                    geoms.append(lowest_marker)
                                    
                                    # Draw escape route line
                                    escape_line = rendering.Line(
                                        tuple(attacker_pos_vmas),
                                        tuple(lowest_point_vmas),
                                        width=4
                                    )
                                    escape_xform = rendering.Transform()
                                    escape_line.add_attr(escape_xform)
                                    escape_line.set_color(0.0, 1.0, 1.0, 0.6)
                                    geoms.append(escape_line)
                                    
                                except Exception:
                                    pass
                    except Exception:
                        pass
        
        return geoms


if __name__ == "__main__":
    import vmas
    
    scenario = Scenario()
    
    # Test with smart attacker policy
    env = vmas.make_env(
        scenario=scenario,
        num_envs=2,
        device="cpu",
        continuous_actions=True,
        num_defenders=1,
        num_attackers=1,
        sensing_radius=0.3,
        speed_ratio=0.3,
        fixed_attacker_policy=False,  # Smart policy enabled
        spawn_area_mode=True,
        spawn_area_width=0.1
    )
    
    print(f"V3 Environment created with {env.n_agents} agents")
    print(f"Smart attacker policy: {not scenario.fixed_attacker_policy}")
    print(f"Sensing radius: {scenario.sensing_radius}")
    print(f"Capture distance: {scenario.capture_distance}")
    print("V3 Features:")
    print("  - Attackers use Apollonius circle feedback loop when sensed")
    print("  - Attackers move straight down when not sensed")
    print("  - Dynamic escape route computation every step")
    
    obs = env.reset()
    print(f"Initial observations shape: {[o.shape for o in obs]}")
    print("\n✓ V3 Environment with smart Apollonius-based attackers ready!")