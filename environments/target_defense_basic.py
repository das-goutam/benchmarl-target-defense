"""
Target Defense Environment in VMAS
Agents control only heading (direction) via first action dimension and always move at maximum speed
Variable number of attackers and defenders with sensing-based observations
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import math
from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    max_steps: int = MISSING
    num_defenders: int = MISSING
    num_attackers: int = MISSING
    sensing_radius: float = MISSING
    attacker_sensing_radius: float = MISSING
    speed_ratio: float = MISSING
    target_distance: float = MISSING
    randomize_attacker_x: bool = MISSING
    num_spawn_positions: int = MISSING
    spawn_area_mode: bool = MISSING
    spawn_area_width: float = MISSING
    enable_wall_constraints: bool = MISSING
    wall_epsilon: float = MISSING
    fixed_attacker_policy: bool = MISSING
    use_apollonius: bool = MISSING

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Y, X

# Try importing analytical solver (external)
try:
    from apollonius_solver import solve_apollonius_optimization
    APOLLONIUS_AVAILABLE = True
except ImportError:
    APOLLONIUS_AVAILABLE = False
    print("Warning: apollonius_solver not available. Using fallback rewards.")

# Integrated Apollonius solver class
class ApollonicsSolver:
    """Integrated Apollonius solver using proven external implementation"""
    
    @staticmethod
    def solve_apollonius_optimization(attacker_pos: np.ndarray, defender_positions: List[np.ndarray], nu: float) -> Dict:
        """Use the proven external solver implementation"""
        try:
            # Use the external solve_apollonius_optimization function
            return solve_apollonius_optimization(attacker_pos, defender_positions, nu)
        except Exception as e:
            # Fallback to Y-position
            return {
                'success': False,
                'defender_payoff': float(attacker_pos[1]),  # Y-position fallback
                'attacker_payoff': -float(attacker_pos[1]),
                'min_x_coordinate': float(attacker_pos[0]),
                'min_y_coordinate': float(attacker_pos[1]),
                'error': str(e)
            }


class Scenario(BaseScenario):
    """
    Target Defense Scenario for VMAS
    Defenders try to sense/intercept attackers before they reach the target line
    
    Action Format:
    - Actions are 2D tensors with shape (batch_size, 2)
    - First dimension: heading angle NORMALIZED to [-1, 1] (maps to [-π, π] radians)
      * 1.0 = π radians (180°, West)
      * 0.5 = π/2 radians (90°, North)
      * 0.0 = 0 radians (0°, East)
      * -0.5 = -π/2 radians (-90°, South)
      * -1.0 = -π radians (-180°)
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
        """
        # print("="*80)
        # print("🚀 TARGET_DEFENSE_ENVIRONMENT: FIXED VERSION - SHARED REWARDS + MOVEMENT SHAPING")
        # print("="*80)
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
        fixed_attacker_policy = kwargs.get('fixed_attacker_policy', True)  # Basic: always fixed policy
        num_spawn_positions = kwargs.get('num_spawn_positions', 3)  # Default 3 spawn positions on top edge
        max_steps = kwargs.get('max_steps', 200)  # Default max steps
        enable_wall_constraints = kwargs.get('enable_wall_constraints', True)
        use_apollonius = kwargs.get('use_apollonius', True)
        spawn_area_mode = kwargs.get('spawn_area_mode', False)  # New: area-based spawning
        spawn_area_width = kwargs.get('spawn_area_width', 0.2)  # New: spawn area height (fraction of world)
        absolute_observations = kwargs.get('absolute_observations', False)  # New: observation mode
        
        # Generate descriptive run name for WandB
        spawn_type = "area" if spawn_area_mode else "disc"
        run_name = f"{num_defenders}v{num_attackers}_sr{sensing_radius:.1f}_sp{speed_ratio:.1f}_{spawn_type}_basic"
        
        print(f"🎯 TARGET_DEFENSE_BASIC: {run_name}")
        print(f"   Config: {num_defenders}v{num_attackers}, sensing={sensing_radius}, speed_ratio={speed_ratio}")
        
        # Store scenario parameters
        self.batch_dim = batch_dim
        self.device = device
        self.num_defenders = num_defenders
        self.num_attackers = num_attackers
        self.sensing_radius = sensing_radius
        self.attacker_sensing_radius = attacker_sensing_radius
        self.speed_ratio = speed_ratio
        
        # Print parameters being used
        # print(f"ENVIRONMENT_PARAMS: defenders={num_defenders}, attackers={num_attackers}")
        # print(f"ENVIRONMENT_PARAMS: sensing_radius={sensing_radius}, speed_ratio={speed_ratio}")
        # print(f"ENVIRONMENT_PARAMS: max_steps={max_steps}, wall_constraints={enable_wall_constraints}")
        # print(f"ENVIRONMENT_PARAMS: use_apollonius={use_apollonius}, movement_rewards=ENABLED")
        self.target_distance = target_distance
        self.randomize_attacker_x = randomize_attacker_x
        self.fixed_attacker_policy = fixed_attacker_policy
        self.num_spawn_positions = num_spawn_positions
        self.max_steps = max_steps
        self.spawn_area_mode = spawn_area_mode
        self.spawn_area_width = spawn_area_width
        self.absolute_observations = absolute_observations
        
        # Speed settings - heterogeneous defenders when num_defenders=3
        if num_defenders == 3:
            # Heterogeneous speeds: Scout (fast), Balanced (medium), Sentinel (slow) - all faster than attacker
            defender_speeds = kwargs.get('defender_speeds', [0.07, 0.05, 0.04])  # Fast, Medium, Slow
            defender_sensing_radii = kwargs.get('defender_sensing_radii', [0.25, 0.3, 0.35])  # Short, Medium, Long
            self.defender_speeds = defender_speeds
            self.defender_sensing_radii = defender_sensing_radii
            avg_defender_speed = np.mean(defender_speeds)
            self.attacker_max_speed = avg_defender_speed * self.speed_ratio  # Attacker slower than average
            print(f"🎯 HETEROGENEOUS DEFENDERS:")
            print(f"   Scout: Speed={defender_speeds[0]}, Sensing={defender_sensing_radii[0]}")
            print(f"   Balanced: Speed={defender_speeds[1]}, Sensing={defender_sensing_radii[1]}")
            print(f"   Sentinel: Speed={defender_speeds[2]}, Sensing={defender_sensing_radii[2]}")
            print(f"   Attacker: Speed={self.attacker_max_speed} (slower than all defenders)")
        else:
            # Homogeneous defenders for 1v1
            defender_step_size = kwargs.get('defender_step_size', 0.05)
            self.defender_max_speed = defender_step_size
            self.attacker_max_speed = self.defender_max_speed * self.speed_ratio
            self.defender_speeds = [self.defender_max_speed] * num_defenders
            self.defender_sensing_radii = [sensing_radius] * num_defenders
        
        # Auto-calculate max_steps: 2x time for attacker to travel 1 unit (full world height)
        attacker_travel_time = 1.0 / self.attacker_max_speed  # Steps to cross world
        calculated_max_steps = int(2 * attacker_travel_time)  # Give 2x buffer time
        self.max_steps = calculated_max_steps  # Override YAML value
        
        print(f"📊 Auto-calculated max_steps: {self.max_steps} (overriding YAML: {max_steps})")
        if hasattr(self, 'defender_speeds'):
            print(f"   Defender speeds: {self.defender_speeds}")
        else:
            print(f"   Defender step size: {self.defender_max_speed}")
        print(f"   Attacker step size: {self.attacker_max_speed}")
        print(f"   Attacker travel time: {attacker_travel_time:.0f} steps")
        
        # Near-wall constraint controls
        self.enable_wall_constraints = bool(kwargs.get("enable_wall_constraints", True))
        self.wall_epsilon = float(kwargs.get("wall_epsilon", 0.03))  # distance from wall to start clamping
        
        # Apollonius solver controls
        self.use_apollonius = bool(kwargs.get("use_apollonius", True)) and APOLLONIUS_AVAILABLE
        self.apollonius_solver = ApollonicsSolver()  # Integrated solver instance
        
        # Create world - 1x1 space from 0 to 1
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=0.5,  # World goes from -0.5 to 0.5 (will offset positions)
            y_semidim=0.5,  # World goes from -0.5 to 0.5 (will offset positions)
            collision_force=0,  # No collisions
            substeps=1,
            dt=1.0  # Unit timestep
        )
        
        # Store world bounds for coordinate transformation
        self.world_min = 0.0
        self.world_max = 1.0
        
        # Create defender agents (heterogeneous when num_defenders=3)
        for i in range(num_defenders):
            # Use individual speeds and sensing for each defender
            if hasattr(self, 'defender_speeds'):
                defender_speed = self.defender_speeds[i]
                defender_sensing = self.defender_sensing_radii[i]
            else:
                defender_speed = self.defender_max_speed
                defender_sensing = sensing_radius
            
            agent = Agent(
                name=f"defender_{i}",
                shape=Sphere(radius=0.02),
                color=defender_color,
                max_speed=defender_speed,  # Individual speed
                rotatable=False,
                silent=True
            )
            agent.is_defender = True
            agent.sensing_radius = defender_sensing  # Individual sensing
            world.add_agent(agent)
        
        # Create attacker agents
        for i in range(num_attackers):
            agent = Agent(
                name=f"attacker_{i}",
                shape=Sphere(radius=0.02),
                color=attacker_color,
                max_speed=self.attacker_max_speed,
                rotatable=False,
                silent=True
                # Note: We keep default action_size (2) and handle heading in process_action
            )
            agent.is_defender = False
            agent.sensing_radius = attacker_sensing_radius
            world.add_agent(agent)
        
        # Store episode tracking variables (will be initialized in reset_world_at)
        # Track sensing for EACH attacker separately
        self.attacker_sensed = None  # Will be shape (batch_dim, num_attackers)
        self.attacker_intercepted = None  # Will be shape (batch_dim, num_attackers) - based on Apollonius payoff
        self.attacker_reached_target = None  # Will be shape (batch_dim, num_attackers)
        self.attacker_sensing_rewards = None  # Will be shape (batch_dim, num_attackers)
        
        # Initialize them here for immediate use
        self.attacker_sensed = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_intercepted = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_reached_target = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_sensing_rewards = torch.zeros((batch_dim, num_attackers), device=device)
        self.defender_has_sensed = torch.zeros((batch_dim, num_defenders), dtype=torch.bool, device=device)
        
        # Step tracking for max_steps termination
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        
        # Trajectory tracking for visualization
        self.max_trajectory_length = 50  # Keep last 50 positions
        self.agent_trajectories = {}  # Will store position history per agent
        
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
        
        # Position defenders evenly along bottom edge (world coordinates [0,1])
        defender_spacing = 1.0 / (self.num_defenders + 1)
        for i, defender in enumerate(defenders):
            world_x = (i + 1) * defender_spacing  # x position in [0,1] world coordinates
            vmas_x = self._world_to_vmas(world_x)  # Convert to VMAS [-0.5, 0.5]
            vmas_y = self._world_to_vmas(0.0)      # Bottom edge = y=0 in world = y=-0.5 in VMAS
            
            if env_index is None:
                defender.state.pos[:, X] = vmas_x
                defender.state.pos[:, Y] = vmas_y
                defender.state.vel[:, :] = 0
            else:
                defender.state.pos[env_index, X] = vmas_x
                defender.state.pos[env_index, Y] = vmas_y
                defender.state.vel[env_index, :] = 0
        
        # Create spawn positions once for all attackers (top edge y=1 or area-based)
        if self.spawn_area_mode:
            # NEW: Area-based spawning - attackers can spawn anywhere in top area band
            if env_index is None:
                batch_size = attackers[0].state.pos.shape[0] if attackers else self.batch_dim
                
                for env_idx in range(batch_size):
                    for i, attacker in enumerate(attackers):
                        # Random X position across full width [0, 1] in world coordinates
                        random_x_world = torch.rand(1).item()  # [0, 1]
                        random_x_vmas = self._world_to_vmas(random_x_world)
                        
                        # Random Y position in spawn area band [1-spawn_area_width, 1] in world coordinates
                        spawn_area_min_y = 1.0 - self.spawn_area_width  # e.g., 0.8 for 0.2 width
                        random_y_world = spawn_area_min_y + torch.rand(1).item() * self.spawn_area_width
                        random_y_vmas = self._world_to_vmas(random_y_world)
                        
                        attacker.state.pos[env_idx, X] = random_x_vmas
                        attacker.state.pos[env_idx, Y] = random_y_vmas
            else:
                # Single environment reset
                for i, attacker in enumerate(attackers):
                    random_x_world = torch.rand(1).item()
                    random_x_vmas = self._world_to_vmas(random_x_world)
                    
                    spawn_area_min_y = 1.0 - self.spawn_area_width
                    random_y_world = spawn_area_min_y + torch.rand(1).item() * self.spawn_area_width
                    random_y_vmas = self._world_to_vmas(random_y_world)
                    
                    attacker.state.pos[env_index, X] = random_x_vmas
                    attacker.state.pos[env_index, Y] = random_y_vmas
        elif self.randomize_attacker_x:
            # Create equally spaced spawn positions along top edge
            # For K positions, space them evenly across [0.1, 0.9] in world coordinates
            if self.num_spawn_positions == 1:
                world_spawn_positions = torch.tensor([0.5], device=self.device)  # Center
            else:
                # Equally spaced positions with 0.1 margin from edges
                spacing = 0.8 / (self.num_spawn_positions - 1)  # Total range is 0.8
                world_spawn_positions = torch.tensor(
                    [0.1 + i * spacing for i in range(self.num_spawn_positions)],
                    device=self.device
                )
            # Convert to VMAS coordinates
            spawn_positions = self._world_to_vmas(world_spawn_positions)
            
            # Ensure attackers spawn at different positions
            if env_index is None:
                batch_size = attackers[0].state.pos.shape[0] if attackers else self.batch_dim
                
                # For each environment, select unique positions for attackers
                for env_idx in range(batch_size):
                    # Get available positions
                    available_positions = spawn_positions.clone()
                    
                    # If we have more attackers than spawn positions, some will have to share
                    if self.num_attackers > self.num_spawn_positions:
                        # Repeat positions to have at least as many as attackers
                        repeats = (self.num_attackers // self.num_spawn_positions) + 1
                        available_positions = available_positions.repeat(repeats)[:self.num_attackers]
                        # Shuffle to randomize which positions get repeated
                        perm = torch.randperm(len(available_positions))
                        available_positions = available_positions[perm]
                    else:
                        # Randomly select unique positions for each attacker
                        perm = torch.randperm(len(available_positions))[:self.num_attackers]
                        available_positions = available_positions[perm]
                    
                    # Assign positions to attackers
                    for i, attacker in enumerate(attackers):
                        if i < len(available_positions):
                            attacker.state.pos[env_idx, X] = available_positions[i]
                        else:
                            # Fallback (shouldn't happen)
                            attacker.state.pos[env_idx, X] = 0.0
            else:
                # Single environment reset
                available_positions = spawn_positions.clone()
                
                if self.num_attackers > self.num_spawn_positions:
                    # Similar logic for single environment
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
        else:
            # Fixed center position for all attackers (original behavior)
            for i, attacker in enumerate(attackers):
                world_x = 0.5  # Center in world coordinates
                vmas_x = self._world_to_vmas(world_x)
                if env_index is None:
                    attacker.state.pos[:, X] = vmas_x
                else:
                    attacker.state.pos[env_index, X] = vmas_x
        
        # Set Y position and velocity for all attackers (only if NOT using area spawning)
        if not self.spawn_area_mode:
            # Original behavior: set Y to top edge
            world_y = 1.0  # Top edge in world coordinates
            vmas_y = self._world_to_vmas(world_y)
            for attacker in attackers:
                if env_index is None:
                    attacker.state.pos[:, Y] = vmas_y
                    attacker.state.vel[:, :] = 0
                else:
                    attacker.state.pos[env_index, Y] = vmas_y
                    attacker.state.vel[env_index, :] = 0
        else:
            # Area spawning mode: Y positions already set above, just reset velocity
            for attacker in attackers:
                if env_index is None:
                    attacker.state.vel[:, :] = 0
                else:
                    attacker.state.vel[env_index, :] = 0
        
        # Reset episode tracking - track each attacker separately
        if env_index is None:
            batch_size = self.batch_dim if hasattr(self, 'batch_dim') else self.world.batch_dim
            device = self.device if hasattr(self, 'device') else self.world.device
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_intercepted = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
            self.step_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            self.attacker_sensed[env_index, :] = False
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
        
        # Reset trajectories for new episode
        if hasattr(self, 'agent_trajectories'):
            for agent in self.world.agents:
                self.agent_trajectories[agent.name] = []
    
    def reward(self, agent: Agent) -> torch.Tensor:
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device

        # print(f"DEBUG_REWARD_CALLED: Agent {agent.name}, is_defender={agent.is_defender}")

        # ensure events are up-to-date; does NOT mutate if already updated
        self.update_events()

        # Only terminal rewards for defenders - SHARED EQUALLY among all defenders
        done_mask = self.done()
        r = torch.zeros(batch_size, device=device)
        
        if agent.is_defender:
            # Check if ANY environment has completed episodes with sensing
            if done_mask.any():
                # Calculate total reward across ALL environments
                for env_idx in range(batch_size):
                    env_total_reward = 0.0
                    
                    # If this environment is done, check for sensing rewards
                    if done_mask[env_idx]:
                        for a_idx in range(self.num_attackers):
                            if self.attacker_sensed[env_idx, a_idx]:
                                # Add reward for this sensed attacker
                                reward_val = self.attacker_sensing_rewards[env_idx, a_idx]
                                env_total_reward += reward_val
                                # print(f"DEBUG_REWARD: Env{env_idx}, Att{a_idx} sensed, reward={reward_val:.2f}")
                    
                    # ALL defenders get the same reward for this environment
                    r[env_idx] = env_total_reward
        
        # Attackers get no rewards (we're only training defenders)
        return r
    
    def observation(self, agent: Agent) -> torch.Tensor:
        """
        Get observation for an agent with absolute/relative observation option
        """
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Get absolute observations setting from stored parameter
        absolute_observations = getattr(self, 'absolute_observations', False)
        
        if absolute_observations:
            # V2: Absolute observations for all agents
            obs_dim = (self.num_defenders + self.num_attackers) * 2
            obs = torch.zeros(batch_size, obs_dim, device=device)
            
            # Get all agents in consistent order
            defenders = sorted([a for a in self.world.agents if a.is_defender], key=lambda x: x.name)
            attackers = sorted([a for a in self.world.agents if not a.is_defender], key=lambda x: x.name)
            all_agents = defenders + attackers
            
            idx = 0
            for other_agent in all_agents:
                if other_agent == agent:
                    # Own position (always visible)
                    obs[:, idx:idx+2] = other_agent.state.pos
                elif agent.is_defender and other_agent.is_defender:
                    # Other defenders (always visible to defenders)
                    obs[:, idx:idx+2] = other_agent.state.pos
                elif not agent.is_defender and not other_agent.is_defender:
                    # Other attackers (always visible to attackers)
                    obs[:, idx:idx+2] = other_agent.state.pos
                else:
                    # Opponents (only if within sensing radius)
                    dist = torch.norm(agent.state.pos - other_agent.state.pos, dim=-1)
                    within_range = dist <= agent.sensing_radius
                    obs[within_range, idx:idx+2] = other_agent.state.pos[within_range]
                
                idx += 2
        else:
            # V1: Relative observations (original working version)
            if agent.is_defender:
                obs_dim = 2 + (self.num_defenders - 1) * 2 + self.num_attackers * 2
            else:
                obs_dim = 2 + self.num_defenders * 2
            
            obs = torch.zeros(batch_size, obs_dim, device=device)
            
            defenders = sorted([a for a in self.world.agents if a.is_defender], key=lambda x: x.name)
            attackers = sorted([a for a in self.world.agents if not a.is_defender], key=lambda x: x.name)
            
            idx = 0
            
            # Own position (absolute)
            obs[:, idx:idx+2] = agent.state.pos
            idx += 2
            
            if agent.is_defender:
                # Other defenders' relative positions
                for other_defender in defenders:
                    if other_defender != agent:
                        relative_pos = other_defender.state.pos - agent.state.pos
                        obs[:, idx:idx+2] = relative_pos
                        idx += 2
                
                # Attackers (absolute if sensed)
                for attacker in attackers:
                    dist = torch.norm(agent.state.pos - attacker.state.pos, dim=-1)
                    within_range = dist <= agent.sensing_radius
                    obs[within_range, idx:idx+2] = attacker.state.pos[within_range]
                    idx += 2
            else:
                # Attackers observe defenders if within range
                for defender in defenders:
                    dist = torch.norm(agent.state.pos - defender.state.pos, dim=-1)
                    within_range = dist <= agent.sensing_radius
                    obs[within_range, idx:idx+2] = defender.state.pos[within_range]
                    idx += 2
        
        return obs
    
    def extra_render(self, env_index: int = 0):
        """Enhanced rendering with all visual elements"""
        
        try:
            from vmas.simulator import rendering
            from vmas.simulator.utils import Color
        except ImportError:
            return []
        
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
        
        geoms = []
        defenders = [a for a in self.world.agents if a.is_defender]
        
        # 1. BLACK WALLS
        wall_lines = [
            ((-0.5, -0.5), (-0.5, 0.5)),  # Left
            ((0.5, -0.5), (0.5, 0.5)),    # Right  
            ((-0.5, 0.5), (0.5, 0.5)),    # Top
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
        
        # 3. BLUE SENSING CIRCLES
        for defender in defenders:
            pos = defender.state.pos[env_index]
            sensing_circle = rendering.make_circle(self.sensing_radius, filled=False)
            sensing_xform = rendering.Transform()
            sensing_xform.set_translation(*pos.cpu().numpy())
            sensing_circle.add_attr(sensing_xform)
            sensing_circle.set_color(0.0, 0.0, 1.0, 0.6)
            geoms.append(sensing_circle)
        
        # 4. RED SPAWN AREA
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
        
        # Spawn position markers (if discrete mode)
        if not getattr(self, 'spawn_area_mode', False) and self.randomize_attacker_x:
            if self.num_spawn_positions > 1:
                spacing = 0.8 / (self.num_spawn_positions - 1)
                spawn_positions = [-0.4 + i * spacing for i in range(self.num_spawn_positions)]
                
                for spawn_x in spawn_positions:
                    spawn_marker = Circle(
                        pos=(spawn_x, 0.5),
                        radius=0.05,
                        color=(0.0, 0.8, 0.0, 0.8),
                        filled=True
                    )
                    geoms.append(spawn_marker)
        
        return geoms
    
    def _clamp_interval(self, theta, lo, hi):
        """Clamp angle theta (radians in [0, 2π)) into [lo, hi] elementwise."""
        lo_t = torch.full_like(theta, lo)
        hi_t = torch.full_like(theta, hi)
        theta = torch.where(theta < lo_t, lo_t, theta)
        theta = torch.where(theta > hi_t, hi_t, theta)
        return theta

    def _clamp_union(self, theta, segments):
        """
        Clamp theta (radians in [0, 2π)) to the closest boundary of a union of segments.
        segments: list of (lo, hi) with 0 <= lo < hi <= 2π.
        """
        B = theta.shape[0]
        inside = torch.zeros(B, dtype=torch.bool, device=theta.device)
        # Track nearest boundary across segments
        best_dist = torch.full_like(theta, float("inf"))
        best_proj = theta.clone()
        for lo, hi in segments:
            in_seg = (theta >= lo) & (theta <= hi)
            inside |= in_seg
            d_lo = torch.abs(theta - lo)
            d_hi = torch.abs(theta - hi)
            proj = torch.where(d_lo <= d_hi, torch.full_like(theta, lo), torch.full_like(theta, hi))
            dist = torch.minimum(d_lo, d_hi)
            better = dist < best_dist
            best_proj = torch.where(better, proj, best_proj)
            best_dist = torch.where(better, dist, best_dist)
        # Keep theta if inside any segment; otherwise project to nearest boundary
        return torch.where(inside, theta, best_proj)

    def _apply_wall_constraints(self, agent, theta_0_2pi):
        """
        Apply state-dependent heading constraints near walls/corners for DEFENDERS.
        theta_0_2pi: [B] angle in [0, 2π).
        Returns: clamped theta in [0, 2π).
        """
        if not self.enable_wall_constraints:
            return theta_0_2pi

        x = agent.state.pos[:, 0]  # X in VMAS coordinates [-0.5, 0.5]
        y = agent.state.pos[:, 1]  # Y in VMAS coordinates [-0.5, 0.5]
        wx = getattr(self.world, "x_semidim", 0.5)  # VMAS world bounds
        wy = getattr(self.world, "y_semidim", 0.5)  # VMAS world bounds
        eps = float(self.wall_epsilon)

        near_right  = (wx - x) <= eps
        near_left   = (x + wx) <= eps
        near_top    = (wy - y) <= eps
        near_bottom = (y + wy) <= eps

        theta = theta_0_2pi.clone()

        pi = math.pi
        two_pi = 2 * pi
        # Corner masks (handle first)
        tr = near_top & near_right     # top-right: 180°–270°
        tl = near_top & near_left      # top-left : 270°–360°
        br = near_bottom & near_right  # bot-right:  90°–180°
        bl = near_bottom & near_left   # bot-left :   0°– 90°

        theta = torch.where(tr, self._clamp_interval(theta, pi, 1.5 * pi), theta)
        theta = torch.where(tl, self._clamp_interval(theta, 1.5 * pi, two_pi), theta)
        theta = torch.where(br, self._clamp_interval(theta, 0.5 * pi, pi), theta)
        theta = torch.where(bl, self._clamp_interval(theta, 0.0, 0.5 * pi), theta)

        # Wall-only masks (exclude corners)
        right_only  = near_right  & ~(near_top | near_bottom)   # 90°–270°
        left_only   = near_left   & ~(near_top | near_bottom)   # 270°–360° ∪ 0°–90°
        top_only    = near_top    & ~(near_left | near_right)   # 180°–360°
        bottom_only = near_bottom & ~(near_left | near_right)   # 0°–180°

        # Right wall: single segment
        if right_only.any():
            theta[right_only] = self._clamp_interval(theta[right_only], 0.5 * pi, 1.5 * pi)

        # Left wall: union of two segments
        if left_only.any():
            segs = [(0.0, 0.5 * pi), (1.5 * pi, two_pi)]
            theta[left_only] = self._clamp_union(theta[left_only], segs)

        # Top wall: single segment
        if top_only.any():
            theta[top_only] = self._clamp_interval(theta[top_only], pi, two_pi)

        # Bottom wall: single segment
        if bottom_only.any():
            theta[bottom_only] = self._clamp_interval(theta[bottom_only], 0.0, pi)

        return theta

    def update_events(self):
        """
        Update sensing events and handle position snapping.
        Called once per step to avoid double processing.
        """
        if hasattr(self, '_events_updated_this_step') and self._events_updated_this_step:
            return  # Already updated this step
        
        # print(f"DEBUG_UPDATE: update_events called, sensing_radius={self.sensing_radius}")
        
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Initialize tracking variables if not already done
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_intercepted = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
        
        # Check for sensing events and track distances
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        # Update sensing status and track distances for EACH attacker-defender pair
        for attacker_idx, attacker in enumerate(attackers):
            for defender_idx, defender in enumerate(defenders):
                dist = torch.norm(attacker.state.pos - defender.state.pos, dim=-1)
                # Store current distance for monitoring
                self.distances[:, defender_idx, attacker_idx] = dist
                
                newly_sensed = (dist <= self.sensing_radius) & ~self.attacker_sensed[:, attacker_idx]
                
                # DEBUG: Check sensing detection
                # if dist.min() <= self.sensing_radius:
                #     print(f"DEBUG_SENSING: Def{defender_idx}-Att{attacker_idx}, dist={dist.min():.4f}, radius={self.sensing_radius}, newly_sensed={newly_sensed.any().item()}")
                
                # When sensing occurs, snap attacker to boundary and compute rewards/interception
                if newly_sensed.any():
                    # Removed defender marking - keep defenders active
                    
                    # Snap attacker position to the sensing boundary
                    for env_idx in torch.where(newly_sensed)[0]:
                        def_pos = defender.state.pos[env_idx]
                        att_pos = attacker.state.pos[env_idx]
                        
                        # Calculate direction from defender to attacker
                        direction = att_pos - def_pos
                        direction_norm = torch.norm(direction)
                        
                        # If inside sensing radius, move to boundary
                        if direction_norm < self.sensing_radius and direction_norm > 0:
                            # Normalize direction and place at boundary
                            direction_normalized = direction / direction_norm
                            attacker.state.pos[env_idx] = def_pos + direction_normalized * self.sensing_radius
                        
                        # Also set velocity to zero
                        attacker.state.vel[env_idx] = torch.zeros_like(attacker.state.vel[env_idx])
                    
                    # Compute payoff using Apollonius solver and determine interception
                    if self.use_apollonius:
                        for env_idx in torch.where(newly_sensed)[0]:
                            # Get current positions for this environment in VMAS coords
                            attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                            # Convert to global frame [0,1] for Apollonius solver
                            attacker_pos = self._vmas_to_world(attacker_pos_vmas)
                            
                            # Get all defender positions for this environment
                            defender_positions = []
                            for def_agent in self.world.agents:
                                if def_agent.is_defender:
                                    defender_pos_vmas = def_agent.state.pos[env_idx].cpu().numpy()
                                    # Convert to global frame [0,1] for Apollonius solver
                                    defender_pos_global = self._vmas_to_world(defender_pos_vmas)
                                    defender_positions.append(defender_pos_global)
                            
                            # Solve Apollonius optimization using integrated solver
                            result = self.apollonius_solver.solve_apollonius_optimization(
                                attacker_pos=attacker_pos,
                                defender_positions=defender_positions,
                                nu=1.0 / self.speed_ratio  # nu = defender_speed / attacker_speed
                            )
                            
                            if result['success']:
                                # Use ONLY Apollonius defender payoff as reward
                                defender_payoff = result['defender_payoff']
                                self.attacker_sensing_rewards[env_idx, attacker_idx] = defender_payoff
                                
                                # Mark as intercepted if payoff is positive
                                if defender_payoff > 0:
                                    self.attacker_intercepted[env_idx, attacker_idx] = True
                            else:
                                # Fallback if Apollonius solver fails - use y-position in global frame [0,1]
                                attacker_y_vmas = attacker.state.pos[env_idx, Y].item()
                                attacker_y_world = self._vmas_to_world(attacker_y_vmas)  # Convert to [0,1]
                                self.attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
                    else:
                        # Fallback if Apollonius not available - use y-position in global frame [0,1]
                        for env_idx in torch.where(newly_sensed)[0]:
                            attacker_y_vmas = attacker.state.pos[env_idx, Y].item()
                            attacker_y_world = self._vmas_to_world(attacker_y_vmas)  # Convert to [0,1]
                            self.attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
                
                self.attacker_sensed[:, attacker_idx] |= newly_sensed
        
        # Check for target reached for EACH attacker (only if not sensed)
        # Target is at bottom edge (y=0 in world, y=-0.5 in VMAS)
        target_y_vmas = self._world_to_vmas(0.0)  # Bottom edge
        for attacker_idx, attacker in enumerate(attackers):
            reached = (attacker.state.pos[:, Y] <= target_y_vmas + self.target_distance) & ~self.attacker_sensed[:, attacker_idx]
            self.attacker_reached_target[:, attacker_idx] |= reached
        
        # Mark that events have been updated this step
        self._events_updated_this_step = True
        
        # Reset flag for next step (in case extra_render is not called)
        # We'll reset this flag when the first agent processes its action next step
        self._reset_events_flag_next_step = True

    def process_action(self, agent: Agent):
        """
        Process agent action (heading control only)
        We use the first action dimension as heading angle and ignore the second
        Actions come in normalized to [-1, 1], we scale to [-π, π]
        Agents always move at maximum speed in the direction specified
        
        Args:
            agent: The agent whose action to process
        """
        # Reset events flag for new step if needed
        if hasattr(self, '_reset_events_flag_next_step') and self._reset_events_flag_next_step:
            self._events_updated_this_step = False
            self._step_incremented_this_step = False
            self._reset_events_flag_next_step = False
        
        # Get batch size and device
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Check if this attacker is already sensed or reached target (becomes inactive)
        if not agent.is_defender:
            # Find which attacker this is
            attacker_idx = -1
            attackers = [a for a in self.world.agents if not a.is_defender]
            for idx, a in enumerate(attackers):
                if a == agent:
                    attacker_idx = idx
                    break
            
            # If this attacker is sensed or reached target, it stops moving
            if attacker_idx >= 0 and hasattr(self, 'attacker_sensed'):
                is_inactive = self.attacker_sensed[:, attacker_idx] | self.attacker_reached_target[:, attacker_idx]
                
                if self.fixed_attacker_policy:
                    # Active attackers move down, inactive ones keep their heading but speed will be 0
                    heading = torch.full((batch_size,), -math.pi/2, device=device)  # Always down heading
                else:
                    # For non-fixed policy, use action heading (speed will be set to 0 below)
                    if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                        normalized_heading = agent.action.u[:, 0]
                        heading = normalized_heading * math.pi  # Use action heading
                    else:
                        heading = torch.zeros(batch_size, device=device)
            else:
                # Fallback if tracking not initialized
                if self.fixed_attacker_policy:
                    heading = torch.full((batch_size,), -math.pi/2, device=device)
                else:
                    heading = torch.zeros(batch_size, device=device)
        else:
            # Defender processing
            # Actions are normalized to [-1, 1], scale to [-π, π]
            if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                normalized_heading = agent.action.u[:, 0]  # In range [-1, 1]
                heading = normalized_heading * math.pi      # [-π, π]
            else:
                heading = torch.zeros(batch_size, device=device)

        # Convert to [0, 2π) for clamping
        theta = torch.remainder(heading, 2 * math.pi)

        # >>> ADD THIS: state-dependent angle clamping for defenders <<<
        if agent.is_defender:
            theta = self._apply_wall_constraints(agent, theta)
        # <<< END ADD >>>
        
        # Convert heading to velocity
                # Convert heading to velocity (use individual speeds)
        if not agent.is_defender:
            max_speed = self.attacker_max_speed
        else:
            # Use individual defender speed
            max_speed = agent.max_speed
        
        # Defenders remain active (removed inactivity logic)
        
        # For attackers, check if they should be inactive
        if not agent.is_defender and hasattr(self, 'attacker_sensed'):
            # Find attacker index again (could optimize by storing earlier)
            attacker_idx = -1
            attackers = [a for a in self.world.agents if not a.is_defender]
            for idx, a in enumerate(attackers):
                if a == agent:
                    attacker_idx = idx
                    break
            
            if attacker_idx >= 0:
                is_inactive = self.attacker_sensed[:, attacker_idx] | self.attacker_reached_target[:, attacker_idx]
                # Set speed to 0 for inactive attackers
                max_speed = torch.where(
                    is_inactive, 
                    torch.zeros(batch_size, device=device), 
                    torch.full((batch_size,), max_speed, device=device)
                )
        
        # Update sensing events once per step (before velocity computation)
        self.update_events()
        
        # Increment step count (only once per step for the first agent processed)
        if not hasattr(self, '_step_incremented_this_step') or not self._step_incremented_this_step:
            if hasattr(self, 'step_count'):
                self.step_count += 1
            self._step_incremented_this_step = True
        
        # Override the action with computed velocity (use theta, not heading)
        # This bypasses the default dynamics processing
        if agent.action is not None and hasattr(agent.action, 'u'):
            agent.action.u[:, 0] = max_speed * torch.cos(theta)
            agent.action.u[:, 1] = max_speed * torch.sin(theta)
        
        # Track trajectory for visualization (only for env_index=0 to save memory)
        if not hasattr(self, '_trajectory_initialized') or not self._trajectory_initialized:
            # Initialize trajectory tracking for all agents
            for a in self.world.agents:
                self.agent_trajectories[a.name] = []
            self._trajectory_initialized = True
        
        # Store current position in trajectory (for first environment only)
        current_pos = agent.state.pos[0].clone().cpu().numpy()
        if agent.name not in self.agent_trajectories:
            self.agent_trajectories[agent.name] = []
        
        self.agent_trajectories[agent.name].append(current_pos.copy())
        
        # Keep only recent positions (sliding window)
        if len(self.agent_trajectories[agent.name]) > self.max_trajectory_length:
            self.agent_trajectories[agent.name] = self.agent_trajectories[agent.name][-self.max_trajectory_length:]
    
    def done(self) -> torch.Tensor:
        """
        Check if episodes are done
        
        Returns:
            Boolean tensor indicating which episodes are complete
        """
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
            device = self.world.device if hasattr(self.world, 'device') else self.device
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Episode is done when ALL attackers are either sensed or reached target
        # Each attacker must reach a terminal state (sensed OR reached target)
        all_attackers_done = (self.attacker_sensed | self.attacker_reached_target).all(dim=1)
        
        # OR when max_steps is reached
        max_steps_reached = torch.zeros_like(all_attackers_done, dtype=torch.bool)
        if hasattr(self, 'step_count'):
            max_steps_reached = self.step_count >= self.max_steps
        
        return all_attackers_done | max_steps_reached
    
    def info(self, agent: Agent) -> Dict:
        """
        Get info dictionary for an agent
        
        Args:
            agent: The agent to get info for
            
        Returns:
            Dictionary with episode information
        """
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
            device = self.world.device if hasattr(self.world, 'device') else self.device
            return {
                "attackers_sensed": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_intercepted": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_reached_target": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attacker_rewards": torch.zeros((batch_size, self.num_attackers), device=device),
                # Aggregate info for backward compatibility
                "sensing_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "interception_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "target_reached": torch.zeros(batch_size, dtype=torch.bool, device=device)
            }
        
        return {
            "attackers_sensed": self.attacker_sensed.clone(),
            "attackers_intercepted": self.attacker_intercepted.clone(),
            "attackers_reached_target": self.attacker_reached_target.clone(),
            "attacker_rewards": self.attacker_sensing_rewards.clone(),
            # Aggregate info for backward compatibility
            "sensing_occurred": self.attacker_sensed.any(dim=1),
            "interception_occurred": self.attacker_intercepted.any(dim=1),
            "target_reached": self.attacker_reached_target.any(dim=1)
        }


if __name__ == "__main__":
    # Test the scenario
    import vmas
    
    # Create environment with variable agent numbers
    scenario = Scenario()
    
    # Test with default configuration (3 defenders, 1 attacker)
    env = vmas.make_env(
        scenario=scenario,
        num_envs=4,  # Vectorized batch of 4 environments
        device="cpu",
        continuous_actions=True,
        num_defenders=3,
        num_attackers=1,
        randomize_attacker_x=True
    )
    
    print(f"Environment created with {env.n_agents} agents")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nAction format:")
    print("  - Shape: (batch_size, 2)")
    print("  - Dim 0: heading angle NORMALIZED to [-1, 1] (maps to [-π, π] radians)")
    print("  - Dim 1: ignored (set to 0)")
    print("  - Example: 0.5 = π/2 radians (90°), -0.5 = -π/2 radians (-90°)")
    
    # Reset and test
    obs = env.reset()
    print(f"\nInitial observations shape: {[o.shape for o in obs]}")
    
    # Debug positions in world coordinates [0,1]
    print("\nDebug: Agent positions in world coordinates [0,1]:")
    for i, agent in enumerate(env.agents):
        vmas_pos = agent.state.pos[0]  # First environment
        world_pos = scenario._vmas_to_world(vmas_pos)
        print(f"  {agent.name}: VMAS {vmas_pos.tolist()} -> World {world_pos.tolist()}")
    
    # Test with different agent numbers
    print("\n" + "="*50)
    print("Testing with 5 defenders and 2 attackers:")
    
    env2 = vmas.make_env(
        scenario=scenario,
        num_envs=2,
        device="cpu",
        continuous_actions=True,
        num_defenders=5,
        num_attackers=2
    )
    
    print(f"Environment created with {env2.n_agents} agents")
    obs2 = env2.reset()
    print(f"Observations shape: {[o.shape for o in obs2]}")
    
    # Test step with heading control
    actions = []
    for i, agent in enumerate(env2.agents):
        if agent.is_defender:
            # Defenders move up (0.5 normalized = π/2 radians)
            action = torch.tensor([[0.5, 0], [0.5, 0]])  # Shape: (num_envs, 2)
        else:
            # Attackers: action provided but overridden by fixed policy
            action = torch.tensor([[-0.5, 0], [-0.5, 0]])  # -0.5 = -π/2 radians
        actions.append(action)
    
    obs, rewards, dones, info = env2.step(actions)
    print(f"\nAfter step:")
    print(f"Rewards shape: {[r.shape for r in rewards]}")
    print(f"Done: {dones}")
    print("\n✓ Environment working correctly with heading-only control!")
    
    # Optional: Interactive rendering (requires pygame)
    # render_interactively(__file__, control_two_agents=False)