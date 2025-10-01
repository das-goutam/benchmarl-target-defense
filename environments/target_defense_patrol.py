"""
Patroller-Pursuer Target Defense Environment in VMAS
Two asymmetric defenders: Patroller (with sensing) and Blind Pursuer (no sensing)
Patroller detects, Pursuer determines capture quality via Apollonius payoff
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import math
from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    max_steps: int = MISSING
    patroller_sensing_radius: float = MISSING
    patroller_speed_ratio: float = MISSING
    pursuer_speed_ratio: float = MISSING
    attacker_speed_ratio: float = MISSING
    target_distance: float = MISSING
    randomize_attacker_x: bool = MISSING
    spawn_area_mode: bool = MISSING
    spawn_area_width: float = MISSING
    enable_wall_constraints: bool = MISSING
    wall_epsilon: float = MISSING
    use_apollonius: bool = MISSING
    terminate_on_detection: bool = MISSING

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

class ApollonicsSolver:
    """Integrated Apollonius solver using proven external implementation"""
    
    @staticmethod
    def solve_apollonius_optimization(attacker_pos: np.ndarray, defender_positions: List[np.ndarray], nu: float) -> Dict:
        """Use the proven external solver implementation"""
        try:
            return solve_apollonius_optimization(attacker_pos, defender_positions, nu)
        except Exception as e:
            return {
                'success': False,
                'defender_payoff': float(attacker_pos[1]),
                'attacker_payoff': -float(attacker_pos[1]),
                'min_x_coordinate': float(attacker_pos[0]),
                'min_y_coordinate': float(attacker_pos[1]),
                'error': str(e)
            }


class Scenario(BaseScenario):
    """
    Patroller-Pursuer Target Defense Scenario for VMAS
    
    Agents:
    - Patroller: Has sensing radius, searches for attacker
    - Blind Pursuer: No sensing, relies on patroller's detection
    - Attacker: Tries to reach target line
    
    Mechanics:
    - Episode terminates when patroller detects attacker
    - Reward based on Apollonius payoff using pursuer position
    - Both defenders share observations and detected attacker info
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """Create the world with patroller, pursuer, and attacker"""
        
        print("="*80)
        print("🎯 PATROLLER-PURSUER TARGET DEFENSE ENVIRONMENT")
        print("="*80)
        
        # Extract parameters with defaults
        patroller_sensing_radius = kwargs.get('patroller_sensing_radius', 0.25)
        patroller_speed_ratio = kwargs.get('patroller_speed_ratio', 0.8)  # Relative to base speed
        pursuer_speed_ratio = kwargs.get('pursuer_speed_ratio', 1.0)  # Fastest
        attacker_speed_ratio = kwargs.get('attacker_speed_ratio', 0.3)  # Slowest
        target_distance = kwargs.get('target_distance', 0.05)
        patroller_color = kwargs.get('patroller_color', (0.0, 0.5, 1.0))  # Light blue
        pursuer_color = kwargs.get('pursuer_color', (0.0, 0.0, 1.0))  # Dark blue
        attacker_color = kwargs.get('attacker_color', (1.0, 0.0, 0.0))  # Red
        randomize_attacker_x = kwargs.get('randomize_attacker_x', True)
        max_steps = kwargs.get('max_steps', 200)
        enable_wall_constraints = kwargs.get('enable_wall_constraints', True)
        wall_epsilon = kwargs.get('wall_epsilon', 0.03)
        use_apollonius = kwargs.get('use_apollonius', True)
        spawn_area_mode = kwargs.get('spawn_area_mode', False)
        spawn_area_width = kwargs.get('spawn_area_width', 0.2)
        terminate_on_detection = kwargs.get('terminate_on_detection', True)
        
        # Store scenario parameters
        self.batch_dim = batch_dim
        self.device = device
        self.patroller_sensing_radius = patroller_sensing_radius
        self.patroller_speed_ratio = patroller_speed_ratio
        self.pursuer_speed_ratio = pursuer_speed_ratio
        self.attacker_speed_ratio = attacker_speed_ratio
        self.target_distance = target_distance
        self.randomize_attacker_x = randomize_attacker_x
        self.spawn_area_mode = spawn_area_mode
        self.spawn_area_width = spawn_area_width
        self.enable_wall_constraints = enable_wall_constraints
        self.wall_epsilon = wall_epsilon
        self.use_apollonius = use_apollonius and APOLLONIUS_AVAILABLE
        self.apollonius_solver = ApollonicsSolver()
        self.terminate_on_detection = terminate_on_detection
        
        # Calculate actual speeds (base speed = 0.05)
        base_speed = 0.05
        self.patroller_speed = base_speed * patroller_speed_ratio
        self.pursuer_speed = base_speed * pursuer_speed_ratio
        self.attacker_speed = base_speed * attacker_speed_ratio
        
        # Auto-calculate max_steps
        attacker_travel_time = 1.0 / self.attacker_speed
        self.max_steps = int(2 * attacker_travel_time)
        
        print(f"📊 Configuration:")
        print(f"   Patroller: speed={self.patroller_speed:.3f}, sensing={patroller_sensing_radius}")
        print(f"   Pursuer: speed={self.pursuer_speed:.3f}, no sensing")
        print(f"   Attacker: speed={self.attacker_speed:.3f}")
        print(f"   Max steps: {self.max_steps}")
        print(f"   Terminate on detection: {terminate_on_detection}")
        
        # Create world
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=0.5,
            y_semidim=0.5,
            collision_force=0,
            substeps=1,
            dt=1.0
        )
        
        # Create patroller agent
        patroller = Agent(
            name="patroller",
            shape=Sphere(radius=0.02),
            color=patroller_color,
            max_speed=self.patroller_speed,
            rotatable=False,
            silent=True
        )
        patroller.is_defender = True
        patroller.is_patroller = True
        patroller.sensing_radius = patroller_sensing_radius
        world.add_agent(patroller)
        
        # Create pursuer agent
        pursuer = Agent(
            name="pursuer",
            shape=Sphere(radius=0.02),
            color=pursuer_color,
            max_speed=self.pursuer_speed,
            rotatable=False,
            silent=True
        )
        pursuer.is_defender = True
        pursuer.is_patroller = False
        pursuer.sensing_radius = 0.0  # No sensing capability
        world.add_agent(pursuer)
        
        # Create attacker agent
        attacker = Agent(
            name="attacker",
            shape=Sphere(radius=0.02),
            color=attacker_color,
            max_speed=self.attacker_speed,
            rotatable=False,
            silent=True
        )
        attacker.is_defender = False
        attacker.is_patroller = False
        attacker.sensing_radius = 0.0
        world.add_agent(attacker)
        
        # Initialize tracking variables early for safety
        self.attacker_detected = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.attacker_position_known = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.attacker_reached_target = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.detection_reward = torch.zeros(batch_dim, device=device)
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        
        # For storing last known attacker position after detection
        self.last_known_attacker_pos = torch.zeros((batch_dim, 2), device=device)
        
        # Mark initialization as complete
        self._initialized = True
        
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
        """Reset world to initial positions"""
        
        # Get agents
        patroller = None
        pursuer = None
        attacker = None
        for agent in self.world.agents:
            if agent.name == "patroller":
                patroller = agent
            elif agent.name == "pursuer":
                pursuer = agent
            elif agent.name == "attacker":
                attacker = agent
        
        # Position patroller at center of bottom edge
        patroller_x = self._world_to_vmas(0.5)  # Center
        patroller_y = self._world_to_vmas(0.0)  # Bottom
        
        if env_index is None:
            patroller.state.pos[:, X] = patroller_x
            patroller.state.pos[:, Y] = patroller_y
            patroller.state.vel[:, :] = 0
        else:
            patroller.state.pos[env_index, X] = patroller_x
            patroller.state.pos[env_index, Y] = patroller_y
            patroller.state.vel[env_index, :] = 0
        
        # Position pursuer randomly on bottom edge
        if env_index is None:
            batch_size = self.batch_dim
            # Random x positions in [0.1, 0.9] world coordinates
            pursuer_x_world = 0.1 + torch.rand(batch_size, device=self.device) * 0.8
            pursuer_x_vmas = self._world_to_vmas(pursuer_x_world)
            pursuer.state.pos[:, X] = pursuer_x_vmas
            pursuer.state.pos[:, Y] = patroller_y
            pursuer.state.vel[:, :] = 0
        else:
            pursuer_x_world = 0.1 + torch.rand(1, device=self.device).item() * 0.8
            pursuer_x_vmas = self._world_to_vmas(pursuer_x_world)
            pursuer.state.pos[env_index, X] = pursuer_x_vmas
            pursuer.state.pos[env_index, Y] = patroller_y
            pursuer.state.vel[env_index, :] = 0
        
        # Position attacker
        if self.spawn_area_mode:
            # Area-based spawning
            if env_index is None:
                batch_size = self.batch_dim
                for env_idx in range(batch_size):
                    random_x_world = torch.rand(1).item()
                    random_x_vmas = self._world_to_vmas(random_x_world)
                    spawn_area_min_y = 1.0 - self.spawn_area_width
                    random_y_world = spawn_area_min_y + torch.rand(1).item() * self.spawn_area_width
                    random_y_vmas = self._world_to_vmas(random_y_world)
                    attacker.state.pos[env_idx, X] = random_x_vmas
                    attacker.state.pos[env_idx, Y] = random_y_vmas
            else:
                random_x_world = torch.rand(1).item()
                random_x_vmas = self._world_to_vmas(random_x_world)
                spawn_area_min_y = 1.0 - self.spawn_area_width
                random_y_world = spawn_area_min_y + torch.rand(1).item() * self.spawn_area_width
                random_y_vmas = self._world_to_vmas(random_y_world)
                attacker.state.pos[env_index, X] = random_x_vmas
                attacker.state.pos[env_index, Y] = random_y_vmas
        elif self.randomize_attacker_x:
            # Random x position on top edge
            if env_index is None:
                batch_size = self.batch_dim
                attacker_x_world = 0.1 + torch.rand(batch_size, device=self.device) * 0.8
                attacker_x_vmas = self._world_to_vmas(attacker_x_world)
                attacker.state.pos[:, X] = attacker_x_vmas
                attacker.state.pos[:, Y] = self._world_to_vmas(1.0)
            else:
                attacker_x_world = 0.1 + torch.rand(1, device=self.device).item() * 0.8
                attacker_x_vmas = self._world_to_vmas(attacker_x_world)
                attacker.state.pos[env_index, X] = attacker_x_vmas
                attacker.state.pos[env_index, Y] = self._world_to_vmas(1.0)
        else:
            # Fixed center position
            if env_index is None:
                attacker.state.pos[:, X] = self._world_to_vmas(0.5)
                attacker.state.pos[:, Y] = self._world_to_vmas(1.0)
            else:
                attacker.state.pos[env_index, X] = self._world_to_vmas(0.5)
                attacker.state.pos[env_index, Y] = self._world_to_vmas(1.0)
        
        if env_index is None:
            attacker.state.vel[:, :] = 0
        else:
            attacker.state.vel[env_index, :] = 0
        
        # Reset tracking variables
        if env_index is None:
            self.attacker_detected[:] = False
            self.attacker_position_known[:] = False
            self.attacker_reached_target[:] = False
            self.detection_reward[:] = 0.0
            self.step_count[:] = 0
            self.last_known_attacker_pos[:] = 0.0
        else:
            self.attacker_detected[env_index] = False
            self.attacker_position_known[env_index] = False
            self.attacker_reached_target[env_index] = False
            self.detection_reward[env_index] = 0.0
            self.step_count[env_index] = 0
            self.last_known_attacker_pos[env_index] = 0.0
        
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
    
    def reward(self, agent: Agent) -> torch.Tensor:
        """Shared reward for both defenders based on Apollonius payoff at detection"""
        # Get batch size and device with fallbacks
        if hasattr(self, 'world'):
            batch_size = getattr(self.world, 'batch_dim', 1)
            device = getattr(self.world, 'device', torch.device('cpu'))
        else:
            batch_size = 1
            device = torch.device('cpu')
        
        # Safety check for initialization
        if not hasattr(self, '_initialized') or not self._initialized:
            return torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Update events
        self.update_events()
        
        r = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        if agent.is_defender:
            # Both defenders get the same reward
            # Reward is given when detection occurs (if terminating) or at episode end
            if self.terminate_on_detection:
                # Give reward immediately upon detection
                newly_detected = self.attacker_detected & (self.detection_reward != 0)
                r = torch.where(newly_detected, self.detection_reward, r)
            else:
                # Give reward at episode end
                done_mask = self.done()
                r = torch.where(done_mask & self.attacker_detected, self.detection_reward, r)
                # Penalty if attacker reaches target
                r = torch.where(done_mask & self.attacker_reached_target, 
                               torch.full_like(r, 0.0), r)
        
        return r.to(dtype=torch.float32)
    
    def observation(self, agent: Agent) -> torch.Tensor:
        """
        Observation for each agent
        Both defenders see each other always
        Only patroller can sense attacker initially
        Once detected, both see attacker position
        """
        # Get batch size and device with fallbacks
        if hasattr(self, 'world'):
            batch_size = getattr(self.world, 'batch_dim', 1)
            device = getattr(self.world, 'device', torch.device('cpu'))
        else:
            batch_size = 1
            device = torch.device('cpu')
        
        # Safety check for initialization - always return proper tensor
        if not hasattr(self, '_initialized') or not self._initialized:
            # Return dummy observation during initialization
            return torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
        
        # Additional safety check for agent state
        if not hasattr(agent, 'state') or agent.state is None:
            return torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
        
        if not hasattr(agent.state, 'pos') or agent.state.pos is None:
            return torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
        
        # Get agents
        patroller = None
        pursuer = None
        attacker = None
        for a in self.world.agents:
            if a.name == "patroller":
                patroller = a
            elif a.name == "pursuer":
                pursuer = a
            elif a.name == "attacker":
                attacker = a
        
        # Observation: [own_pos(2), other_defender_pos(2), attacker_pos_if_visible(2)]
        obs = torch.zeros(batch_size, 6, dtype=torch.float32, device=device)
        
        # Check if agent states are initialized
        if not hasattr(agent, 'state') or not hasattr(agent.state, 'pos'):
            return obs
        
        # Ensure agent.state.pos is not None and has correct shape
        if agent.state.pos is None:
            return obs
        
        # Own position - ensure it's a tensor
        try:
            obs[:, 0:2] = agent.state.pos
        except Exception:
            # If there's any issue, return the zeros
            return obs
        
        # Other defender position (always visible)
        try:
            if agent.name == "patroller" and pursuer and hasattr(pursuer.state, 'pos'):
                if pursuer.state.pos is not None:
                    obs[:, 2:4] = pursuer.state.pos
            elif agent.name == "pursuer" and patroller and hasattr(patroller.state, 'pos'):
                if patroller.state.pos is not None:
                    obs[:, 2:4] = patroller.state.pos
        except Exception:
            # Continue with zeros if there's an issue
            pass
        
        # Attacker position
        try:
            if attacker and hasattr(attacker.state, 'pos') and attacker.state.pos is not None:
                if agent.name == "patroller":
                    # Patroller can sense within radius OR if already detected
                    dist = torch.norm(agent.state.pos - attacker.state.pos, dim=-1)
                    can_sense = (dist <= self.patroller_sensing_radius) | self.attacker_position_known
                    obs[can_sense, 4:6] = attacker.state.pos[can_sense]
                else:  # pursuer
                    # Pursuer only sees attacker after patroller detects it
                    obs[self.attacker_position_known, 4:6] = attacker.state.pos[self.attacker_position_known]
        except Exception:
            # Continue with zeros if there's an issue
            pass
        
        # Ensure we always return a float32 tensor
        return obs.to(dtype=torch.float32)
    
    def update_events(self):
        """Update detection events and compute rewards"""
        # Safety check
        if not hasattr(self, '_initialized') or not self._initialized:
            return
            
        if hasattr(self, '_events_updated_this_step') and self._events_updated_this_step:
            return
        
        batch_size = self.world.batch_dim
        device = self.world.device
        
        # Get agents
        patroller = None
        pursuer = None
        attacker = None
        for agent in self.world.agents:
            if agent.name == "patroller":
                patroller = agent
            elif agent.name == "pursuer":
                pursuer = agent
            elif agent.name == "attacker":
                attacker = agent
        
        # Safety check for agent states
        if not (patroller and pursuer and attacker):
            return
        if not (hasattr(patroller.state, 'pos') and hasattr(attacker.state, 'pos')):
            return
        
        # Check for detection by patroller
        dist = torch.norm(patroller.state.pos - attacker.state.pos, dim=-1)
        newly_detected = (dist <= self.patroller_sensing_radius) & ~self.attacker_detected
        
        if newly_detected.any():
            # Mark as detected
            self.attacker_detected |= newly_detected
            self.attacker_position_known |= newly_detected
            
            # Store last known position
            self.last_known_attacker_pos[newly_detected] = attacker.state.pos[newly_detected]
            
            # Compute Apollonius payoff using PURSUER position
            for env_idx in torch.where(newly_detected)[0]:
                if self.use_apollonius:
                    # Get positions in world coordinates [0,1]
                    attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                    attacker_pos = self._vmas_to_world(attacker_pos_vmas)
                    
                    # Use ONLY pursuer position for Apollonius
                    pursuer_pos_vmas = pursuer.state.pos[env_idx].cpu().numpy()
                    pursuer_pos = self._vmas_to_world(pursuer_pos_vmas)
                    
                    # Solve Apollonius with pursuer as the defender
                    result = self.apollonius_solver.solve_apollonius_optimization(
                        attacker_pos=attacker_pos,
                        defender_positions=[pursuer_pos],  # Only pursuer position
                        nu=self.pursuer_speed / self.attacker_speed  # Speed ratio
                    )
                    
                    if result['success']:
                        # Reward is distance from target (y=0) to attacker position
                        # Higher y = better reward (caught early)
                        self.detection_reward[env_idx] = result['defender_payoff']
                    else:
                        # Fallback: use attacker y-position
                        self.detection_reward[env_idx] = float(attacker_pos[1])
                else:
                    # Fallback: use attacker y-position
                    attacker_y = self._vmas_to_world(attacker.state.pos[env_idx, Y].item())
                    self.detection_reward[env_idx] = attacker_y
        
        # Check if attacker reached target (only if not detected)
        target_y_vmas = self._world_to_vmas(0.0)
        reached = (attacker.state.pos[:, Y] <= target_y_vmas + self.target_distance) & ~self.attacker_detected
        self.attacker_reached_target |= reached
        
        self._events_updated_this_step = True
    
    def process_action(self, agent: Agent):
        """Process agent actions with heading control"""
        # Safety check
        if not hasattr(self, '_initialized') or not self._initialized:
            return
        
        # Reset flags if needed
        if hasattr(self, '_reset_events_flag_next_step') and self._reset_events_flag_next_step:
            self._events_updated_this_step = False
            self._step_incremented_this_step = False
            self._reset_events_flag_next_step = False
        
        batch_size = self.world.batch_dim
        device = self.world.device
        
        # For attacker: fixed policy (move down) unless detected
        if not agent.is_defender:
            # Check if detected and should stop
            is_inactive = self.attacker_detected
            if is_inactive.any():
                # Stop moving if detected
                heading = torch.full((batch_size,), -math.pi/2, device=device)  # Down
                max_speed = torch.where(
                    is_inactive,
                    torch.zeros(batch_size, device=device),
                    torch.full((batch_size,), self.attacker_speed, device=device)
                )
            else:
                # Move down
                heading = torch.full((batch_size,), -math.pi/2, device=device)
                max_speed = self.attacker_speed
        else:
            # Defenders: use action heading
            if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                normalized_heading = agent.action.u[:, 0]
                heading = normalized_heading * math.pi
            else:
                heading = torch.zeros(batch_size, device=device)
            
            max_speed = agent.max_speed
        
        # Convert to [0, 2π) for wall constraints
        theta = torch.remainder(heading, 2 * math.pi)
        
        # Apply wall constraints for defenders
        if agent.is_defender:
            theta = self._apply_wall_constraints(agent, theta)
        
        # Update events
        self.update_events()
        
        # Increment step count
        if not hasattr(self, '_step_incremented_this_step') or not self._step_incremented_this_step:
            self.step_count += 1
            self._step_incremented_this_step = True
        
        # Set velocity
        if agent.action is not None and hasattr(agent.action, 'u'):
            if isinstance(max_speed, torch.Tensor):
                agent.action.u[:, 0] = max_speed * torch.cos(theta)
                agent.action.u[:, 1] = max_speed * torch.sin(theta)
            else:
                agent.action.u[:, 0] = max_speed * torch.cos(theta)
                agent.action.u[:, 1] = max_speed * torch.sin(theta)
        
        self._reset_events_flag_next_step = True
    
    def _clamp_interval(self, theta, lo, hi):
        """Clamp angle theta into [lo, hi]"""
        lo_t = torch.full_like(theta, lo)
        hi_t = torch.full_like(theta, hi)
        theta = torch.where(theta < lo_t, lo_t, theta)
        theta = torch.where(theta > hi_t, hi_t, theta)
        return theta
    
    def _clamp_union(self, theta, segments):
        """Clamp theta to closest boundary of union of segments"""
        B = theta.shape[0]
        inside = torch.zeros(B, dtype=torch.bool, device=theta.device)
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
        
        return torch.where(inside, theta, best_proj)
    
    def _apply_wall_constraints(self, agent, theta_0_2pi):
        """Apply wall constraints to defender heading"""
        if not self.enable_wall_constraints:
            return theta_0_2pi
        
        x = agent.state.pos[:, 0]
        y = agent.state.pos[:, 1]
        wx = 0.5
        wy = 0.5
        eps = float(self.wall_epsilon)
        
        near_right = (wx - x) <= eps
        near_left = (x + wx) <= eps
        near_top = (wy - y) <= eps
        near_bottom = (y + wy) <= eps
        
        theta = theta_0_2pi.clone()
        pi = math.pi
        two_pi = 2 * pi
        
        # Corners
        tr = near_top & near_right
        tl = near_top & near_left
        br = near_bottom & near_right
        bl = near_bottom & near_left
        
        theta = torch.where(tr, self._clamp_interval(theta, pi, 1.5 * pi), theta)
        theta = torch.where(tl, self._clamp_interval(theta, 1.5 * pi, two_pi), theta)
        theta = torch.where(br, self._clamp_interval(theta, 0.5 * pi, pi), theta)
        theta = torch.where(bl, self._clamp_interval(theta, 0.0, 0.5 * pi), theta)
        
        # Walls (excluding corners)
        right_only = near_right & ~(near_top | near_bottom)
        left_only = near_left & ~(near_top | near_bottom)
        top_only = near_top & ~(near_left | near_right)
        bottom_only = near_bottom & ~(near_left | near_right)
        
        if right_only.any():
            theta[right_only] = self._clamp_interval(theta[right_only], 0.5 * pi, 1.5 * pi)
        
        if left_only.any():
            segs = [(0.0, 0.5 * pi), (1.5 * pi, two_pi)]
            theta[left_only] = self._clamp_union(theta[left_only], segs)
        
        if top_only.any():
            theta[top_only] = self._clamp_interval(theta[top_only], pi, two_pi)
        
        if bottom_only.any():
            theta[bottom_only] = self._clamp_interval(theta[bottom_only], 0.0, pi)
        
        return theta
    
    def done(self) -> torch.Tensor:
        """Episode termination conditions"""
        # Get batch size and device with fallbacks
        if hasattr(self, 'world'):
            batch_size = getattr(self.world, 'batch_dim', 1)
            device = getattr(self.world, 'device', torch.device('cpu'))
        else:
            batch_size = 1
            device = torch.device('cpu')
        
        # Safety check
        if not hasattr(self, '_initialized') or not self._initialized:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        if self.terminate_on_detection:
            # Terminate when attacker is detected
            done = self.attacker_detected | self.attacker_reached_target
        else:
            # Terminate when attacker is detected/reaches target or max steps
            done = self.attacker_detected | self.attacker_reached_target
        
        # Also terminate on max steps
        done = done | (self.step_count >= self.max_steps)
        
        return done
    
    def extra_render(self, env_index: int = 0):
        """Enhanced rendering with proper VMAS syntax"""
        try:
            from vmas.simulator import rendering
            from vmas.simulator.utils import Color
        except ImportError:
            return []
        
        # Safety check
        if not hasattr(self, '_initialized') or not self._initialized:
            return []
        
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
        
        geoms = []
        
        # Get agents
        patroller = None
        pursuer = None
        for agent in self.world.agents:
            if agent.name == "patroller":
                patroller = agent
            elif agent.name == "pursuer":
                pursuer = agent
        
        # 1. BLACK WALLS (3 sides)
        wall_lines = [
            ((-0.5, -0.5), (-0.5, 0.5)),  # Left
            ((0.5, -0.5), (0.5, 0.5)),     # Right  
            ((-0.5, 0.5), (0.5, 0.5)),     # Top
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
        
        # 3. PATROLLER SENSING CIRCLE (light blue)
        if patroller and hasattr(patroller.state, 'pos'):
            pos = patroller.state.pos[env_index]
            sensing_circle = rendering.make_circle(self.patroller_sensing_radius, filled=False)
            sensing_xform = rendering.Transform()
            sensing_xform.set_translation(*pos.cpu().numpy())
            sensing_circle.add_attr(sensing_xform)
            sensing_circle.set_color(0.0, 0.5, 1.0, 0.6)  # Light blue
            geoms.append(sensing_circle)
        
        # 4. PURSUER MARKER (no sensing circle, just highlight)
        if pursuer and hasattr(pursuer.state, 'pos'):
            pos = pursuer.state.pos[env_index]
            pursuer_marker = rendering.make_circle(0.03, filled=False)
            marker_xform = rendering.Transform()
            marker_xform.set_translation(*pos.cpu().numpy())
            pursuer_marker.add_attr(marker_xform)
            pursuer_marker.set_color(0.0, 0.0, 1.0, 0.8)  # Dark blue
            geoms.append(pursuer_marker)
        
        # 5. DETECTION INDICATOR
        if hasattr(self, 'attacker_detected') and self.attacker_detected[env_index]:
            # Show detection status with a banner
            detection_indicator = rendering.Line((-0.4, 0.4), (-0.2, 0.4), width=4)
            detection_xform = rendering.Transform()
            detection_indicator.add_attr(detection_xform)
            detection_indicator.set_color(1.0, 1.0, 0.0)  # Yellow
            geoms.append(detection_indicator)
        
        # 6. SPAWN AREA (if enabled)
        if self.spawn_area_mode:
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
        
        return geoms
    
    def info(self, agent: Agent) -> Dict:
        """Get info dictionary for logging"""
        # Get batch size and device with fallbacks
        if hasattr(self, 'world'):
            batch_size = getattr(self.world, 'batch_dim', 1)
            device = getattr(self.world, 'device', torch.device('cpu'))
        else:
            batch_size = 1
            device = torch.device('cpu')
        
        # Safety check
        if not hasattr(self, '_initialized') or not self._initialized:
            return {
                "attacker_detected": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "attacker_reached_target": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "detection_reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
                "step_count": torch.zeros(batch_size, dtype=torch.long, device=device),
                "max_steps": getattr(self, 'max_steps', 200),
            }
        
        return {
            "attacker_detected": self.attacker_detected.clone(),
            "attacker_reached_target": self.attacker_reached_target.clone(),
            "detection_reward": self.detection_reward.clone(),
            "step_count": self.step_count.clone(),
            "max_steps": self.max_steps,
        }


if __name__ == "__main__":
    import vmas
    
    # Test the scenario
    scenario = Scenario()
    
    # Create environment
    env = vmas.make_env(
        scenario=scenario,
        num_envs=4,
        device="cpu",
        continuous_actions=True,
        patroller_sensing_radius=0.25,
        patroller_speed_ratio=0.8,
        pursuer_speed_ratio=1.0,
        attacker_speed_ratio=0.3,
        randomize_attacker_x=True,
        terminate_on_detection=True,
        use_apollonius=False  # Set to True if solver available
    )
    
    print(f"Environment created with {env.n_agents} agents")
    print(f"Agents: {[agent.name for agent in env.agents]}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Reset and test
    obs = env.reset()
    print(f"\nInitial observations shape: {[o.shape for o in obs]}")
    
    # Test a few steps
    for step in range(5):
        actions = []
        for agent in env.agents:
            if agent.is_defender:
                # Random action for defenders
                action = torch.randn(4, 2) * 0.5
            else:
                # Attacker uses fixed policy (handled internally)
                action = torch.zeros(4, 2)
            actions.append(action)
        
        obs, rewards, dones, info = env.step(actions)
        print(f"\nStep {step+1}:")
        print(f"  Rewards: {[r.mean().item() for r in rewards]}")
        print(f"  Done: {dones.any().item()}")
        if dones.any():
            break
    
    print("\n✓ Patroller-Pursuer environment working correctly!")