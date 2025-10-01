"""
Patroller-Pursuer Target Defense Environment V2 - Smart Attacker Version
Patroller detects, attacker uses smart Apollonius policy based on PURSUER position after detection
Game continues until capture (pursuer catches attacker)
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
    from apollonius_solver import solve_apollonius_optimization
    APOLLONIUS_AVAILABLE = True
except ImportError:
    APOLLONIUS_AVAILABLE = False
    print("Warning: apollonius_solver not available. Using fallback rewards.")


@dataclass
class TaskConfig:
    """Configuration for End2End Patrol Target Defense task"""
    max_steps: int = 1000
    patroller_sensing_radius: float = 0.35
    patroller_speed_ratio: float = 1.0
    pursuer_speed_ratio: float = 1.0
    attacker_speed_ratio: float = 0.2
    target_distance: float = 0.05
    randomize_attacker_x: bool = True
    spawn_area_mode: bool = True
    spawn_area_width: float = 0.1
    enable_wall_constraints: bool = False
    wall_epsilon: float = 0.03
    use_apollonius: bool = True
    terminate_on_detection: bool = False  # Key difference - continue until capture
    capture_distance: float = 0.07


def compute_apollonius_circle(pos_a, pos_d, speed_ratio):
    """
    Compute the Apollonius circle for pursuit-evasion game.
    The Apollonius circle is the locus of points P such that |PA|/|PD| = speed_ratio.
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
    Patroller-Pursuer Target Defense V2 - Smart Attacker
    
    Mechanics:
    - Patroller detects attacker with sensing radius
    - Once detected, attacker uses smart Apollonius policy based on PURSUER position
    - Pursuer captures attacker when within capture distance
    - Episode ends on capture or target reached
    - Both defenders share observations after detection
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """Create the world with patroller, pursuer, and smart attacker"""
        
        print("="*80)
        print("🎯 PATROLLER-PURSUER SMART ATTACKER ENVIRONMENT")
        print("="*80)
        
        # Extract parameters
        patroller_sensing_radius = kwargs.get('patroller_sensing_radius', 0.25)
        capture_distance = kwargs.get('capture_distance', 0.07)  # New parameter
        patroller_speed_ratio = kwargs.get('patroller_speed_ratio', 0.8)
        pursuer_speed_ratio = kwargs.get('pursuer_speed_ratio', 1.0)
        attacker_speed_ratio = kwargs.get('attacker_speed_ratio', 0.3)
        target_distance = kwargs.get('target_distance', 0.05)
        patroller_color = kwargs.get('patroller_color', (0.0, 0.5, 1.0))
        pursuer_color = kwargs.get('pursuer_color', (0.0, 0.0, 1.0))
        attacker_color = kwargs.get('attacker_color', (1.0, 0.0, 0.0))
        randomize_attacker_x = kwargs.get('randomize_attacker_x', True)
        max_steps = kwargs.get('max_steps', 500)  # Longer episodes
        enable_wall_constraints = kwargs.get('enable_wall_constraints', True)
        wall_epsilon = kwargs.get('wall_epsilon', 0.03)
        use_apollonius = kwargs.get('use_apollonius', True)
        spawn_area_mode = kwargs.get('spawn_area_mode', False)
        spawn_area_width = kwargs.get('spawn_area_width', 0.2)
        
        # Store parameters
        self.batch_dim = batch_dim
        self.device = device
        self.patroller_sensing_radius = patroller_sensing_radius
        self.capture_distance = capture_distance
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
        self.max_steps = max_steps
        
        # Calculate actual speeds
        base_speed = 0.05
        self.patroller_speed = base_speed * patroller_speed_ratio
        self.pursuer_speed = base_speed * pursuer_speed_ratio
        self.attacker_speed = base_speed * attacker_speed_ratio
        
        print(f"📊 Configuration:")
        print(f"   Patroller: speed={self.patroller_speed:.3f}, sensing={patroller_sensing_radius}")
        print(f"   Pursuer: speed={self.pursuer_speed:.3f}, capture={capture_distance}")
        print(f"   Attacker: speed={self.attacker_speed:.3f}, smart policy after detection")
        print(f"   Max steps: {self.max_steps}")
        
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
        
        # Create agents
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
        pursuer.sensing_radius = 0.0
        world.add_agent(pursuer)
        
        attacker = Agent(
            name="attacker",
            shape=Sphere(radius=0.02),
            color=attacker_color,
            max_speed=self.attacker_speed,
            rotatable=False,
            silent=True,
            controllable=False  # Non-controllable with smart policy
        )
        attacker.is_defender = False
        attacker.is_patroller = False
        attacker.sensing_radius = 0.0
        world.add_agent(attacker)
        
        # Initialize tracking variables
        self.attacker_detected = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.attacker_position_known = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.attacker_captured = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.attacker_reached_target = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.capture_reward = torch.zeros(batch_dim, device=device)
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        
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
        patroller_x = self._world_to_vmas(0.5)
        patroller_y = self._world_to_vmas(0.0)
        
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
            self.attacker_captured[:] = False
            self.attacker_reached_target[:] = False
            self.capture_reward[:] = 0.0
            self.step_count[:] = 0
        else:
            self.attacker_detected[env_index] = False
            self.attacker_position_known[env_index] = False
            self.attacker_captured[env_index] = False
            self.attacker_reached_target[env_index] = False
            self.capture_reward[env_index] = 0.0
            self.step_count[env_index] = 0
        
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
    
    def _compute_smart_attacker_heading(self, batch_size, device):
        """
        Compute smart attacker heading using Apollonius circle based on PURSUER position
        When attacker is detected, it computes optimal escape route from pursuer
        """
        pursuer = None
        attacker = None
        for agent in self.world.agents:
            if agent.name == "pursuer":
                pursuer = agent
            elif agent.name == "attacker":
                attacker = agent
        
        if not pursuer or not attacker:
            return torch.full((batch_size,), -math.pi/2, device=device)
        
        heading = torch.full((batch_size,), -math.pi/2, device=device)  # Default: straight down
        
        for env_idx in range(batch_size):
            # Check if detected
            is_detected = self.attacker_detected[env_idx].item()
            
            if not is_detected:
                # NOT DETECTED: Move straight down
                heading[env_idx] = -math.pi/2
            else:
                # DETECTED: Use Apollonius circle to escape from PURSUER
                try:
                    attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                    attacker_pos_world = self._vmas_to_world(attacker_pos_vmas)
                    
                    # Use PURSUER position for Apollonius
                    pursuer_pos_vmas = pursuer.state.pos[env_idx].cpu().numpy()
                    pursuer_pos_world = self._vmas_to_world(pursuer_pos_vmas)
                    
                    # Compute Apollonius circle with pursuer as threat
                    center, radius, lowest_point = compute_apollonius_circle(
                        pos_a=attacker_pos_world,
                        pos_d=pursuer_pos_world,
                        speed_ratio=self.attacker_speed_ratio / self.pursuer_speed_ratio
                    )
                    
                    # Move toward lowest point
                    target_vmas = self._world_to_vmas(lowest_point)
                    attacker_vmas = attacker.state.pos[env_idx].cpu().numpy()
                    
                    direction = target_vmas - attacker_vmas
                    if np.linalg.norm(direction) > 1e-6:
                        heading_angle = np.arctan2(direction[1], direction[0])
                        heading[env_idx] = torch.tensor(heading_angle, device=device)
                    else:
                        heading[env_idx] = -math.pi/2
                        
                except Exception as e:
                    # On any error, default to straight down
                    heading[env_idx] = -math.pi/2
        
        return heading
    
    def reward(self, agent: Agent) -> torch.Tensor:
        """Reward based on capture with Apollonius payoff"""
        batch_size = self.world.batch_dim
        device = self.world.device
        
        self.update_events()
        
        r = torch.zeros(batch_size, device=device)
        
        if agent.is_defender:
            # Reward on capture
            newly_captured = self.attacker_captured & (self.capture_reward != 0)
            r = torch.where(newly_captured, self.capture_reward, r)
            
            # Penalty if attacker reaches target
            r = torch.where(self.attacker_reached_target, 
                           torch.full_like(r, 0.0), r)
        
        return r
    
    def observation(self, agent: Agent) -> torch.Tensor:
        """Observation with shared visibility after detection"""
        batch_size = self.world.batch_dim
        device = self.world.device
        
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
        
        obs = torch.zeros(batch_size, 6, device=device)
        
        # Own position
        obs[:, 0:2] = agent.state.pos
        
        # Other defender position (always visible)
        if agent.name == "patroller" and pursuer:
            obs[:, 2:4] = pursuer.state.pos
        elif agent.name == "pursuer" and patroller:
            obs[:, 2:4] = patroller.state.pos
        
        # Attacker position
        if attacker:
            if agent.name == "patroller":
                # Patroller can sense within radius OR if already detected
                dist = torch.norm(agent.state.pos - attacker.state.pos, dim=-1)
                can_sense = (dist <= self.patroller_sensing_radius) | self.attacker_position_known
                obs[can_sense, 4:6] = attacker.state.pos[can_sense]
            else:  # pursuer
                # Pursuer sees attacker after detection (shared visibility)
                obs[self.attacker_position_known, 4:6] = attacker.state.pos[self.attacker_position_known]
        
        return obs
    
    def update_events(self):
        """Update detection and capture events"""
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
        
        # Check for detection by patroller
        dist_to_patroller = torch.norm(patroller.state.pos - attacker.state.pos, dim=-1)
        newly_detected = (dist_to_patroller <= self.patroller_sensing_radius) & ~self.attacker_detected
        
        if newly_detected.any():
            self.attacker_detected |= newly_detected
            self.attacker_position_known |= newly_detected  # Share visibility
        
        # Check for CAPTURE by pursuer (only after detection)
        dist_to_pursuer = torch.norm(pursuer.state.pos - attacker.state.pos, dim=-1)
        can_capture = self.attacker_detected  # Must be detected first
        newly_captured = (dist_to_pursuer <= self.capture_distance) & can_capture & ~self.attacker_captured
        
        if newly_captured.any():
            self.attacker_captured |= newly_captured
            
            # Compute Apollonius payoff at capture
            for env_idx in torch.where(newly_captured)[0]:
                if self.use_apollonius:
                    attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                    attacker_pos = self._vmas_to_world(attacker_pos_vmas)
                    
                    pursuer_pos_vmas = pursuer.state.pos[env_idx].cpu().numpy()
                    pursuer_pos = self._vmas_to_world(pursuer_pos_vmas)
                    
                    try:
                        result = solve_apollonius_optimization(
                            attacker_pos=attacker_pos,
                            defender_positions=[pursuer_pos],
                            nu=self.pursuer_speed / self.attacker_speed
                        )

                        if result['success']:
                            # Reward is distance from target (y=0) to attacker position
                            # Higher y = better reward (caught early)
                            self.capture_reward[env_idx] = result['defender_payoff']
                        else:
                            self.capture_reward[env_idx] = float(attacker_pos[1])
                    except:
                        self.capture_reward[env_idx] = float(attacker_pos[1])
                else:
                    attacker_y = self._vmas_to_world(attacker.state.pos[env_idx, Y].item())
                    self.capture_reward[env_idx] = attacker_y
            
            # Stop attacker on capture
            attacker.state.vel[newly_captured] = 0
        
        # Check if attacker reached target
        target_y_vmas = self._world_to_vmas(0.0)
        reached = (attacker.state.pos[:, Y] <= target_y_vmas + self.target_distance) & ~self.attacker_captured
        self.attacker_reached_target |= reached
        
        self._events_updated_this_step = True
    
    def process_action(self, agent: Agent):
        """Process actions with smart attacker policy"""
        if hasattr(self, '_reset_events_flag_next_step') and self._reset_events_flag_next_step:
            self._events_updated_this_step = False
            self._step_incremented_this_step = False
            self._reset_events_flag_next_step = False
        
        batch_size = self.world.batch_dim
        device = self.world.device
        
        # For attacker: smart policy after detection
        if not agent.is_defender:
            # Ensure action structure exists
            if agent.action is None or not hasattr(agent.action, 'u'):
                from vmas.simulator.core import TorchVectorizedObject
                agent.action = TorchVectorizedObject()
                agent.action.u = torch.zeros((batch_size, 2), device=device)
            
            # Check if captured or reached target
            is_inactive = self.attacker_captured | self.attacker_reached_target
            
            # Use smart heading based on detection status
            heading = self._compute_smart_attacker_heading(batch_size, device)
            
            # Set speed (0 if inactive)
            max_speed = torch.where(
                is_inactive,
                torch.zeros(batch_size, device=device),
                torch.full((batch_size,), self.attacker_speed, device=device)
            )
        else:
            # Defenders: use action heading
            if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                normalized_heading = agent.action.u[:, 0]
                heading = normalized_heading * math.pi
            else:
                heading = torch.zeros(batch_size, device=device)
            
            max_speed = agent.max_speed
        
        theta = torch.remainder(heading, 2 * math.pi)
        
        # Apply wall constraints for defenders
        if agent.is_defender and self.enable_wall_constraints:
            theta = self._apply_wall_constraints(agent, theta)
        
        # Update events
        self.update_events()
        
        # Increment step count
        if not hasattr(self, '_step_incremented_this_step') or not self._step_incremented_this_step:
            self.step_count += 1
            self._step_incremented_this_step = True
        
        # Set velocity
        if isinstance(max_speed, torch.Tensor):
            agent.action.u[:, 0] = max_speed * torch.cos(theta)
            agent.action.u[:, 1] = max_speed * torch.sin(theta)
        else:
            agent.action.u[:, 0] = max_speed * torch.cos(theta)
            agent.action.u[:, 1] = max_speed * torch.sin(theta)
        
        self._reset_events_flag_next_step = True
    
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
        
        theta = torch.where(tr, torch.clamp(theta, pi, 1.5 * pi), theta)
        theta = torch.where(tl, torch.clamp(theta, 1.5 * pi, two_pi), theta)
        theta = torch.where(br, torch.clamp(theta, 0.5 * pi, pi), theta)
        theta = torch.where(bl, torch.clamp(theta, 0.0, 0.5 * pi), theta)
        
        return theta
    
    def done(self) -> torch.Tensor:
        """Episode ends on capture or target reached"""
        done = self.attacker_captured | self.attacker_reached_target
        done = done | (self.step_count >= self.max_steps)
        return done
    
    def info(self, agent: Agent) -> Dict:
        """Get info dictionary"""
        return {
            "attacker_detected": self.attacker_detected.clone(),
            "attacker_captured": self.attacker_captured.clone(),
            "attacker_reached_target": self.attacker_reached_target.clone(),
            "capture_reward": self.capture_reward.clone(),
            "step_count": self.step_count.clone(),
            "max_steps": self.max_steps,
        }
    
    def extra_render(self, env_index: int = 0):
        """Enhanced rendering showing Apollonius circles after detection"""
        try:
            from vmas.simulator import rendering
            from vmas.simulator.utils import Color
        except ImportError:
            return []
        
        geoms = []
        
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
        
        # Walls and target line
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
        
        target_line = rendering.Line((-0.5, -0.5), (0.5, -0.5), width=10)
        target_xform = rendering.Transform()
        target_line.add_attr(target_xform)
        target_line.set_color(*Color.GREEN.value)
        geoms.append(target_line)
        
        # Patroller sensing circle
        if patroller:
            pos = patroller.state.pos[env_index]
            sensing_circle = rendering.make_circle(self.patroller_sensing_radius, filled=False)
            sensing_xform = rendering.Transform()
            sensing_xform.set_translation(*pos.cpu().numpy())
            sensing_circle.add_attr(sensing_xform)
            sensing_circle.set_color(0.0, 0.5, 1.0, 0.6)
            geoms.append(sensing_circle)
        
        # Pursuer capture circle
        if pursuer:
            pos = pursuer.state.pos[env_index]
            capture_circle = rendering.make_circle(self.capture_distance, filled=False)
            capture_xform = rendering.Transform()
            capture_xform.set_translation(*pos.cpu().numpy())
            capture_circle.add_attr(capture_xform)
            capture_circle.set_color(1.0, 0.0, 1.0, 0.8)
            geoms.append(capture_circle)
        
        # Apollonius circle visualization (after detection)
        if self.attacker_detected[env_index] and pursuer and attacker:
            try:
                attacker_pos_vmas = attacker.state.pos[env_index].cpu().numpy()
                attacker_pos_world = self._vmas_to_world(attacker_pos_vmas)
                
                pursuer_pos_vmas = pursuer.state.pos[env_index].cpu().numpy()
                pursuer_pos_world = self._vmas_to_world(pursuer_pos_vmas)
                
                center, radius, lowest_point = compute_apollonius_circle(
                    pos_a=attacker_pos_world,
                    pos_d=pursuer_pos_world,
                    speed_ratio=self.attacker_speed_ratio / self.pursuer_speed_ratio
                )
                
                # Draw Apollonius circle
                center_vmas = self._world_to_vmas(center)
                apo_circle = rendering.make_circle(radius, filled=False)
                apo_xform = rendering.Transform()
                apo_xform.set_translation(*center_vmas)
                apo_circle.add_attr(apo_xform)
                apo_circle.set_color(0.5, 0.0, 1.0, 0.8)  # Purple
                geoms.append(apo_circle)
                
                # Draw lowest point
                lowest_point_vmas = self._world_to_vmas(lowest_point)
                lowest_marker = rendering.make_circle(0.03, filled=True)
                lowest_xform = rendering.Transform()
                lowest_xform.set_translation(*lowest_point_vmas)
                lowest_marker.add_attr(lowest_xform)
                lowest_marker.set_color(0.0, 1.0, 1.0, 1.0)  # Cyan
                geoms.append(lowest_marker)
                
            except Exception:
                pass
        
        return geoms


if __name__ == "__main__":
    import vmas
    
    scenario = Scenario()
    
    env = vmas.make_env(
        scenario=scenario,
        num_envs=4,
        device="cpu",
        continuous_actions=True,
        patroller_sensing_radius=0.25,
        capture_distance=0.07,
        patroller_speed_ratio=0.8,
        pursuer_speed_ratio=1.0,
        attacker_speed_ratio=0.3,
        randomize_attacker_x=True,
        use_apollonius=False  # Set to True if solver available
    )
    
    print(f"✓ Smart Attacker Patroller-Pursuer environment ready!")
    print(f"  - Patroller detects at {scenario.patroller_sensing_radius}")
    print(f"  - Attacker uses smart policy after detection (based on pursuer)")
    print(f"  - Pursuer captures at {scenario.capture_distance}")
    print(f"  - Game continues until capture")