import numpy as np
from scipy.stats import norm
from dataclasses import dataclass

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0

@dataclass
class Env:
    job_num: int
    machine_num: int
    family_num: int
    family: np.ndarray
    job_to_family: dict
    deadline: np.ndarray
    duration: np.ndarray
    spec_cdf: np.ndarray
    scaled_v: np.ndarray
    # scaled_cvar: np.ndarray

class Environment:
    def __init__(self, job_num, machine_num, family_num) -> None:
        """
        Initialize the Environment with the number of jobs, machines, and families.
        """
        self.job_num = job_num  # Total number of jobs
        self.machine_num = machine_num  # Total number of machines
        self.family_num = family_num  # Total number of job families

    def _calculate_conditional_means(self, mean_matrix, std_matrix, c):
        """
        Compute the conditional mean of a truncated normal distribution.
        """
        z_matrix = (c - mean_matrix) / std_matrix
        phi_matrix = norm.pdf(z_matrix)
        Phi_matrix = norm.cdf(z_matrix)
        conditional_means = mean_matrix + std_matrix * (phi_matrix / (1 - Phi_matrix))
        conditional_means[Phi_matrix >= 1] = 0  # If c is too large, set mean to zero
        return conditional_means
    
    def _deadline_distribution(self, rng, tight, a: float, b_a: float):
        """
        Generate deadlines based on the given tightness level.
        """
        b = a + b_a
        if tight in ['low', 'very_low', 'base']:
            return int(rng.triangular(left=a, mode=b, right=b))
        elif tight == 'high':
            if rng.random() < 0.25:
                return rng.integers(a, b)
            else:
                return int(rng.triangular(left=a, mode=b, right=b))
        elif tight == 'very_high':
            if rng.random() < 0.5:
                return rng.integers(a, b)
            else:
                return int(rng.triangular(left=a, mode=b, right=b))
        else:
            raise TypeError("Invalid tightness type")
        
    def _quality_bound(self, quality: str):
        """
        Define quality bounds based on the quality level.
        """
        quality_levels = {
            'very_low': (0.076, 0.046, 0.15, 0.2),
            'low': (0.078, 0.047, 0.15, 0.2),
            'base': (0.08, 0.048, 0.15, 0.2),
            'high': (0.082, 0.049, 0.15, 0.2),
            'very_high': (0.084, 0.05, 0.15, 0.2)
        }
        return quality_levels.get(quality, None)

    def reset(self, seed=42, tight: str = 'high', quality: str = 'high'):
        """
        Reset the environment with a new seed, job tightness, and quality level.
        """
        rng = np.random.default_rng(seed)  # Initialize random generator
        self.family = np.array_split(np.arange(self.job_num), self.family_num)
        
        # Map jobs to their respective families
        self.job_to_family = {}
        for family_index, jobs in enumerate(self.family):
            for job in jobs:
                self.job_to_family[job] = family_index
        
        # Initialize job durations and deadlines
        self.deadline = np.zeros(self.job_num, dtype=int)
        self.duration = rng.integers(1, 20, size=self.job_num)
        
        # Compute deadline distribution parameters
        a = 5 * self.job_num / self.machine_num
        if tight == 'very_low':
            a *= 2
        elif tight == 'low':
            a *= 1.5
        elif tight not in ['base', 'high', 'very_high']:
            raise TypeError("Invalid tightness type")
        
        b_a = np.mean(self.duration) * self.job_num / self.machine_num
        
        # Assign deadlines for each job
        for i in range(self.job_num):
            self.deadline[i] = self._deadline_distribution(rng, tight, a, b_a)
            if self.deadline[i] < self.duration[i]:
                print("Warning: Unusual instance detected, adjusting deadline.")
                self.deadline[i] = self.duration[i]
        
        # Assign quality parameters
        mean, std, spec_a, spec_b = self._quality_bound(quality)
        self.spec = rng.uniform(spec_a, spec_b, size=self.job_num)  # Random specification values
        self.mean = np.full((self.job_num, self.machine_num), mean)
        self.std = np.full((self.job_num, self.machine_num), std)
        
        # Compute probability and scaled values
        self.spec_cdf = np.round(norm.cdf(self.spec[:, np.newaxis], loc=self.mean, scale=self.std), decimals=3)
        self.scaled_v = np.round(self._calculate_conditional_means(self.mean, self.std, self.spec[:, np.newaxis]) / self.spec[:, np.newaxis], decimals=3)
        
        self._create_state()
        return self.env

    def _create_state(self):
        """
        Create the environment's state representation.
        """
        self.env = Env(job_num = self.job_num, 
                       machine_num = self.machine_num, 
                       family_num = self.family_num, 
                       family = self.family,
                       job_to_family = self.job_to_family,
                       deadline = self.deadline, 
                       duration = self.duration, 
                       spec_cdf = self.spec_cdf, 
                       scaled_v = self.scaled_v,
                       # scaled_cvar = self.scaled_cvar,
                       )
        