import numpy as np
import gym

class mmWaveEnv(gym.Env):
    def __init__(self):
        self.Horizon = int((3600/3)*4)        # 4 hours. Time slot is 3 seconds
        
        self.road_width = 7                   # in meters
        self.road_length = 100                # in meters
        self.bs_location = np.array([50,7])   # in meters
        
        self.Nbeams = 10                                                        # Number of mmWave beams
        self.narrow_beam_angle = int(180/self.Nbeams)                           # in degrees
        self.beam_angle_ratio = 2                                               # MUST be greater than 1
        self.broad_beam_angle = self.beam_angle_ratio*self.narrow_beam_angle    # in degrees
        
        # Computing the beam directions for both narrow and broad beams
        self.beam_directions = self.compute_beam_directions(self.narrow_beam_angle)
        
        self.Ngps = 5                  # Number of GPS observations per measurement
        self.gps_sample_time = 0.2     # in seconds. Time required by GPS to collect one sample.
        self.bs_response_time = 0.5    # in seconds.
        
        self.car_direction_prob = np.array([0.5, 0.5])
        
        self.max_speed = 75  # in kmph
        self.velocity_bins = np.array([[0,20],[20,40],[40,60],[60,self.max_speed]], dtype=np.float32) # in kmph
        self.acceleration_bins = np.array([[-30,-10],[-10,-5],[-5,0],[0,4],[4,8],[8,13]], dtype=np.float32) # in kmph per sec
        self.Nv_bins = len(self.velocity_bins)
        self.Na_bins = len(self.acceleration_bins)
        self.velocity_bins_ix = np.arange(self.Nv_bins)
        self.acceleration_bins_ix = np.arange(self.Na_bins)
        
        # self.velocity_prob = np.array([0.2,0.4,0.3,0.1], dtype=np.float32)
        self.velocity_prob = np.array([0.2,0.1,0.3,0.4], dtype=np.float32)
        self.acceleration_prob = np.zeros((self.Nv_bins,self.Na_bins), dtype=np.float32)
        self.acceleration_prob[0] = [0.02,0.05,0.10,0.2,0.4,0.23]
        self.acceleration_prob[1] = [0.03,0.09,0.16,0.3,0.3,0.12]
        self.acceleration_prob[2] = [0.03,0.12,0.20,0.3,0.23,0.12]
        self.acceleration_prob[3] = [0.09,0.16,0.23,0.32,0.12,0.09]
        for i in range(self.Nv_bins):
            self.acceleration_prob[i,:] = self.acceleration_prob[i,:]/np.sum(self.acceleration_prob[i,:])
        
        r_max = np.sqrt((self.bs_location[0]+30)**2+self.road_width**2)
        self.power_bias = 2*np.log10(r_max)+np.log10(self.beam_angle_ratio)
        
        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Discrete(self.Nbeams)])  # Action space
        
        self.delta = self.max_speed*(1000/3600)*(self.gps_sample_time*(self.Ngps-1)+self.bs_response_time)
        x_space = gym.spaces.Box(-self.delta*np.ones(self.Ngps, dtype=np.float32), (self.delta+self.road_length)*np.ones(self.Ngps, dtype=np.float32))
        y_space = gym.spaces.Box(np.zeros(self.Ngps, dtype=np.float32), self.road_width*np.ones(self.Ngps, dtype=np.float32))
        self.observation_space = gym.spaces.Tuple((x_space, y_space))        
        
        self.observation = (np.zeros(self.Ngps), np.zeros(self.Ngps))    # Used to store the current observation (history of car position measurement)
        self.car_position = np.array([0,0])                              # Used to store the current position of the car
        self.t = 0                                                       # Current time slot.
        
    def step(self, action):
        reward, correct = self.reward_func(action)
        
        truncated = False
        info = {}
        
        if self.t>=self.Horizon:
            terminated = True
            return -1, reward, correct, terminated, truncated, info
        else:
            terminated = False
            self.generate_observation()
            return self.observation, reward, correct, terminated, truncated, info
    
    def reset(self):
        self.t = 0
        self.generate_observation()
        
        return self.observation
    
    def render(self):
        pass
        
    def compute_beam_directions(self, beam_angle):
        return np.array([(180-beam_angle)*n/(self.Nbeams-1)+(beam_angle/2) for n in range(self.Nbeams)])
    
    def generate_observation(self):
        # Current direction
        dir_cur = np.random.choice([-1,1], p=self.car_direction_prob)
        
        # Generating y-coordinates of the car
        if dir_cur==1:
            mid = (self.road_width/2)/2
            delta = (self.road_width/2)/10
            y_position = np.random.uniform(low=mid-delta, high=mid+delta, size=(self.Ngps+1))            
        else:
            mid = (self.road_width/2)/2 + (self.road_width/2)
            delta = (self.road_width/2)/10
            y_position = np.random.uniform(low=mid-delta, high=mid+delta, size=(self.Ngps+1))
            
        # Generating x-coordinates of the car
        x_position = np.zeros(self.Ngps+1)
        
        x_position[0] = np.random.uniform(low=0, high=self.road_length)
            
        v_ix = np.random.choice(self.velocity_bins_ix, p=self.velocity_prob)
        velocity = (self.velocity_bins[v_ix][1]-self.velocity_bins[v_ix][0])*np.random.uniform()+self.velocity_bins[v_ix][0]
        
        for m in range(1,self.Ngps+1,1):
            if m==1:
                x_position[m] = x_position[m-1]-dir_cur*velocity*(1000/3600)*self.bs_response_time
            else:
                x_position[m] = x_position[m-1]-dir_cur*velocity*(1000/3600)*self.gps_sample_time
            
            v_ix = self.find_bin(velocity)
            a_ix = np.random.choice(self.acceleration_bins_ix, p=self.acceleration_prob[v_ix])
            acceleration = (self.acceleration_bins[a_ix][1]-self.acceleration_bins[a_ix][0])*np.random.uniform()+self.acceleration_bins[a_ix][0]
            if m==self.Ngps:
                velocity += acceleration*self.bs_response_time
            else:
                velocity += acceleration*self.gps_sample_time
            velocity = np.clip(velocity, 0, self.max_speed)
            
        x_position = np.clip(x_position, a_min=-self.delta, a_max=self.delta+self.road_length)
        
        # Generating current position of the car and the observation        
        x_pos = x_position[1:].reshape(self.Ngps,1)
        y_pos = y_position[1:].reshape(self.Ngps,1)
        pos = np.hstack((x_pos, y_pos))
        
        noise = np.random.uniform(low=-2/4, high=2/4, size=(self.Ngps,2))
        noise += np.random.uniform(low=-2, high=2, size=(2))
        
        pos += noise
        
        pos[:,0] = np.clip(pos[:,0], a_min=-self.delta, a_max=self.delta+self.road_length)
        pos[:,1] = np.clip(pos[:,1], a_min=0, a_max=self.road_width)
        
        self.observation = (pos[:,0], pos[:,1])
        self.car_position = np.array([x_position[0], y_position[0]])        
        self.t+=1
    
    def find_bin(self, velocity):
        for i in range(self.Nv_bins):
            if velocity>=self.velocity_bins[i][0] and velocity<=self.velocity_bins[i][1]:
                return i
    
    def reward_func(self, action):
        beam_type = action[0]
        beam_number = action[1]
        
        angle_bounds = np.zeros(2)
        if beam_type==0:
            # Narrow beam
            angle_bounds[0] = self.beam_directions[beam_number] - self.narrow_beam_angle/2
            angle_bounds[1] = self.beam_directions[beam_number] + self.narrow_beam_angle/2
        else:
            # Broad beam
            angle_bounds[0] = self.beam_directions[beam_number] - self.broad_beam_angle/2
            angle_bounds[1] = self.beam_directions[beam_number] + self.broad_beam_angle/2
        
        angle = (180/np.pi)*np.arctan2(self.bs_location[1]-self.car_position[1], self.bs_location[0]-self.car_position[0])
        if angle>=angle_bounds[0] and angle<=angle_bounds[1]:
            r = np.sqrt((self.bs_location[0]-self.car_position[0])**2+(self.bs_location[1]-self.car_position[1])**2)
            if beam_type==0:
                power_avg = -2*np.log10(r)+self.power_bias
            else:
                power_avg = -2*np.log10(r)-np.log10(self.beam_angle_ratio)+self.power_bias
                
            power = np.random.normal(loc=power_avg, scale=power_avg)
            power = max(0, power)
            
            return power, 1
        else:
            return 0, 0