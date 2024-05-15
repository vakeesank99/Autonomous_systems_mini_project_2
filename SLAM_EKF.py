import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

N_STATE = 3 # state variables
SENSOR_RANGE = 3 # radius in meter

# ---> Landmark parameters
# Single landmark
# land_mark_arr = [(7,7)] # position from lower-left corner of map (m,m)

# Scattered Landmarks
land_mark_arr=[ (7,1),
                (6,2),
                (5,3),
                (6,6),
                (0,8)   
                  ]

# Rectangular pattern
# land_mark_arr = [(4,4),
#              (4,8),
#              (8,8),
#              (12,8),
#              (16,8),
#              (16,4),
#              (12,4),
#              (8,4)]
N_LANDMARKS = len(land_mark_arr)
M_WIDTH=N_STATE+2*N_LANDMARKS #width of the matrices

R = np.diag([0.002,0.002,0.0005]) # sigma_x, sigma_y, sigma_theta
Q = np.diag([0.003,0.005]) # sigma_r, sigma_phi

mu = np.zeros((M_WIDTH,1)) 
sigma = np.zeros((M_WIDTH,M_WIDTH)) 

mu[:] = np.nan 
np.fill_diagonal(sigma,100) 
# for i in range(3,width):
#     sigma0[i][i]=np.inf #high uncertainity 


Fx = np.block([[np.eye(3),np.zeros((N_STATE,2*N_LANDMARKS))]]) 

class Robo:
    """
    Used to initialized the robo instance
        - position of the robo is needed    
    """
    def __init__(self,pos):
        """
        position should be in 1x3 matrix
            - x,y, theta        
        """
        self.pos = pos
    def move(self,ut,dt):
        """
        Move the x,y coordinate of the robot according to the 
        linear velocity -v and angular velocity -w additionally 
        time period between two states are given in dt
        """
        v=ut[0]
        w=ut[1]
        theta=self.pos[2]
        arr= [ v/w*np.sin(theta+w*dt)-v/w*np.sin(theta), -v/w*np.cos(theta+w*dt)+v/w*np.cos(theta), w*dt]
        self.pos= self.pos+ np.array(arr,dtype=float)
    def sense(self,landmarks):
        """
        array of the landmarks given to find whether there in 
        the sensing range if then
        return distance , angle w.r.t robo and land mark id 
        """
        rx, ry, rtheta = self.pos
        zs = [] 
        for (lidx,landmark) in enumerate(landmarks): 
            lx,ly = landmark
            dist = np.linalg.norm(np.array([lx-rx,ly-ry])) 
            phi = np.arctan2(ly-ry,lx-rx) - rtheta 
            phi = np.arctan2(np.sin(phi),np.cos(phi)) 
            if dist<SENSOR_RANGE:
                zs.append((dist,phi,lidx))
        return zs

class EKF:
    """
    Used to keep track the mu and sigma 
    for the SLAM and EKF integration 
    """
    def __init__(self,mu,sigma):
        """
        mu: state transition matrix 
        sigma: covariance matrix
        """
        self.mu=mu
        self.sigma=sigma

    def predict(self,u,dt):
        """
        predict the mu and sigma using the control input u
        and dt . u accomodates the velocity and angular velocity
        """
        theta = self.mu[2]
        v,w = u[0],u[1]
        yt = np.zeros((N_STATE,1)) 
        yt[0] = v/w*np.sin(theta+w*dt)-v/w*np.sin(theta) if w>0.01 else v*np.cos(theta)*dt 
        yt[1] = -v/w*np.cos(theta+w*dt)+v/w*np.cos(theta) if w>0.01 else v*np.sin(theta)*dt 
        yt[2] = w*dt 
        self.mu = self.mu + Fx.T@yt 
        # noise_mu = 0
        # mu_t = np.add(yt , np.random.normal(noise_mu,noise_sigma,(width,width)))

        p = np.zeros((3,3)) 
        p[0,2] = vt/wt*np.cos(theta+wt*dt)-vt/wt*np.cos(theta) if w>0.01 else -v*np.sin(theta)*dt 
        p[1,2] = vt/wt*np.sin(theta+wt*dt)-vt/wt*np.sin(theta) if w>0.01 else v*np.cos(theta)*dt 
        G = np.identity(M_WIDTH) + Fx.T@p@Fx 
        noise_sigma = Fx.T@R@Fx 
        self.sigma = G@self.sigma@G.T + noise_sigma

    def update(self,zs):
        """
        update the mu and sigma according to the measurements
        of the sensor. 
        """
        rx,ry,theta = self.mu[0,0],self.mu[1,0],self.mu[2,0] # robot 
        for z in zs:
            (dist,phi,lidx) = z
            mu_landmark = self.mu[N_STATE+lidx*2:N_STATE+lidx*2+2] 
            if np.isnan(mu_landmark[0]): 
                mu_landmark[0] = rx + dist*np.cos(phi+theta) 
                mu_landmark[1] = ry+ dist*np.sin(phi+theta) 
                self.mu[N_STATE+lidx*2:N_STATE+lidx*2+2] = mu_landmark 
            delta  = mu_landmark - np.array([[rx],[ry]]) 
            q = np.linalg.norm(delta)**2 

            dist_est = np.sqrt(q) 
            phi_est = np.arctan2(delta[1,0],delta[0,0])-theta; phi_est = np.arctan2(np.sin(phi_est),np.cos(phi_est)) 
            z_est_arr = np.array([[dist_est],[phi_est]]) 
            z_act_arr = np.array([[dist],[phi]]) 
            delta_zs= z_act_arr-z_est_arr 

            Fxj = np.block([[Fx],[np.zeros((2,Fx.shape[1]))]])
            Fxj[N_STATE:N_STATE+2,N_STATE+2*lidx:N_STATE+2*lidx+2] = np.eye(2)
            H = np.array([[-delta[0,0]/dist_est,-delta[1,0]/dist_est,0,delta[0,0]/dist_est,delta[1,0]/dist_est],\
                        [delta[1,0]/q,-delta[0,0]/q,-1,-delta[1,0]/q,+delta[0,0]/q]])
            H = H@Fxj
            K =self.sigma@H.T@np.linalg.inv(H@self.sigma@H.T + Q) 
    
            self.mu = self.mu + K@delta_zs
            self.sigma = (np.identity(M_WIDTH) - K@H)@self.sigma


def show_robot_estimate(mu,sigma):
    '''
    find the eigen vales and draw the elipses
    for the robot
    '''
    rx,ry = mu[0],mu[1]
    eigenvals,angle = find_eigen(sigma[0:2,0:2]) 
    sigma_pixel = eigenvals[0], eigenvals[1] 
    show_uncertainty_ellipse((rx,ry),sigma_pixel,angle) 
    
def show_landmark_estimate(mu,sigma):
    '''
    find the eigen vales and draw the elipses
    for the observed landmarks
    '''
    for lidx in range(N_LANDMARKS): # For each landmark 
        lx,ly = mu[N_STATE+lidx*2], mu[N_STATE+lidx*2+1]
        lsigma=sigma[N_STATE+lidx*2:N_STATE+lidx*2+2,N_STATE+lidx*2:N_STATE+lidx*2+2]
        if ~np.isnan(lx): # if land mark found
            eigenvals,angle = find_eigen(lsigma) 
            # if np.max(eigenvals)<500: # threshold 
            sigma_pixel = eigenvals[0], eigenvals[1]
            show_uncertainty_ellipse((lx,ly),sigma_pixel,angle) 


def find_eigen(sigma):
    '''
    from the covariance matrix it wil find the eigen of it
    and return the angle as well
    '''
    [eigenvals,eigenvecs] = np.linalg.eig(sigma) 
    angle = 180.*np.arctan2(eigenvecs[1][0],eigenvecs[0][0])/np.pi 
    return eigenvals, angle


def show_uncertainty_ellipse(center,width,angle):
    '''
    draw the elipse on the axes which already in plotted stage
    '''
    ellipse = Ellipse(
        xy=center,
        width=5*width[0],
        height=5*width[1],
        angle=angle,
        facecolor="none",
        edgecolor="b"
    )
    ax.add_patch(ellipse)


if __name__ == '__main__':
    #simulation param
    dt = 1
    MAX_SIM_TIME= 70
    sim_time=0
    # intial pos for robot
    x_init = np.array([6,8,np.pi]) 
    robo = Robo(x_init)
    
    mu[0:3] = np.expand_dims(x_init,axis=1)
    sigma[0:3,0:3] = 0.1*np.eye(3)
    sigma[2,2] = 0 

    ekf = EKF(mu,sigma)
    EPSILON = 1E-9
    vt= 0.4 # ms-1
    wt = 0.1 +EPSILON#rad s-1
    u= np.array((vt,wt),dtype=float)

    # history
    his_predict = mu[0:N_STATE]
    his_true = his_predict
    # hxDR = xTrue

    fig, ax = plt.subplots()
    x_val = [x[0] for x in land_mark_arr]
    y_val = [x[1] for x in land_mark_arr]
    plt.scatter(x_val,y_val) #draw the all land marks

    while MAX_SIM_TIME >= sim_time:
        plt.cla()
        time.sleep(0.1) 
        sim_time += dt 
        robo.move(u,dt) # move the robot by control u
        zs = robo.sense(land_mark_arr) #observed land marks
        #ekf steps
        ekf.predict(u,dt) 
        ekf.update(zs) 
        #current robo position
        rx,ry,rt = robo.pos
        plt.plot(rx,ry,".r") # plotting the robo
        ax.set(xlim=(-50, 50),  ylim=(-50, 50) ) #yticks=np.arange(1, 10) xticks=np.arange(-50, 50),
        for z in zs: 
            dist,theta,lidx = z 
            lx,ly = rx +dist*np.cos(theta+rt),ry+dist*np.sin(theta+rt) 
            plt.plot([rx, lx], [ry, ly], '-g') #plot the observed land marks during the movement

        show_robot_estimate(ekf.mu,ekf.sigma) #robot uncertainity elipse
        show_landmark_estimate(ekf.mu,ekf.sigma) #land marks uncertainity elipse

        his_predict = np.hstack((his_predict, mu[0:N_STATE]))


        # hxDR = np.hstack((hxDR, xDR))
        his_true = np.hstack((his_true, [[rx],[ry],[rt]]))
        # plt.plot(xEst[0], xEst[1], ".r")
        plt.plot(his_true[0, :], his_true[1, :], "--r")
        # plt.plot(hxDR[0, :],
        #             hxDR[1, :], "-k")
        plt.plot(his_predict[0, :], his_predict[1, :], "-b")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)
        plt.savefig(r'images2/foo%s.png'%sim_time)





