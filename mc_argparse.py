"""
    A class that contains all member functions required to perform the determination of thermodynamic parameters
    with variable chemical potential, in the graphite lattice.

    inits:
    :self.n_iterations: Renamed to make more descriptive and easier to find. Previously n. Command line argument 1.
    :self.q_relaxation: Renamed from q, for the same reasons as n. Command line argument 2.
    :self.L: size of lattice LxLxP (L lateral size of triangular lattice).
    :self.P: number of planes (default = 4)
    :self.binsize: Number of iterations which thermodynamic averaging will be performed after.
    :self.defect_proportion: Proportion of defects within the lattice.
    :self.no_inplane: number of in plane neighbours.
    :self.no_outofplane: Number of second nearest neighbours
    :self.inplane: csv file of coordinates of in-plane neighbours
    :self.outofplane: csv file of coordinates of second nearest neighbours
    :self.lattice: Initialises the starting empty lattice of size layersxdimxdim.
    :self.occ1: Occupation number of sublattice 1
    :self.occ2: Occupation number of sublattice 2
    :self.uint: Interaction energy per lattice site
    :self.utotal: Total energy per lattice site
    :self.avg: Thermodynamic average for a specified variable
    :self.sd: Standard deviation for a specified variable

    Function: fix_sites, fixes alternating sites to represent the FCC structure and aren't included in MC steps.
    Returns: Updates 4 dimensional array with alternating sites with a fixed '2' value.

    Function: distribute_defects, distributes defects, fixing lithium sites as a '3' value to signify those fixed by
    defects
    Returns: Update 4 dimensional array with fixed '3' value sites.

    Function: Ulattice, calculates the internal energy of the total lattice.
    Returns: A tuple of the twos values: Total energy of the lattice, Total interaction energy terms of the lattice

    Function: Sublattice_1: Calculates the occupancy of alternate sites in the lattice for sublattice 1
    Returns: The occupancy of the lattice

    Function: Sublattice_2: Calculates the occupancy of alternate sites in the lattice for sublattice 2
    Returns: The occupancy of the lattice

    Function: Hamiltonian: Calculates the Energy Hamiltonian for a change of a binary state in a randomly selected site
    :param: a, b, c:    Randomly generated numbers for the indexing of the selected site
    :param: mu:      Chemical potential value of the system
    Returns: Delta H - the energy change if the site were to change state, float value.

    Function: monte_carlo: Runs a Monte Carlo simulation of lithium intercalation into LixMn2O4.
    :param mu:      Chemical potential value of the system
    Returns: Updated lattice after the Monte Carlo algorithm

    Function: write_file: creates results directory, if it doesn't exist already, and outputs csv of all parameters.
    Returns: csv file. Numbers in file: Run number.

    Function: standard-dev, calculates the thermodynamic average and the standard deviation for an inputted variable.
    :param: array: An array of values recorded every certain number of iterations for a specified variable
    Returns: Two float values, one for the average and one for the standard deviation

    Function: thermo_averaging: calculation of thermodynamic averages.
    Returns: Prints all of the necessary averages into a spreadsheet for data analysis.
    Also returns a dictionary, whose key is the chemical potential and value is a list of all other parameters.
"""

from __future__ import division
from sys import argv
import argparse
import sys
import numpy as np
import random, os, csv, logging, uuid
import pickle as pkl
from time import strftime
import errno

class MonteCarlo:
    def __init__(self):   # This part of the code is run when the class is instantiated.
        self.parser = argparse.ArgumentParser(description='MC code arguments')
        self.energy_params=['E0','g','delta1']
        self.parser.add_argument('--mu_ranges',type=float,help='Mu values to calculate',default=[-0.300,0.000,0.005],nargs='+')
        self.parser.add_argument('--E0', type=float, help='Point term, in kT', default= -4.5051789)
        self.parser.add_argument('--g', type=float, help='In plane interaction term', default= -0.458199)
        self.parser.add_argument('--delta1', type=float, help='Interaction between planes.', default= 1.106435)
        self.parser.add_argument('--alpha4', type=float, help='Exponential amplitude on E0', default= 0.000000)
        self.parser.add_argument('--beta4', type=float, help='Exponential decay constant on E0.', default= 0.00000)        
        self.parser.add_argument('--n_iterations',type=int, help='Number of iterations.', default=1000000)
        self.parser.add_argument('--q_relaxation',type=int, help='Relaxation steps before calculating averages.',default=0)
        self.parser.add_argument('--L',type=int, help='Length of triangular lattice.',default=36)
        self.parser.add_argument('--P',type=int, help='Number of graphite planes.',default=2) # Note: to be equivalent to Maxi, with the existing data structur, my P value should be halved!        
        self.parser.add_argument('--T', type=float, help='Temperature in Kelvin',default=293.0)
        self.parser.add_argument('--T_readin',type=str,help='Temperature value read in from file', default=None) 
        self.parser.add_argument('--number_readin',type=str,help='Number of directory to read from',default=None)
        self.parser.add_argument('--parallel_dir',type=str,help='Location to save files after parallel runs.',default=None)
        self.parser.add_argument('--binsize',type=int,help='Frequency of averaging',default=2000)
        self.parser.add_argument('--hec', type=str,help='Indicates if resume a serial or set of parallel runs from input file.',default='hec_s')
        self.parser.add_argument('--serial_dir',type=str,help='Number of directory to save files into.',default=None)
        self.parser.add_argument('--readin_id',type=int,help='Index of files to read in from.',default=0)
        self.args = self.parser.parse_args(argv[1:])
        self.arg_dict = dict(vars(self.args))                    
        self.e = 1.60218E-19 # electronic charge, in J 
        self.avogadro = 6.02214086E+23 # Avogadro number.
        self.k_B = 1.38064852E-23 # Boltzmann constant.
        self.no_inplane = 6                 # Number of nearest neighbours
        self.no_outofplane = 2                 # Number of next nearest neighbours                    
#        self.n_iterations = int(argv[1])    # Renamed to make more descriptive and easier to find. Previously n.
#        self.q_relaxation = int(argv[2])    # Renamed from q, for the same reasons as n.
#        self.dim = int(argv[3])
        self.E0 = self.arg_dict['E0']
        self.g = self.arg_dict['g']
        self.delta1 = self.arg_dict['delta1']
        self.alpha4 = self.arg_dict['alpha4']
        self.beta4 = self.arg_dict['beta4']
        self.binsize = self.arg_dict['binsize']
        self.L = self.arg_dict['L']
        self.P = self.arg_dict['P']
        self.T = self.arg_dict['T']
        self.kT = self.k_B * self.T # Boltzmann factor in J.
        self.n_iterations = self.arg_dict['n_iterations']
        self.q_relaxation = self.arg_dict['q_relaxation']
        self.mu_ranges = self.arg_dict['mu_ranges']
        self.mu_min,self.mu_max,self.mu_inc = self.mu_ranges
        self.readin_id = self.arg_dict['readin_id']
        
        if self.arg_dict['hec'] is not None:
            self.input_path='/home/hpc/34/mercerm1/mc_graphite/input/'
            self.output_path='/storage/hpc/34/mercerm1/mc_graphite/output/'
        if self.arg_dict['T_readin'] is not None:
            self.T_readin=self.arg_dict['T_readin']
            self.number_readin=self.arg_dict['number_readin']
            self.local_readin_dir=str(self.T_readin)+'/'+str(self.number_readin)+'/'
            self.lattice_input_array()            
        if self.arg_dict['parallel_dir'] is not None:
            self.parallel_dir=self.arg_dict['parallel_dir']
#            os.mkdir(self.output_path+self.parallel_dir)	

        self.active_sites = 2*self.L*self.L*self.P  # Number of Lithium sites within the lattice. In data structure this actually correponds to twice the number of planes!
#        self.zero_energy = self.active_sites * (-self.epsilon + 0.5 * self.no_nearest * self.j1 +  0.5 * self.no_second * self.j2) # Sets energy scale (in kT) to cope with -1, +1 structure. Extra 0.5 to account for double counting of nn.
        self.M = self.active_sites / 2 # Particles on each sublattice!
        
        self.file_paths()
        self.inplane1 = np.genfromtxt(self.input_path + 'ip1.csv', dtype=np.int, delimiter=',')
        self.outofplane1 = np.genfromtxt(self.input_path + 'oop1.csv', dtype=np.int, delimiter=',')
        self.inplane2 = np.genfromtxt(self.input_path + 'ip2.csv', dtype=np.int, delimiter=',')
        self.outofplane2 = np.genfromtxt(self.input_path + 'oop2.csv', dtype=np.int, delimiter=',')        
        self.lattice = np.ones((2,self.L, self.L, self.P), dtype=np.float64)  # Initialises starting filled lattice. First index (0,1)    
        self.lattice *= -0.5 # Empties the lattice and converts to spin 1/2.
        self.occ1 = 0
        self.occ2 = 0
        self.total_energy = 0
        self.int_energy = 0
#        self.uint = 0
#        self.utotal = 0
        self.sample_no = int((self.n_iterations - self.q_relaxation)/self.binsize)
        self.headers = ['Chemical Potential (eV)', 'Electrode Potential (V)', 'Sublattice 1', 'Sub1sd', 'Sublattice 2', 'Sub2sd',
                        'N', 'N_sd', 'NN', 'NN_sd', 'U', 'U_sd', 'UN', 'UN_sd', 'UIE', 'UIE_sd', 'covUN', 'varNN',
                        'dU', 'dS']
        self.current_time = strftime("%c")
        self.file_header = 'Run on : ' + str(self.current_time) + '\n' +\
                           'No. of iterations : ' + str(self.n_iterations) + '\n' +\
                           'Relaxation steps : ' + str(self.q_relaxation) +'\n' +\
                           'Number of averages : ' + str(self.sample_no) + '\n' +\
                           'Temperature : ' + str(self.T) + '\n' +\
                           'Lateral size of lattice L : ' + str(self.L) +'\n' +\
                           'Number of planes P : '+ str(self.P) + '\n' +\
                           'E0 : ' + str(self.E0) + ' g : ' + str(self.g) + ' delta1 : ' + str(self.delta1) + '\n' +\
                           'alpha4 : ' + str(self.alpha4) + ' beta4 : ' + str(self.beta4) + '\n\n' 

        
        self.input_variables = [self.current_time, self.n_iterations, self.L, self.P, self.q_relaxation, self.T]
        self.file_data = []  # Empty list into which output data is placed.

    def check_path_exists(self,path):
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
        
    def file_paths(self):
        self.unique_identifier=str(uuid.uuid1())[0:8]
        self.check_path_exists(self.input_path)
        self.check_path_exists(self.output_path)       

        if self.arg_dict['hec'] == 'hec_s':
            self.first_dir_level=self.output_path+'serial'+'/'
        elif self.arg_dict['hec'] == 'hec_p':
            self.first_dir_level=self.output_path+'parallel'+'/'

        self.second_dir_level=self.first_dir_level+str(self.T)+'/'

        if self.arg_dict['parallel_dir'] is None and self.arg_dict['serial_dir'] is None:
            counter=0
            proto_dir=self.second_dir_level+str(counter)+'/'
            while os.path.exists(proto_dir):
               proto_dir=self.second_dir_level+str(counter)+'/'
               counter+=1
            self.third_dir=proto_dir
        elif self.arg_dict['parallel_dir'] is not None:
            self.third_dir=self.second_dir_level+self.arg_dict['parallel_dir']+'/'
        elif self.arg_dict['serial_dir'] is not None:
            self.third_dir=self.second_dir_level+self.arg_dict['serial_dir']+'/'

        self.final_dir=self.third_dir+self.unique_identifier+'/'
        self.output_filename = self.final_dir+'L_%d_P_%d_temp_%d.csv' % (self.L,self.P,self.T)    
        self.output_file_path=self.final_dir
        self.check_path_exists(self.final_dir)
        print self.output_file_path
        
    def lattice_input_array(self):
    	self.readin_dir=self.output_path+'serial/'+self.local_readin_dir
    	unique_id=os.listdir(self.readin_dir)[self.readin_id]
        print 'unique_id=', unique_id
        self.full_readin=self.readin_dir+unique_id+'/'
        print 'self.full_readin=',self.full_readin
        file_list=os.listdir(self.full_readin)
        print 'file_list=',file_list
    	lattice_input_dict={}
    	for f in file_list:
    	    if f.startswith('lattice'):
    	        mu_value=f.split('_')[-1]
    	        lattice_input_dict[mu_value]=f
    	self.locator_dict=dict((v,k) for k,v in lattice_input_dict.iteritems())        
    	self.master_dict={}
    	for filename in self.locator_dict:
    	    file_path=self.full_readin + filename	
    	    mu_value=float(filename.split('_')[-1])
    	    try:
    	        with open(file_path, 'rb') as f:	    
    	            lattice_array=pkl.load(f)
    	    except:
    	        raise
    	    self.master_dict[mu_value]=lattice_array
            
    def ulattice(self, mu, mode = 1):
        self.total_energy = 0
        self.int_energy = 0

        # Mode = -1 will start from a filled lattice and count the empty sites.
        if mode == 1:
            self.occ1 = 0
            self.occ2 = 0
        else:
            self.occ1 = int(self.active_sites/2)
            self.occ2 = int(self.active_sites/2)

        for j in range(self.L):
            for k in range(self.L):
                for l in range(self.P):
                    nnsum = 0 # Refreshes counters for each unit cell.
                    nnnsum = 0
                    for i in (0,1):
                        deltaE1 = 0
                        deltaE2 = 0
                        site_occ = self.lattice[i,j,k,l]
                        if i == 0 and site_occ == 0.5:
                            self.occ1 += 1 * mode
                        elif i == 1 and site_occ == 0.5:
                            self.occ2 += 1 * mode

                        if i == 0:    
                            neighs = (self.inplane1,self.outofplane1)
                        else:    
                            neighs = (self.inplane2,self.outofplane2)                            
                        for element in neighs[0]:
                            deltaE1 += (self.lattice[element[0], (j + element[1]) % self.L, (k + element[2]) % self.L, (l + element[3]) % self.P]) * site_occ * self.g

                        for element in neighs[1]:
                            deltaE2 += (self.lattice[element[0], (j + element[1]) % self.L, (k + element[2]) % self.L, (l + element[3]) % self.P]) * site_occ * self.delta1
#  Calculation of the Total energy and interaction energy hamiltonian
#  Interaction terms have a factor of a half to account for double counting
                        self.total_energy += (deltaE1 + deltaE2)/2.
                        self.int_energy += (deltaE1 + deltaE2)/2.

        self.x = (self.occ1 + self.occ2) / (2*self.M) # Lattice occupation
        self.N = self.x * self.active_sites # Occupancy of lattice.
        self.E0prime = self.E0 + self.alpha4 * np.exp(-self.beta4 * self.x)
        self.total_energy += (self.E0prime - mu) * self.N # Check this!
        self.int_energy += self.E0prime * self.N

        print 'Energy initialised at:', self.total_energy

    def hamiltonian(self, a, b, c, d, mu):
        ipsum = 0 # In plane (g) sum.
        oopsum = 0  # Out of plane (delta1) sum.
        site = self.lattice[a, b, c, d]  # Setting the selected site

        occ_change = -site * 2 # Minus sign because we assume that the site will change in the energy expression. 2 for spin 1/2 -1/2 data structure.
        self.xold = (self.occ1+self.occ2)/(2 * self.M)
        self.xnew = self.xold + occ_change/self.active_sites
        self.E0prime_o = self.E0 + self.alpha4 * np.exp(-self.beta4 * self.xold)
        self.E0prime_n = self.E0 + self.alpha4 * np.exp(-self.beta4 * self.xnew)

        if a == 0:
            neighs = (self.inplane1, self.outofplane1)
        else:
            neighs = (self.inplane2, self.outofplane2)
            
        for element in neighs[0]:
            ipsum += (self.lattice[element[0], (b + element[1]) % self.L,
                                   (c + element[2]) % self.L, (d + element[3]) % self.P])

        for element in neighs[1]:
            oopsum += (self.lattice[element[0], (b + element[1]) % self.L,
                                    (c + element[2]) % self.L, (d + element[3]) % self.P])
            
        self.trial_u = - 2 * (self.g * ipsum + self.delta1 * oopsum + 0.5 * self.E0prime_o + 0.5 * self.E0prime_n) * site
        self.trial_change = self.trial_u + 2 * mu * site   # Hamiltonian change calculated for sites current state. 2 to account for going from +1/2 to -1/2


    def monte_carlo(self, mu):
        a = random.randint(0, 1)                    # Chooses either the 0 or 1 'sublattice'
        b = random.randint(0, (self.L-1))      # Generates random numbers for selection of random sites
        c = random.randint(0, (self.L-1))
        d = random.randint(0, (self.P-1))
        
        site = int(self.lattice[a, b, c, d] * 2)            # Selects random spin site. Acounts for spin 1/2.
        self.hamiltonian(a, b, c, d, mu)  # Calls Hamiltonian function, calculates delta H for the site

        if self.trial_change <= 0:
            site *= -1
            self.total_energy += self.trial_change
            self.int_energy += self.trial_u
            if a == 0:
                self.occ1 += site
            else:
                self.occ2 += site
        else:
            e = random.random()                         # Random number generated for comparison between 0 and 1
            p = np.exp(-self.trial_change)     # Probability of spin changing. p is in units of kT, so no Boltzmann factor denominator is needed!

            if e < p:                               # Comparison of random number and probability
                site *= -1
                self.total_energy += self.trial_change
                self.int_energy += self.trial_u 
                if a == 0:
                    self.occ1 += site
                else:
                    self.occ2 += site

        self.lattice[a,b,c,d] = site / 2. # Reassigns lattice site and goes to spin 1/2.

    def write_file(self):
        self.csv_name=self.output_file_path+self.unique_identifier+'_'+str(self.T)+'.csv'
        print self.csv_name
        with open(self.csv_name, 'wb') as csvfile:
            file_object = csv.writer(csvfile, delimiter=',', quotechar='"')
            file_object.writerow([self.file_header])
            file_object.writerow('')
            file_object.writerow(self.headers)
            for line in self.file_data:
                file_object.writerow(line)

    def standard_dev(self, array):
        self.avg = np.sum(array)/(self.sample_no - 1)  # Calculates mean for the data set

        self.sd = np.sqrt(np.sum((array - self.avg)**2))/(self.sample_no - 1)  # Calculates the standard deviation for the data set

    def thermo_averaging(self):
        with open(self.output_file_path + 'status_temp_%d' % self.T, 'w') as f:
            f.write(self.file_header + '\n')
            f.write('\n')
            
        ''' Start of Monte Carlo loop'''
        if self.arg_dict['T_readin'] is None:
            a = np.arange(self.mu_min, self.mu_max, self.mu_inc)
            mu_range = a
            print mu_range
        else:
            mu_range = sorted([float(key) for key in self.master_dict.keys()])
        
        for mu in mu_range:
#            self.energy_init    # Iterates over a range of chemical potential values
#            if mu > -4.1:
#                mode = -1
#            else:
#                mode = 1
  
#            self.ulattice()
            self.mu = mu * self.e / self.kT # Converts working version of mu to kT units
            if self.arg_dict['T_readin'] is not None: 	
                self.lattice = self.master_dict[mu] # Stored using eV mu value.
            self.ulattice(self.mu)

            with open((self.output_file_path + '_current_mu_T_%d') % self.T, 'w') as f, open((self.output_file_path + 'status_T_%d') % self.T, 'a+') as f_b:
                f.write('mu=%.3f' % mu)
                f.flush()

                '''Refreshes counters for thermodynamic averaging'''
                x1_arr = np.empty((self.sample_no - 1))
                x2_arr = np.empty((self.sample_no - 1))
                n_arr = np.empty((self.sample_no - 1))
                nn_arr = np.empty((self.sample_no - 1))
                uie_arr = np.empty((self.sample_no - 1))
                umcs_arr = np.empty((self.sample_no - 1))
                un_arr = np.empty((self.sample_no - 1))

                avg_no = 0

                for itt in xrange(0, self.n_iterations):        # Moved outside of the mc function to allow the therm. avging
                    self.monte_carlo(float(self.mu))  # Runs Monte Carlo algorithm for each chemical potential value. Mu is in kT.
                    '''Thermodynamic averaging'''
                    if itt > self.q_relaxation:
                        if itt % self.binsize == 0:  # Determines frequency of calculations
#                        self.ulattice()       # Calculate respective occupancies for sublattice 1 and 2
                        # Set 'sublattice 1' (x1) as max occupancy and 'sublattice 2' as the min occupancy lattices
                            x1 = (max(self.occ1, self.occ2)) / (0.5*self.active_sites)
                            x2 = (min(self.occ1, self.occ2)) / (0.5*self.active_sites)
                        # Calculate total occupancy of the lattice
                            n = (self.occ1 + self.occ2)/self.active_sites
                            nn = n*n
                            x1_arr[avg_no] = x1
                            x2_arr[avg_no] = x2
                            n_arr[avg_no] = n
                            nn_arr[avg_no] = nn

                            uie = (float(self.int_energy)/self.active_sites)  # Internal energy per site
                            umcs = (float(self.total_energy)/self.active_sites)  # Total energy per sites
                            uie_arr[avg_no] = uie
                            umcs_arr[avg_no] = umcs
                            un = uie * n
                            un_arr[avg_no] = un
                            avg_no += 1

                            if itt % (self.binsize * 100) == 0:
                                f_b.write('mu=%.3f, itt=%d x1=%d, x2=%d, Etot=%.3f, U=%.3f, un=%.3f\n' % (mu, itt, self.occ1, self.occ2, self.total_energy, self.int_energy, un))
                                f_b.flush()
                                
            
            if self.arg_dict['hec'] == 'hec_s':
                with open(self.output_file_path +  ('lattice_T_%d_mu_%.3f') % (self.T, mu), 'wb') as f:
                    pkl.dump(self.lattice, f)       

            '''Final calculation of averages for a specific chemical potential'''
            ep = mu*(-1)                    # Chemical potential converted to the electrode potential
            print 'EP = ', ep, ' final avg no = ', avg_no, 'sample_no = ', self.sample_no
            x1sd = np.std(x1_arr)
            X1 = np.mean(x1_arr)
            x2sd = np.std(x2_arr)
            X2 = np.mean(x2_arr)
            Nsd = np.std(n_arr)
            N = np.mean(n_arr)
            NNsd = np.std(nn_arr)
            NN = np.mean(nn_arr)
            UIEsd = np.std(uie_arr)
            UIE = np.mean(uie_arr)
            Usd = np.std(umcs_arr)
            U = np.mean(umcs_arr)
            UNsd = np.std(un_arr)
            UN = np.mean(un_arr)

            print 'un_arr', un_arr

            covUN = (UN - (UIE*N))            # Calculation of cov(UN) and Var(N) from definition
            varNN = (NN - (N*N))

            dU_1 = covUN / varNN # All in units of kT!
            dU_kT = dU_1 
            mu_kT = mu * self.e / self.kT # mu is entered in eV!
            dS_kT = (1/self.T) * (dU_kT - mu_kT)  # Converts to J

            attributes = [mu, ep, X1, x1sd, X2, x2sd, N, Nsd, NN, NNsd, U, Usd, UN, UNsd, UIE, UIEsd, covUN, varNN, dU_kT, dS_kT]
            # Where the attributes are the chemical potential value, the electrode potential value, the occupancy of
            # sublattice 1 followed by the standard deviation, the occupancy of sublattice 2 followed by the standard
            # deviation, the total occupancy of the lattice (and sd), the value of the occupancy squared (and sd). The
            # total energy of the lattice (and sd) and then the interaction energy only of the lattice (and sd). Finally
            # there is the covariance of the UN values, the variance of N and then the calculated dU/dN and dS/dN values
            # in kJ/mol
            self.file_data.append(attributes)  # Attributes is a list of lists, will eventually be written to the file.

        self.write_file()
        #  print self.lattice
        #  return attribute_dictionary

if __name__ == '__main__':
    
    log_filename = 'logs/logfile.out'
    if not os.path.exists('logs'):
        os.mkdir('logs')

    logging.basicConfig(filename=log_filename, level=logging.DEBUG)
    logging.debug('Run on ' + str(strftime("%c")))

    try:
        mc = MonteCarlo()               # Instantiates the class
        mc.thermo_averaging()  # Runs the thermodynamic averaging function from the class.
    except:
        logging.exception('Exception raised on main handler')
        raise
